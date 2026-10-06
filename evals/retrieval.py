"""Retrieval benchmark: does hybrid search actually find the right tool?

Toolforge's reuse decision starts with retrieval: given a capability need, find the stored tool
that satisfies it. This measures that step in isolation, on a hand-labelled set
(``retrieval_set.json``): 32 tool cards, 50 needs with a known correct tool (written with
different vocabulary from the cards, and with near-miss distractors such as roman_to_int vs
int_to_roman), and 15 needs that NO tool satisfies (several are deliberate near-misses, e.g.
SHA-1 when only SHA-256 and MD5 exist), where reusing anything would be a bug.

Rankers compared, all over the same cards:
  * bm25    - lexical only
  * dense   - embedding cosine only
  * hybrid  - both, fused with Reciprocal Rank Fusion (what Toolforge uses)

Reported per ranker: hit@1, hit@3 and MRR on the positives. Then the reuse GATE (the calibrated
cosine thresholds that decide auto-reuse / ask the judge / create) is scored on positives and
negatives, and the thresholds that would best separate them on this set are suggested.

    python -m evals.retrieval                       # default offline hashing embedder
    python -m evals.retrieval --embedder nvidia     # neural embeddings (needs NVIDIA_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolforge.config import Settings  # noqa: E402
from toolforge.embeddings import Embedder, make_embedder, tokenize  # noqa: E402
from toolforge.models import Need, ToolDraft  # noqa: E402
from toolforge.retrieval import BM25, Doc, hybrid_search  # noqa: E402

HERE = Path(__file__).parent
TOP_K = Settings().top_k  # the agent shows its judge the hybrid top-k
RESULTS = HERE / "results"


def load_set(path: Path = HERE / "retrieval_set.json") -> tuple[list[ToolDraft], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = [ToolDraft(name=t["name"], description=t["description"], parameters=t["parameters"],
                       code=f"def {t['name']}():\n    pass\n") for t in data["tools"]]
    return tools, data["queries"]


SWEEP = (1.0, 0.5, 0.25, 0.1)  # BM25 weight in RRF (dense weight fixed at 1.0)


def rank_all(embedder: Embedder, tools: list[ToolDraft], queries: list[dict[str, Any]],
             lexical_weight: float | None = None) -> dict[str, Any]:
    lw = embedder.lexical_weight if lexical_weight is None else lexical_weight
    texts = [t.search_text() for t in tools]
    names = [t.name for t in tools]
    doc_vecs = embedder.embed(texts, "passage")
    q_texts = [Need(name_hint=q["name_hint"], description=q["description"]).query() for q in queries]
    q_vecs = embedder.embed(q_texts, "query")
    bm25 = BM25([tokenize(t) for t in texts])
    docs = [Doc(n, t, v) for n, t, v in zip(names, texts, doc_vecs)]

    out = []
    for q, qt, qv in zip(queries, q_texts, q_vecs):
        cos = doc_vecs @ qv
        lex = bm25.scores(tokenize(qt))
        dense_order = [names[i] for i in np.argsort(-cos)]
        bm25_order = [names[i] for i in np.argsort(-lex, kind="stable")] if lex.max() > 0 else []
        hybrid_hits = hybrid_search(qt, qv, docs, k=len(docs), lexical_weight=lw)
        hybrid_order = [h.key for h in hybrid_hits]
        sweep = {f"hybrid@{w}": [h.key for h in hybrid_search(qt, qv, docs, k=len(docs), lexical_weight=w)]
                 for w in SWEEP}
        gold_cos = float(cos[names.index(q["gold"])]) if q["gold"] else None
        out.append({"query": qt, "gold": q["gold"], "top_cosine": float(cos.max()),
                    "top_tool": dense_order[0], "gold_cosine": gold_cos,
                    "orders": {"bm25": bm25_order, "dense": dense_order, "hybrid": hybrid_order, **sweep},
                    "hybrid_top_cosine": float(cos[names.index(hybrid_order[0])]),
                    "top_k": [(h.key, round(h.cosine, 4)) for h in hybrid_hits[:TOP_K]]})
    return {"rows": out, "lexical_weight": lw}


def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    pos = [r for r in rows if r["gold"]]
    metrics = {}
    for ranker in pos[0]["orders"] if pos else ():
        h1 = h3 = rr = 0.0
        for r in pos:
            order = r["orders"][ranker]
            if r["gold"] in order:
                rank = order.index(r["gold"]) + 1
                h1 += rank == 1
                h3 += rank <= 3
                rr += 1 / rank
        n = len(pos) or 1
        metrics[ranker] = {"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "mrr": round(rr / n, 3)}
    return metrics


def gate_decision(r: dict[str, Any], reuse_t: float, consider_t: float) -> tuple[str, str | None]:
    """Mirror of Forge.match: candidates are the hybrid top-k above the consider threshold; the one with
    the highest cosine is auto-reused at/above the reuse threshold, otherwise the LLM judge decides."""
    cands = [(k, c) for k, c in r["top_k"] if c >= consider_t]
    if not cands:
        return "create", None
    key, c = max(cands, key=lambda kc: kc[1])
    return ("auto_reuse", key) if c >= reuse_t else ("judge", None)


def gate_metrics(rows: list[dict[str, Any]], reuse_t: float, consider_t: float) -> dict[str, Any]:
    """Score the cosine gate alone (before any LLM judge)."""
    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    dp = [(r, *gate_decision(r, reuse_t, consider_t)) for r in pos]
    dn = [gate_decision(r, reuse_t, consider_t)[0] for r in neg]
    gold_shown = sum(1 for r, d, _ in dp if d == "judge" and r["gold"] in
                     [k for k, c in r["top_k"] if c >= consider_t])
    return {"reuse_threshold": reuse_t, "consider_threshold": consider_t,
            "lexical_weight": rows[0].get("lexical_weight") if rows else None,
            "positives": {"auto_reused_correctly": sum(1 for r, d, k in dp if d == "auto_reuse" and k == r["gold"]),
                          "auto_reused_wrong_tool": sum(1 for r, d, k in dp if d == "auto_reuse" and k != r["gold"]),
                          "sent_to_judge": sum(1 for _, d, _ in dp if d == "judge"),
                          "judge_shown_correct_tool": gold_shown,
                          "missed_would_duplicate": sum(1 for _, d, _ in dp if d == "create"), "total": len(pos)},
            "negatives": {"auto_reused_wrongly": dn.count("auto_reuse"), "sent_to_judge": dn.count("judge"),
                          "correctly_created": dn.count("create"), "total": len(neg)}}


def judge_metrics(embedder: Embedder, llm: Any, tools: list[ToolDraft], queries: list[dict[str, Any]],
                  settings: Settings, log=print, lexical_weight: float | None = None) -> dict[str, Any]:
    """End-to-end reuse decision: run the agent's REAL match node (retrieval + gate + LLM judge)."""
    import tempfile

    from toolforge.graph import Forge
    from toolforge.knowledge import Knowledge
    from toolforge.models import Verification
    from toolforge.registry import Registry
    from toolforge.sandbox import Sandbox

    with tempfile.TemporaryDirectory(prefix="tf-retrieval-", ignore_cleanup_errors=True) as tmp:
        registry = Registry(str(Path(tmp) / "lib.db"))
        knowledge = Knowledge(registry, embedder, lexical_weight=lexical_weight)
        vecs = embedder.embed([t.search_text() for t in tools], "passage")
        for t, v in zip(tools, vecs):
            registry.add_tool(t, v, origin_task="retrieval benchmark", verification=Verification())
        registry.set_meta("embedder", embedder.identity)
        forge = Forge(llm, knowledge, Sandbox(), settings)
        outcomes = []
        for i, q in enumerate(queries, 1):
            need = Need(name_hint=q["name_hint"], description=q["description"])
            out = forge.match({"task": q["description"], "needs": [need.model_dump()], "idx": 0, "trace": []})
            chosen = out["chosen"] if out["decision"] == "reuse" else None
            outcomes.append({"query": need.query(), "gold": q["gold"], "chosen": chosen,
                             "reason": out["trace"][-1]["reason"]})
            ok = chosen == q["gold"]
            log(f"  {i:>2}/{len(queries)} {'✓' if ok else '✗'} {q['name_hint']:<24} -> {chosen or 'create'}"
                + ("" if ok else f"   (wanted {q['gold'] or 'create'})"))
        registry.close()
    pos = [o for o in outcomes if o["gold"]]
    neg = [o for o in outcomes if not o["gold"]]
    return {"positives": {"reused_correct_tool": sum(o["chosen"] == o["gold"] for o in pos),
                          "reused_wrong_tool": sum(1 for o in pos if o["chosen"] and o["chosen"] != o["gold"]),
                          "created_duplicate": sum(1 for o in pos if not o["chosen"]), "total": len(pos)},
            "negatives": {"reused_wrongly": sum(1 for o in neg if o["chosen"]),
                          "correctly_created": sum(1 for o in neg if not o["chosen"]), "total": len(neg)},
            "decision_accuracy": round(sum(o["chosen"] == o["gold"] for o in outcomes) / len(outcomes), 3),
            "outcomes": outcomes}


def suggest_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    """consider: as high as possible while the judge is still shown the correct tool for ~all positives.
    reuse: just above the highest cosine any negative or wrong top tool reaches, so auto-reuse is never wrong."""
    gold_c = sorted(r["gold_cosine"] for r in rows if r["gold"])
    risky = [r["top_cosine"] for r in rows if not r["gold"]]
    risky += [r["top_cosine"] for r in rows if r["gold"] and r["top_tool"] != r["gold"]]
    consider = round(gold_c[max(0, int(len(gold_c) * 0.04) - 1)] - 0.01, 2) if gold_c else 0.2
    reuse = round(max(risky) + 0.01, 2) if risky else 0.9
    return {"reuse_threshold": max(reuse, round(consider + 0.05, 2)), "consider_threshold": consider}


def render(embedder: Embedder, ranking: dict[str, Any], gate: dict[str, Any], suggested: dict[str, float],
           gate_suggested: dict[str, Any], n_tools: int, n_pos: int, n_neg: int,
           judge: dict[str, Any] | None = None, model: str = "") -> str:
    lw = gate.get("lexical_weight", embedder.lexical_weight)
    lines = [f"## Retrieval benchmark: {n_tools} tools, {n_pos} needs with a correct tool, {n_neg} with none",
             f"Embedder: `{embedder.identity}`", "", "| ranker | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
    for name in ("bm25", "dense", "hybrid"):
        m = ranking[name]
        label = {"bm25": "BM25 (lexical)", "dense": "dense (embedding cosine)",
                 "hybrid": f"**hybrid: BM25 + dense, RRF** (BM25 weight {lw})"}[name]
        lines.append(f"| {label} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['mrr']:.3f} |")
    sweep = [(k, m) for k, m in ranking.items() if k.startswith("hybrid@")]
    if sweep:
        lines += ["", "Fusion sweep: BM25 weight in RRF (dense weight 1.0). Chosen on this same set, so treat a "
                  "small difference as noise.", "", "| BM25 weight | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
        lines += [f"| {k.split('@')[1]} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['mrr']:.3f} |" for k, m in sweep]

    def gate_lines(title: str, g: dict[str, Any]) -> list[str]:
        p, n = g["positives"], g["negatives"]
        return ["", f"**{title}** (auto-reuse ≥ {g['reuse_threshold']}, ask the judge ≥ "
                f"{g['consider_threshold']}, otherwise create)", "",
                "| | auto-reused | sent to judge | created |", "|---|---|---|---|",
                f"| needs with a correct tool ({p['total']}) | {p['auto_reused_correctly']} right, "
                f"{p['auto_reused_wrong_tool']} wrong | {p['sent_to_judge']} (correct tool shown to it: "
                f"{p['judge_shown_correct_tool']}) | {p['missed_would_duplicate']} (duplicate) |",
                f"| needs with no tool ({n['total']}) | **{n['auto_reused_wrongly']}** (wrong) | "
                f"{n['sent_to_judge']} | {n['correctly_created']} |"]

    lines += gate_lines("Reuse gate, shipped thresholds", gate)
    lines += gate_lines("Reuse gate, thresholds suggested by this set", gate_suggested)
    lines += ["", f"Suggested: `TOOLFORGE_REUSE_THRESHOLD={suggested['reuse_threshold']}`, "
              f"`TOOLFORGE_CONSIDER_THRESHOLD={suggested['consider_threshold']}`. Anything sent to the judge is "
              "still checked by an LLM before reuse; only auto-reuse skips that check."]
    if judge:
        p, n = judge["positives"], judge["negatives"]
        lines += ["", f"**End to end, with the LLM judge** (`{model}`, the agent's real match step, "
                  "shipped thresholds): "
                  f"decision accuracy **{judge['decision_accuracy']:.0%}**", "",
                  "| | reused the right tool | reused a wrong tool | built a new tool |", "|---|---|---|---|",
                  f"| needs with a correct tool ({p['total']}) | **{p['reused_correct_tool']}** | "
                  f"{p['reused_wrong_tool']} | {p['created_duplicate']} (duplicate) |",
                  f"| needs with no tool ({n['total']}) | — | **{n['reused_wrongly']}** | {n['correctly_created']} |"]
    return "\n".join(lines) + "\n"


def run(embedder: Embedder, set_path: Path | None = None, llm: Any = None, settings: Settings | None = None,
        log=print, lexical_weight: float | None = None) -> dict[str, Any]:
    tools, queries = load_set(set_path) if set_path else load_set()
    ranked = rank_all(embedder, tools, queries, lexical_weight)
    rows = ranked["rows"]
    for r in rows:
        r["lexical_weight"] = ranked["lexical_weight"]
    ranking = ranking_metrics(rows)
    gate = gate_metrics(rows, embedder.reuse_threshold, embedder.consider_threshold)
    suggested = suggest_thresholds(rows)
    gate_s = gate_metrics(rows, suggested["reuse_threshold"], suggested["consider_threshold"])
    judge = None
    if llm is not None:
        settings = settings or Settings()
        judge = judge_metrics(embedder, llm, tools, queries, settings, log=log, lexical_weight=ranked["lexical_weight"])
    n_pos = sum(1 for q in queries if q["gold"])
    model = f"{settings.provider}:{settings.model}" if settings else ""
    md = render(embedder, ranking, gate, suggested, gate_s, len(tools), n_pos, len(queries) - n_pos, judge, model)
    return {"embedder": embedder.identity, "ranking": ranking, "gate": gate, "suggested": suggested,
            "gate_suggested": gate_s, "judge": judge, "rows": rows, "markdown": md}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", help="hashing | nvidia | gemini | openai (default: TOOLFORGE_EMBEDDER)")
    ap.add_argument("--embed-model", help="embedding model id (default: the embedder's default)")
    ap.add_argument("--lexical-weight", type=float, help="BM25 weight in hybrid fusion (default: embedder's)")
    ap.add_argument("--probe", action="store_true", help="list which embedding models your key can actually use")
    ap.add_argument("--judge", action="store_true",
                    help="also run the agent's real match step with the LLM judge (~1 LLM call per need)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    settings = Settings()
    if args.embedder:
        settings.embedder = args.embedder
        if not args.embed_model and args.embedder.lower() != (Settings().embedder or "").lower():
            settings.embed_model = None  # TOOLFORGE_EMBED_MODEL belongs to the other embedder
    if args.embed_model:
        settings.embed_model = args.embed_model
    if args.probe:
        embedder = make_embedder(settings)
        if not hasattr(embedder, "probe"):
            print(f"--probe works for API embedders, not {settings.embedder!r}")
            return 1
        working = []
        for model_id, ok, detail in embedder.probe():
            print(f"{'OK  ' if ok else 'FAIL'} {model_id:<55} {detail}")
            working += [model_id] if ok else []
        if working:
            print(f"\nUse one of these, e.g.:\n  python -m evals.retrieval --embedder {settings.embedder} "
                  f"--embed-model {working[0]}\nand set TOOLFORGE_EMBED_MODEL={working[0]} in .env "
                  "to use it in the agent.")
        return 0 if working else 1
    llm = None
    if args.judge:
        from toolforge.llm import make_llm

        llm = make_llm(settings)
        print(f"judging {settings.provider}:{settings.model} ...")
    from toolforge.embeddings import EmbeddingError

    try:
        result = run(make_embedder(settings), llm=llm, settings=settings,
                     lexical_weight=args.lexical_weight if args.lexical_weight is not None else settings.lexical_weight)
    except EmbeddingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    RESULTS.mkdir(exist_ok=True)
    tag = settings.embedder.lower() + ("_judge" if args.judge else "")
    (RESULTS / f"retrieval_{tag}.json").write_text(json.dumps({k: v for k, v in result.items() if k != "markdown"},
                                                              indent=1), encoding="utf-8")
    (RESULTS / f"retrieval_{tag}.md").write_text(result["markdown"], encoding="utf-8")
    print(result["markdown"])
    misses = [r for r in result["rows"] if r["gold"] and r["orders"]["hybrid"][0] != r["gold"]]
    if misses:
        print("hybrid top-1 misses (retrieval only):")
        for r in misses:
            print(f"  {r['query'][:70]!r}: got {r['orders']['hybrid'][0]}, wanted {r['gold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
