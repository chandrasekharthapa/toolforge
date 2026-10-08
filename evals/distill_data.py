"""Build a dataset for distilling Toolforge's LLM reuse-judge into a small cross-encoder reranker.

Teacher: the agent's LLM judge (same prompt, same candidate format as ``Forge.match``).
Student: a cross-encoder that scores (need, tool card) pairs; trained in ``notebooks/train_reranker.ipynb``.

Leakage controls
  * Training tools are NEW tool cards written by the LLM, not the 32 tools of the hand-labelled
    retrieval set; generated cards too close to an evaluation tool are dropped (and reported).
  * Train/validation are split BY TOOL, so validation measures generalisation to unseen tools.
  * The test set is the hand-labelled 65-need set (human gold labels, unseen tools). The teacher also
    judges exactly those candidate lists, so teacher and student are compared on identical inputs.

Steps (each resumable; outputs in evals/distill/):
  1. tools   - generate ~50 tool cards across domains
  2. needs   - per tool: needs it satisfies + near-miss needs it does not
  3. label   - retrieve top-4 candidates per need; the teacher picks one or "create"
  4. export  - pairwise JSONL (query, card, label) + grouped decision files for evaluation

    python -m evals.distill_data              # all steps (~570 LLM calls; resumable)
    python -m evals.distill_data --export     # only rebuild the exported files
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.retrieval import load_set  # noqa: E402
from toolforge import prompts as P  # noqa: E402
from toolforge.config import Settings  # noqa: E402
from toolforge.embeddings import Embedder, make_embedder  # noqa: E402
from toolforge.llm import LLM, LLMConfigError  # noqa: E402
from toolforge.models import Need, ToolDraft  # noqa: E402
from toolforge.retrieval import Doc, hybrid_search  # noqa: E402

OUT = Path(__file__).parent / "distill"
TOP_K = 4
DOMAINS = [
    "text processing and string manipulation", "number theory and integer maths", "geometry and shapes",
    "unit and measurement conversion", "dates, times and calendars", "personal finance and loans",
    "statistics on lists of numbers", "encoding, hashing and checksums", "lists, sorting and searching",
    "colours, formatting and parsing", "physics and engineering formulas", "health and fitness calculations",
]

TOOLS_SYSTEM = """You design tools for an AI agent. A tool is one pure, deterministic Python function using only the
standard library. Write tool cards (no code). Return JSON:
{"tools": [{"name": "snake_case", "description": "one or two sentences: what it computes and returns",
            "parameters": {"type": "object", "properties": {"arg": {"type": "integer"}}, "required": ["arg"]}}]}"""

NEEDS_SYSTEM = """You write capability needs, the way an agent's planner phrases them: a snake_case name_hint and a
one-sentence description. Vary vocabulary and phrasing; do not copy the tool's own wording. Return JSON:
{"satisfied": [{"name_hint": "...", "description": "..."}],
 "near_miss": [{"name_hint": "...", "description": "..."}]}"""


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split("_")) - {"to", "of", "the", "a"}, set(b.split("_")) - {"to", "of", "the", "a"}
    return len(x & y) / len(x | y) if x | y else 0.0


def card_text(tool: ToolDraft) -> str:
    return tool.search_text()


def _draft(t: dict[str, Any]) -> ToolDraft:
    return ToolDraft(name=t["name"], description=t["description"], parameters=t.get("parameters") or {},
                     code=f"def {t['name']}():\n    pass\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------------------------------ 1. tools

def generate_tools(llm: LLM, embedder: Embedder, eval_tools: list[ToolDraft], per_domain: int = 4,
                   log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    path = OUT / "tools.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["kept"]
    eval_names = sorted(t.name for t in eval_tools)
    raw: list[dict[str, Any]] = []
    for domain in DOMAINS:
        prompt = (f"Domain: {domain}. Write {per_domain} distinct tool cards in this domain.\n"
                  f"These capabilities ALREADY EXIST; do not write tools that compute the same thing (or a trivial "
                  f"variant): {', '.join(eval_names)}.")
        try:
            raw += llm.complete_json(TOOLS_SYSTEM, prompt, temperature=0.8).get("tools", [])
        except ValueError:
            log(f"  skipped domain {domain!r}: unparseable reply")
    # leakage guard: drop cards whose nearest evaluation tool is too similar
    eval_docs = [Doc(t.name, card_text(t), v) for t, v in
                 zip(eval_tools, embedder.embed([card_text(t) for t in eval_tools], "passage"))]
    kept, dropped, seen = [], [], set(eval_names)
    for t in raw:
        try:
            d = _draft(t)
        except Exception:  # noqa: BLE001 - malformed card
            continue
        if d.name in seen:
            dropped.append({"name": d.name, "why": "duplicate name"})
            continue
        q = card_text(d)
        top = hybrid_search(q, embedder.embed([q], "query")[0], eval_docs, k=1)[0]
        name_overlap = max(_jaccard(d.name, e) for e in eval_names)
        if top.cosine >= embedder.reuse_threshold * 0.85 or name_overlap >= 0.5:
            why = (f"too close to eval tool {top.key} (cos {top.cosine:.2f})" if name_overlap < 0.5
                   else f"name overlaps an eval tool ({name_overlap:.2f})")
            dropped.append({"name": d.name, "why": why})
            continue
        seen.add(d.name)
        kept.append({"name": d.name, "description": d.description, "parameters": d.parameters,
                     "nearest_eval_tool": top.key, "nearest_cosine": round(top.cosine, 3)})
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=1), encoding="utf-8")
    log(f"tools: kept {len(kept)}, dropped {len(dropped)} (duplicates / too close to the evaluation tools)")
    return kept


# ------------------------------------------------------------------------------------------ 2. needs

def generate_needs(llm: LLM, tools: list[dict[str, Any]], n_sat: int = 6, n_miss: int = 3,
                   log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    path = OUT / "needs.jsonl"
    rows = _load_jsonl(path)
    done = {r["source_tool"] for r in rows}
    for i, t in enumerate(tools, 1):
        if t["name"] in done:
            continue
        prompt = (f"Tool: {t['name']}: {t['description']}\nParameters: {json.dumps(t['parameters'])}\n\n"
                  f"Write {n_sat} needs this tool fully satisfies, and {n_miss} near-miss needs on a similar topic "
                  "that it does NOT satisfy (a different computation, not just different inputs).")
        try:
            data = llm.complete_json(NEEDS_SYSTEM, prompt, temperature=0.9)
        except ValueError:
            log(f"  needs {i}/{len(tools)} {t['name']}: unparseable, skipped")
            continue
        for kind, items in (("satisfied", data.get("satisfied", [])), ("near_miss", data.get("near_miss", []))):
            for item in items:
                try:
                    need = Need(name_hint=str(item["name_hint"]), description=str(item["description"]))
                except Exception:  # noqa: BLE001
                    continue
                row = {"source_tool": t["name"], "kind": kind, **need.model_dump()}
                _append(path, row)
                rows.append(row)
        log(f"  needs {i}/{len(tools)} {t['name']}")
    return rows


# ------------------------------------------------------------------------------------------ 3. label

def teacher_decision(llm: LLM, task: str, need: Need, candidates: list[ToolDraft]) -> tuple[str | None, str]:
    """Exactly the judge call Forge.match makes (same system prompt, same prompt layout)."""
    prompt = (f"Task: {task}\nNeed: {need.name_hint} — {need.description}\n\n"
              f"Candidate tools:\n{P.tool_cards(candidates)}")  # type: ignore[arg-type]
    try:
        verdict = llm.complete_json(P.JUDGE, prompt)
    except ValueError:
        return None, "unparseable"
    pick = verdict.get("choice")
    return (pick if pick in {c.name for c in candidates} else None), str(verdict.get("reason", ""))[:200]


def _cards(tools: list[ToolDraft]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]


class _Card:
    """Minimal object with .card() so P.tool_cards renders it exactly like a registry Tool."""

    def __init__(self, d: ToolDraft) -> None:
        self.d = d

    def card(self) -> dict[str, Any]:
        return {"name": self.d.name, "description": self.d.description, "parameters": self.d.parameters}

    @property
    def name(self) -> str:
        return self.d.name


def label(llm: LLM, embedder: Embedder, library: list[ToolDraft], needs: list[dict[str, Any]], out_name: str,
          lexical_weight: float, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    path = OUT / out_name
    rows = _load_jsonl(path)
    done = {(r["name_hint"], r["description"]) for r in rows}
    docs = [Doc(t.name, card_text(t), v, t) for t, v in
            zip(library, embedder.embed([card_text(t) for t in library], "passage"))]
    for n in needs:
        if (n["name_hint"], n["description"]) in done:
            continue
        need = Need(name_hint=n["name_hint"], description=n["description"])
        q = need.query()
        hits = hybrid_search(q, embedder.embed([q], "query")[0], docs, k=TOP_K, lexical_weight=lexical_weight)
        cands = [h.payload for h in hits]
        before, t0 = llm.usage.snapshot(), time.perf_counter()
        choice, reason = teacher_decision(llm, n["description"], need, [_Card(c) for c in cands])
        used = llm.usage.since(before)
        row = {**n, "query": q, "candidates": _cards(cands), "candidate_texts": [card_text(c) for c in cands],
               "cosines": [round(h.cosine, 4) for h in hits], "teacher": choice, "teacher_reason": reason,
               "teacher_latency_s": round(time.perf_counter() - t0, 3), "teacher_tokens": used.total_tokens}
        _append(path, row)
        rows.append(row)
        log(f"  {out_name} {len(rows)}/{len(needs)} {n['name_hint']:<28} -> {choice or 'create'}")
    return rows


# ----------------------------------------------------------------------------------------- 4. export

def export(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], val_frac: float = 0.2,
           seed: int = 0) -> dict[str, Any]:
    tools = sorted({r["source_tool"] for r in train_rows})
    rng = random.Random(seed)
    val_tools = set(rng.sample(tools, max(1, round(len(tools) * val_frac)))) if tools else set()

    def pairs(rows: list[dict[str, Any]], gold_key: str) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            for cand, text, cos in zip(r["candidates"], r["candidate_texts"], r["cosines"]):
                out.append({"query": r["query"], "card": text, "label": int(cand["name"] == r[gold_key]),
                            "cosine": cos, "group": r.get("source_tool", "")})
        return out

    splits = {"train": [r for r in train_rows if r["source_tool"] not in val_tools],
              "val": [r for r in train_rows if r["source_tool"] in val_tools]}
    files = {}
    for name, rows in splits.items():
        (OUT / f"pairs_{name}.jsonl").write_text("\n".join(json.dumps(p) for p in pairs(rows, "teacher")) + "\n",
                                                 encoding="utf-8")
        (OUT / f"decisions_{name}.jsonl").write_text(
            "\n".join(json.dumps({k: r[k] for k in ("query", "candidate_texts", "candidates", "teacher", "kind",
                                                     "source_tool")}) for r in rows) + "\n", encoding="utf-8")
        files[name] = {"needs": len(rows), "pairs": len(pairs(rows, "teacher")),
                       "positives": sum(p["label"] for p in pairs(rows, "teacher"))}
    # test: human gold labels on unseen tools, plus the teacher's decision on the same candidates
    (OUT / "decisions_test.jsonl").write_text(
        "\n".join(json.dumps({"query": r["query"], "candidate_texts": r["candidate_texts"],
                              "candidates": r["candidates"], "gold": r["gold"], "teacher": r["teacher"]})
                  for r in test_rows) + "\n", encoding="utf-8")
    teacher_acc = (sum(r["teacher"] == r["gold"] for r in test_rows) / len(test_rows)) if test_rows else 0.0
    gold_reachable = (sum(1 for r in test_rows if r["gold"] is None or r["gold"] in
                          {c["name"] for c in r["candidates"]}) / len(test_rows)) if test_rows else 0.0
    timed = [r for r in train_rows + test_rows if "teacher_latency_s" in r]
    cost = ({"teacher_mean_latency_s": round(sum(r["teacher_latency_s"] for r in timed) / len(timed), 3),
             "teacher_mean_tokens": round(sum(r["teacher_tokens"] for r in timed) / len(timed), 1),
             "teacher_calls": len(timed)} if timed else {})
    stats = {"splits": files, "val_tools": sorted(val_tools), "test_needs": len(test_rows), **cost,
             "teacher_test_accuracy": round(teacher_acc, 3), "test_gold_in_candidates": round(gold_reachable, 3)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="only rebuild exported files from saved labels")
    ap.add_argument("--per-domain", type=int, default=4)
    ap.add_argument("--embedder", help="retrieval embedder for candidates (default: TOOLFORGE_EMBEDDER)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    if args.embedder:
        settings.embedder = args.embedder
    embedder = make_embedder(settings)
    lw = settings.lexical_weight if settings.lexical_weight is not None else embedder.lexical_weight
    eval_tools, eval_queries = load_set()

    if not args.export:
        from toolforge.llm import make_llm

        llm = make_llm(settings)
        print(f"teacher: {settings.provider}:{settings.model} · embedder: {embedder.identity}")
        try:
            tools = generate_tools(llm, embedder, eval_tools, args.per_domain)
            needs = generate_needs(llm, tools)
            label(llm, embedder, [_draft(t) for t in tools], needs, "labelled_train.jsonl", lw)
            test_needs = [{"source_tool": "", "kind": "test", "name_hint": q["name_hint"],
                           "description": q["description"], "gold": q["gold"]} for q in eval_queries]
            label(llm, embedder, eval_tools, test_needs, "labelled_test.jsonl", lw)
        except LLMConfigError as e:
            print(f"\nstopped: {e}\n  Progress is saved; run again to resume.", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nstopped. Progress is saved; run again to resume.", file=sys.stderr)
            return 130
    stats = export(_load_jsonl(OUT / "labelled_train.jsonl"), _load_jsonl(OUT / "labelled_test.jsonl"))
    print(json.dumps(stats, indent=1))
    bundle = colab_bundle()
    print(f"\nColab bundle: {bundle}  (upload it in notebooks/train_reranker.ipynb, option B)")
    return 0


def colab_bundle(dest: Path | None = None) -> Path:
    """Zip exactly what the Colab notebook needs, with forward-slash paths (Compress-Archive on
    Windows PowerShell 5 writes backslashes, which Linux unzips as literal file names)."""
    import zipfile

    root = Path(__file__).resolve().parents[1]
    dest = dest or root / "toolforge_colab.zip"
    files = [Path(__file__).with_name("train_reranker.py"),
             *sorted(p for p in OUT.glob("*") if p.is_file() and p.suffix in {".jsonl", ".json"})]
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = f.relative_to(root) if f.is_relative_to(root) else Path("evals/distill") / f.name
            z.write(f, rel.as_posix())
    return dest


if __name__ == "__main__":
    raise SystemExit(main())
