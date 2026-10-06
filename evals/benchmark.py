"""Benchmark: does a self-built tool library pay for itself, and does RAG help?

Runs the same ordered task list under several modes and records accuracy, tokens,
LLM calls, latency and tool reuse:

* ``fresh``          – empty library for every task (always forges): the baseline
* ``library``        – one shared library, RAG on (examples + lessons)
* ``library-norag``  – one shared library, RAG off (ablation)
* ``no-diff``        – shared library, RAG on, differential verification off (ablation)

    python -m evals.benchmark                       # all modes, all tasks
    python -m evals.benchmark --modes fresh library --limit 10 --delay 4

Writes evals/results/benchmark.{json,md,png}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolforge.agent import Toolforge  # noqa: E402
from toolforge.config import Settings  # noqa: E402
from toolforge.llm import LLMConfigError  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
MODES = ("fresh", "library", "library-norag", "no-diff")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

AgentFactory = Callable[[str, bool, bool], Toolforge]  # (db_path, rag, differential) -> agent


def grade(answer: str, expected: Any) -> bool:
    if isinstance(expected, bool):
        expected = "yes" if expected else "no"
    if isinstance(expected, (int, float)):
        for raw in _NUM.findall(answer.replace("−", "-")):
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            if math.isclose(value, float(expected), rel_tol=1e-9, abs_tol=0.005):  # within half a cent
                return True
        return False
    text = answer.lower()
    target = str(expected).lower()
    if target in {"yes", "no"}:
        return re.search(rf"\b{target}\b", text) is not None
    return target in text


def default_factory(db_path: str, rag: bool, differential: bool) -> Toolforge:
    settings = Settings()
    settings.db_path = db_path
    settings.differential = differential
    return Toolforge(settings, rag=rag)


def _load_progress(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a half-written last line from a killed run
            break
    return rows


def _append_progress(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_mode(mode: str, tasks: list[dict[str, Any]], factory: AgentFactory = default_factory,
             delay: float = 0.0, log: Callable[[str], None] = print,
             checkpoint_dir: Path | None = None) -> list[dict[str, Any]]:
    """Run one mode. With ``checkpoint_dir`` every finished task is saved immediately and a
    re-run resumes after the last saved task (library mode keeps its tool database there too)."""
    progress = checkpoint_dir / f"{mode}.jsonl" if checkpoint_dir else None
    rows = _load_progress(progress)
    done = {r["id"] for r in rows}
    if rows:
        log(f"[{mode:>13}] resuming: {len(done)}/{len(tasks)} tasks already done")
    started = time.perf_counter()
    ran = 0
    # ignore_cleanup_errors: on Windows a lingering file handle must not crash the report
    with tempfile.TemporaryDirectory(prefix=f"tf-bench-{mode}-", ignore_cleanup_errors=True) as tmp:
        shared = None
        if mode != "fresh":
            db = (checkpoint_dir / f"{mode}.db") if checkpoint_dir else (Path(tmp) / "shared.db")
            shared = factory(str(db), mode != "library-norag", mode != "no-diff")
        for i, item in enumerate(tasks):
            if item["id"] in done:
                continue
            agent = shared or factory(str(Path(tmp) / f"fresh-{i}.db"), True, True)
            try:
                result = agent.run(item["task"])
                row = {"id": item["id"], "family": item["family"], "correct": grade(result.answer, item["expected"]),
                       "answer": result.answer, "expected": item["expected"], "tokens": result.total_tokens,
                       "llm_calls": result.llm_calls, "latency_s": result.latency_s,
                       "created": result.created, "reused": result.reused, "failed": result.failed_needs,
                       "tool_calls": len(result.tool_calls)}
            except LLMConfigError:
                for a in {id(x): x for x in (agent, shared) if x is not None}.values():
                    a.close()
                raise  # a bad key or model fails every task; stop instead of recording crashes
            except Exception as e:  # noqa: BLE001 - a crashed task is a failed task, keep going
                row = {"id": item["id"], "family": item["family"], "correct": False, "answer": f"ERROR {e}",
                       "expected": item["expected"], "tokens": 0, "llm_calls": 0, "latency_s": 0.0,
                       "created": [], "reused": [], "failed": ["<crash>"], "tool_calls": 0}
            finally:
                if agent is not shared:
                    agent.close()
            rows.append(row)
            _append_progress(progress, row)
            ran += 1
            left = sum(1 for t in tasks if t["id"] not in {r["id"] for r in rows})
            eta = (time.perf_counter() - started) / ran * left
            mark = "✓" if row["correct"] else "✗"
            log(f"[{mode:>13}] {len(rows):>2}/{len(tasks)} {mark} {item['id']:<11} tokens={row['tokens']:>6} "
                f"created={row['created']} reused={row['reused']}"
                + (f"  (~{eta / 60:.0f} min left in this mode)" if left else ""))
            if delay and left:
                time.sleep(delay)
        order = {t["id"]: n for n, t in enumerate(tasks)}
        rows.sort(key=lambda r: order.get(r["id"], 1 << 30))
        if shared is not None:
            rows.append({"_library": shared.registry.stats()})
            shared.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = [r for r in rows if "id" in r]
    n = len(tasks)
    created = sum(len(r["created"]) for r in tasks)
    reused = sum(len(r["reused"]) for r in tasks)
    tok = [r["tokens"] for r in tasks]
    # "verified": the answer is right AND came from a sandboxed call to a verified tool, not the model's head
    verified = sum(1 for r in tasks if r["correct"] and r.get("tool_calls", 1) > 0)
    return {
        "tasks": n,
        "accuracy": round(sum(r["correct"] for r in tasks) / n, 3) if n else 0,
        "verified_accuracy": round(verified / n, 3) if n else 0,
        "unverified_answers": sum(1 for r in tasks if r.get("tool_calls", 1) == 0),
        "total_tokens": sum(tok),
        "mean_tokens": round(sum(tok) / n) if n else 0,
        "mean_llm_calls": round(sum(r["llm_calls"] for r in tasks) / n, 2) if n else 0,
        "mean_latency_s": round(sum(r["latency_s"] for r in tasks) / n, 2) if n else 0,
        "tools_created": created,
        "tools_reused": reused,
        "reuse_rate": round(reused / (created + reused), 3) if created + reused else 0,
        "failed_needs": sum(len(r["failed"]) for r in tasks),
    }


def reuse_comparison(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Same task, both modes, on exactly the tasks where the library run reused a tool and built none."""
    if "fresh" not in results or "library" not in results:
        return None
    fresh = {r["id"]: r for r in results["fresh"] if "id" in r}
    pairs = [(fresh[r["id"]], r) for r in results["library"]
             if "id" in r and r["reused"] and not r["created"] and r["id"] in fresh]
    if not pairs:
        return None
    ft, lt = sum(f["tokens"] for f, _ in pairs), sum(lib["tokens"] for _, lib in pairs)
    fl, ll = sum(f["latency_s"] for f, _ in pairs), sum(lib["latency_s"] for _, lib in pairs)
    return {"tasks": [lib["id"] for _, lib in pairs], "fresh_tokens": ft, "library_tokens": lt,
            "token_saving": round(1 - lt / ft, 3) if ft else 0, "fresh_latency_s": round(fl, 1),
            "library_latency_s": round(ll, 1), "latency_saving": round(1 - ll / fl, 3) if fl else 0}


def render_markdown(summaries: dict[str, dict[str, Any]], meta: dict[str, Any],
                    reuse: dict[str, Any] | None = None) -> str:
    cols = [("accuracy", "accuracy"), ("verified_accuracy", "verified accuracy"),
            ("unverified_answers", "answers without a tool"), ("mean_tokens", "mean tokens / task"),
            ("mean_latency_s", "mean latency (s)"), ("tools_created", "tools built"), ("reuse_rate", "reuse rate")]
    lines = [f"## Benchmark: {meta['tasks']} tasks · `{meta['model']}` · {meta['date']}", "",
             "| mode | " + " | ".join(label for _, label in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, s in summaries.items():
        lines.append(f"| {mode} | " + " | ".join(str(s[k]) for k, _ in cols) + " |")
    lines += ["", "*Verified accuracy* counts an answer only if it is correct **and** was computed by a sandboxed "
              "call to a verified tool, not by the model on its own."]
    if reuse:
        lines += ["", f"**On the {len(reuse['tasks'])} tasks where the library reused a verified tool, tokens fell "
                  f"{reuse['token_saving']:.0%} ({reuse['fresh_tokens']:,} → {reuse['library_tokens']:,}) and "
                  f"latency fell {reuse['latency_saving']:.0%} ({reuse['fresh_latency_s']}s → "
                  f"{reuse['library_latency_s']}s) versus building from scratch.**"]
    for ablation, what in (("library-norag", "RAG (retrieved examples + lessons) off"),
                           ("no-diff", "differential verification off")):
        if "library" in summaries and ablation in summaries:
            a, b = summaries["library"], summaries[ablation]
            lines += ["", f"**Ablation, {what}:** verified accuracy {b['verified_accuracy']:.0%} vs "
                      f"{a['verified_accuracy']:.0%} with it on; failed tool builds {b['failed_needs']} vs "
                      f"{a['failed_needs']}; mean tokens {b['mean_tokens']:,} vs {a['mean_tokens']:,}."]
    if "fresh" in summaries and "library" in summaries and summaries["fresh"]["mean_tokens"]:
        saving = 1 - summaries["library"]["mean_tokens"] / summaries["fresh"]["mean_tokens"]
        lines += ["", f"Across all {meta['tasks']} tasks (including first-time builds of new tools) mean tokens "
                  f"per task fell {saving:.0%}. Single run; LLM cost varies run to run."]
    return "\n".join(lines) + "\n"


def plot(results: dict[str, list[dict[str, Any]]], path: Path) -> bool:
    """Cumulative token cost per mode; markers show where the library reused a verified tool."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except ImportError:
        return False
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
    colors = {"fresh": "#8a8984", "library": "#2a78d6", "library-norag": "#eb6834", "no-diff": "#1baf7a"}
    labels = {"fresh": "fresh (empty library every task)", "library": "library (reuses verified tools)",
              "library-norag": "library, RAG off", "no-diff": "library, no differential check"}
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    any_reuse = False
    for mode, rows in results.items():
        tasks = [r for r in rows if "id" in r]
        xs, cum, total = list(range(1, len(tasks) + 1)), [], 0
        for r in tasks:
            total += r["tokens"]
            cum.append(total)
        color = colors.get(mode, ink2)
        ax.plot(xs, cum, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                label=labels.get(mode, mode), zorder=2)
        if mode != "fresh":
            hits = [(x, y) for x, y, r in zip(xs, cum, tasks) if r["reused"] and not r["created"]]
            if hits:
                ax.scatter(*zip(*hits), s=48, color=color, edgecolors=surface, linewidths=2, zorder=3)
                any_reuse = True
        ax.annotate(f"{total / 1000:.0f}k", (xs[-1], cum[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=ink)
    n = max((len([r for r in rows if "id" in r]) for rows in results.values()), default=1)
    ax.set_xticks(sorted({1, *range(5, n + 1, 5), n}))
    ax.set_xlim(0.5, n + 1.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    ax.set_xlabel("task # (in benchmark order)", color=ink2, fontsize=9)
    ax.set_ylabel("cumulative LLM tokens", color=ink2, fontsize=9)
    ax.set_title("Cost of solving the 20-task stream", color=ink, fontsize=11, loc="left")
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.grid(axis="y", color=grid, linewidth=1)
    ax.set_axisbelow(True)
    ax.margins(x=0.06)
    handles, _ = ax.get_legend_handles_labels()
    if any_reuse:  # one legend entry for the dots, in neutral ink, whatever line they sit on
        from matplotlib.lines import Line2D

        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6, color=ink2,
                              label="dot = task solved by reusing a verified tool"))
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=ink2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return True


def merge_saved(results: dict[str, list[dict[str, Any]]], meta: dict[str, Any],
                task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Keep modes from an earlier run of the SAME model, embedder and task list, so an ablation can be
    run on its own (e.g. --modes library-norag) and land next to the fresh/library results."""
    path = RESULTS / "benchmark.json"
    if not path.exists():
        return results
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return results
    old = saved.get("meta", {})
    same = old.get("model") == meta["model"] and old.get("embedder", meta["embedder"]) == meta["embedder"]
    merged = {}
    for mode in MODES:
        if mode in results:
            merged[mode] = results[mode]
        elif same and mode in saved.get("rows", {}):
            ids = [r["id"] for r in saved["rows"][mode] if "id" in r]
            if ids == task_ids:
                merged[mode] = saved["rows"][mode]
    return merged


def write_report(results: dict[str, list[dict[str, Any]]], meta: dict[str, Any]) -> None:
    summaries = {mode: summarize(rows) for mode, rows in results.items()}
    reuse = reuse_comparison(results)
    (RESULTS / "benchmark.json").write_text(json.dumps({"meta": meta, "summary": summaries, "reuse": reuse,
                                                        "rows": results}, indent=2), encoding="utf-8")
    md = render_markdown(summaries, meta, reuse)
    if plot(results, RESULTS / "benchmark.png"):
        md += "\n![cumulative tokens](benchmark.png)\n"
    (RESULTS / "benchmark.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between tasks (rate limits)")
    ap.add_argument("--tasks", default=str(HERE / "tasks.json"))
    ap.add_argument("--restart", action="store_true", help="discard saved progress and start from the first task")
    ap.add_argument("--report", action="store_true",
                    help="rebuild benchmark.md/png from the saved benchmark.json (re-grading answers) without running")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):  # never crash on a character the console cannot show
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.report:
        saved = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
        for rows in saved["rows"].values():
            for r in rows:
                if "id" in r:
                    r["correct"] = grade(r["answer"], r["expected"])
        write_report(saved["rows"], saved["meta"])
        return 0
    tasks = json.loads(Path(args.tasks).read_text())[: args.limit]
    settings = Settings()
    print(f"benchmark: {len(tasks)} tasks x {len(args.modes)} modes on {settings.provider}:{settings.model}")
    RESULTS.mkdir(exist_ok=True)

    # Progress lives in a folder keyed by model + task list, so re-running the SAME command
    # resumes, while changing the model or the task set starts a clean run.
    key = hashlib.sha256(json.dumps([settings.provider, settings.model, settings.embedder,
                                     [t["id"] for t in tasks]]).encode()).hexdigest()[:10]
    checkpoint = RESULTS / ".progress" / key
    if args.restart and checkpoint.exists():
        shutil.rmtree(checkpoint, ignore_errors=True)
    checkpoint.mkdir(parents=True, exist_ok=True)

    try:
        results = {mode: run_mode(mode, tasks, delay=args.delay, checkpoint_dir=checkpoint) for mode in args.modes}
    except LLMConfigError as e:
        print(f"\nbenchmark stopped: {e}\n  Finished tasks are saved; run the same command again to resume.",
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nstopped. Every finished task is saved: run the same command again to resume "
              "(add --restart to start over).", file=sys.stderr)
        return 130
    meta = {"tasks": len(tasks), "model": f"{settings.provider}:{settings.model or 'default'}",
            "embedder": settings.embedder, "date": time.strftime("%Y-%m-%d")}
    write_report(merge_saved(results, meta, [t["id"] for t in tasks]), meta)
    shutil.rmtree(checkpoint, ignore_errors=True)  # report written; the next run starts clean
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
