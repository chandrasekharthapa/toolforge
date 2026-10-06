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
import json
import math
import re
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
            if math.isclose(value, float(expected), rel_tol=1e-6, abs_tol=0.011):
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


def run_mode(mode: str, tasks: list[dict[str, Any]], factory: AgentFactory = default_factory,
             delay: float = 0.0, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    rows = []
    # ignore_cleanup_errors: on Windows a lingering file handle must not crash the report
    with tempfile.TemporaryDirectory(prefix=f"tf-bench-{mode}-", ignore_cleanup_errors=True) as tmp:
        shared = None
        if mode != "fresh":
            shared = factory(str(Path(tmp) / "shared.db"), mode != "library-norag", mode != "no-diff")
        for i, item in enumerate(tasks):
            agent = shared or factory(str(Path(tmp) / f"fresh-{i}.db"), True, True)
            try:
                result = agent.run(item["task"])
                row = {"id": item["id"], "family": item["family"], "correct": grade(result.answer, item["expected"]),
                       "answer": result.answer, "expected": item["expected"], "tokens": result.total_tokens,
                       "llm_calls": result.llm_calls, "latency_s": result.latency_s,
                       "created": result.created, "reused": result.reused, "failed": result.failed_needs}
            except LLMConfigError:
                for a in {id(x): x for x in (agent, shared) if x is not None}.values():
                    a.close()
                raise  # a bad key or model fails every task; stop instead of recording crashes
            except Exception as e:  # noqa: BLE001 - a crashed task is a failed task, keep going
                row = {"id": item["id"], "family": item["family"], "correct": False, "answer": f"ERROR {e}",
                       "expected": item["expected"], "tokens": 0, "llm_calls": 0, "latency_s": 0.0,
                       "created": [], "reused": [], "failed": ["<crash>"]}
            finally:
                if agent is not shared:
                    agent.close()
            rows.append(row)
            mark = "✓" if row["correct"] else "✗"
            log(f"[{mode:>13}] {i + 1:>2}/{len(tasks)} {mark} {item['id']:<11} tokens={row['tokens']:>6} "
                f"created={row['created']} reused={row['reused']}")
            if delay:
                time.sleep(delay)
        if shared is not None:
            rows.append({"_library": shared.registry.stats()})
            shared.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = [r for r in rows if "id" in r]
    n = len(tasks)
    half = n // 2
    created = sum(len(r["created"]) for r in tasks)
    reused = sum(len(r["reused"]) for r in tasks)
    tok = [r["tokens"] for r in tasks]
    return {
        "tasks": n,
        "accuracy": round(sum(r["correct"] for r in tasks) / n, 3) if n else 0,
        "total_tokens": sum(tok),
        "mean_tokens": round(sum(tok) / n) if n else 0,
        "mean_tokens_first_half": round(sum(tok[:half]) / half) if half else 0,
        "mean_tokens_second_half": round(sum(tok[half:]) / (n - half)) if n - half else 0,
        "mean_llm_calls": round(sum(r["llm_calls"] for r in tasks) / n, 2) if n else 0,
        "mean_latency_s": round(sum(r["latency_s"] for r in tasks) / n, 2) if n else 0,
        "tools_created": created,
        "tools_reused": reused,
        "reuse_rate": round(reused / (created + reused), 3) if created + reused else 0,
        "failed_needs": sum(len(r["failed"]) for r in tasks),
    }


def render_markdown(summaries: dict[str, dict[str, Any]], meta: dict[str, Any]) -> str:
    cols = ["accuracy", "mean_tokens", "mean_tokens_first_half", "mean_tokens_second_half",
            "mean_llm_calls", "mean_latency_s", "tools_created", "reuse_rate"]
    lines = [f"## Benchmark — {meta['tasks']} tasks · model `{meta['model']}` · {meta['date']}", "",
             "| mode | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, s in summaries.items():
        lines.append(f"| {mode} | " + " | ".join(str(s[c]) for c in cols) + " |")
    if "fresh" in summaries and "library" in summaries and summaries["fresh"]["mean_tokens"]:
        saving = 1 - summaries["library"]["mean_tokens"] / summaries["fresh"]["mean_tokens"]
        lines += ["", f"**A shared library cut mean tokens per task by {saving:.0%} versus forging from scratch.**"]
    return "\n".join(lines) + "\n"


def plot(results: dict[str, list[dict[str, Any]]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    palette = {"fresh": "#9a9a9a", "library": "#2a6fdb", "library-norag": "#e0a020", "no-diff": "#c0504d"}
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for mode, rows in results.items():
        tasks = [r for r in rows if "id" in r]
        cum, total = [], 0
        for r in tasks:
            total += r["tokens"]
            cum.append(total)
        ax.plot(range(1, len(cum) + 1), cum, label=mode, color=palette.get(mode), linewidth=2)
    ax.set_xlabel("task #")
    ax.set_ylabel("cumulative LLM tokens")
    ax.set_title("Cost of solving the task stream")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between tasks (rate limits)")
    ap.add_argument("--tasks", default=str(HERE / "tasks.json"))
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())[: args.limit]
    settings = Settings()
    RESULTS.mkdir(exist_ok=True)

    try:
        results = {mode: run_mode(mode, tasks, delay=args.delay) for mode in args.modes}
    except LLMConfigError as e:
        print(f"\nbenchmark stopped: {e}", file=sys.stderr)
        return 2
    summaries = {mode: summarize(rows) for mode, rows in results.items()}
    meta = {"tasks": len(tasks), "model": f"{settings.provider}:{settings.model or 'default'}",
            "embedder": settings.embedder, "date": time.strftime("%Y-%m-%d")}

    (RESULTS / "benchmark.json").write_text(json.dumps({"meta": meta, "summary": summaries,
                                                        "rows": results}, indent=2))
    md = render_markdown(summaries, meta)
    if plot(results, RESULTS / "benchmark.png"):
        md += "\n![cumulative tokens](benchmark.png)\n"
    (RESULTS / "benchmark.md").write_text(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
