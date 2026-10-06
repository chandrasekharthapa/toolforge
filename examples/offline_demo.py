"""See the whole Toolforge loop in two seconds, with no API key.

A scripted model plays every LLM role. Its first draft has a subtle bug (it forgets
that date ranges can be given in reverse) and its own unit tests miss it, because
they never try a reversed range. Differential fuzzing against an independent
reference implementation exposes the bug; the arbiter turns the disagreement into
regression tests; the repair passes; a lesson is stored; the second task reuses
the tool without writing any code.

    python examples/offline_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from fakes import DAYS_BETWEEN, Brain, draft  # noqa: E402

from toolforge.agent import Toolforge  # noqa: E402
from toolforge.cli import _event  # noqa: E402
from toolforge.config import Settings  # noqa: E402
from toolforge.embeddings import HashingEmbedder  # noqa: E402
from toolforge.registry import Registry  # noqa: E402


def main() -> None:
    buggy = DAYS_BETWEEN.replace("abs((b - a).days)", "(b - a).days")
    forward_tests = [
        {"kwargs": {"start": "2024-01-15", "end": "2024-03-01"}, "expected": 46},
        {"kwargs": {"start": "2024-02-28", "end": "2024-03-01"}, "expected": 2},
        {"kwargs": {"start": "2020-01-01", "end": "2020-12-31"}, "expected": 365},
    ]
    lesson = {"mistake": "assumed date ranges are always given in order",
              "fix": "return abs((end - start).days) so reversed ranges work"}

    brain = Brain().plan("How many days", ("days_between", "Number of days between two ISO dates."))
    brain.references["days_between"] = DAYS_BETWEEN
    brain.will_write("days_between", draft(buggy, tests=forward_tests), draft(tests=forward_tests, lesson=lesson))
    brain.judge_choice = "days_between"
    brain.arbiter = lambda cases: {"verdicts": [
        {"input": c["input"], "expected": abs((date.fromisoformat(c["input"]["end"])
                                               - date.fromisoformat(c["input"]["start"])).days)} for c in cases]}

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(provider="scripted", embedder="hashing", db_path=f"{tmp}/demo.db", fuzz_cases=30)
        forge = Toolforge(settings, llm=brain.llm(), embedder=HashingEmbedder(), registry=Registry(settings.db_path))

        for task in ("How many days between 2024-01-15 and 2024-03-01?",
                     "How many days between 2023-12-25 and 2024-02-14?"):
            print(f"\n\033[1m▶ {task}\033[0m")
            result = forge.run(task, on_event=_event)
            print(f"\033[1manswer  \033[0m {result.answer}   "
                  f"\033[2m({result.llm_calls} LLM calls, created={result.created}, reused={result.reused})\033[0m")

        tool = forge.registry.get("days_between")
        print(f"\n\033[1mlibrary\033[0m  {tool.name} v{tool.version}: {len(tool.tests)} tests "
              f"(3 written by the model + {len(tool.tests) - 3} added by the arbiter), "
              f"differential={tool.verification.differential}")
        for lesson_row in forge.registry.list_lessons():
            print(f"\033[1mlesson\033[0m   {lesson_row.mistake} → {lesson_row.fix}")


if __name__ == "__main__":
    main()
