import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import DAYS_BETWEEN, Brain, draft  # noqa: E402

from evals.benchmark import grade, run_mode, summarize  # noqa: E402
from evals.redteam import run_offline  # noqa: E402
from toolforge.agent import Toolforge  # noqa: E402
from toolforge.config import Settings  # noqa: E402
from toolforge.embeddings import HashingEmbedder  # noqa: E402
from toolforge.registry import Registry  # noqa: E402


@pytest.mark.parametrize("answer, expected, ok", [
    ("The answer is 46.", 46, True),
    ("about 14,176.25 dollars", 14176.25, True),
    ("62.1 miles", 62.14, False),
    ("It is −40 °F", -40.0, True),
    ("Yes, 7919 is prime.", "yes", True),
    ("Not prime: no", "yes", False),
    ("mcmxciv", "MCMXCIV", True),
])
def test_grade(answer, expected, ok):
    assert grade(answer, expected) is ok


def test_library_mode_is_cheaper_than_fresh_mode():
    tasks = [{"id": f"d{i}", "family": "dates", "task": f"How many days between {a} and {b}?", "expected": e}
             for i, (a, b, e) in enumerate([("2024-01-15", "2024-03-01", 46), ("2023-12-25", "2024-02-14", 51),
                                            ("2000-02-28", "2000-03-01", 2)])]

    def factory(db_path, rag, differential):
        brain = Brain().plan("How many days", ("days_between", "Number of days between two ISO dates."))
        brain.references["days_between"] = DAYS_BETWEEN
        brain.judge_choice = "days_between"
        brain.will_write("days_between", *[draft() for _ in range(5)])
        settings = Settings(provider="scripted", embedder="hashing", db_path=db_path, fuzz_cases=10,
                            differential=differential)
        return Toolforge(settings, llm=brain.llm(), embedder=HashingEmbedder(),
                         registry=Registry(db_path), rag=rag)

    fresh = summarize(run_mode("fresh", tasks, factory, log=lambda _: None))
    library = summarize(run_mode("library", tasks, factory, log=lambda _: None))

    assert fresh["accuracy"] == library["accuracy"] == 1.0
    assert fresh["tools_created"] == 3 and library["tools_created"] == 1
    assert library["reuse_rate"] == round(2 / 3, 3)
    assert library["total_tokens"] < fresh["total_tokens"]


def test_redteam_corpus_has_zero_escapes():
    outcomes = run_offline(timeout=2.0)
    assert len(outcomes) >= 30
    assert not [o.payload for o in outcomes if o.escaped_combined]
    assert not [o.payload for o in outcomes if o.escaped_sandbox_only]  # defence in depth
