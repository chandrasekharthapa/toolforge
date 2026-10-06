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


def _date_tasks():
    return [{"id": f"d{i}", "family": "dates", "task": f"How many days between {a} and {b}?", "expected": e}
            for i, (a, b, e) in enumerate([("2024-01-15", "2024-03-01", 46), ("2023-12-25", "2024-02-14", 51),
                                           ("2000-02-28", "2000-03-01", 2)])]


def _factory(interrupt_on: str | None = None):
    def factory(db_path, rag, differential):
        brain = Brain().plan("How many days", ("days_between", "Number of days between two ISO dates."))
        brain.references["days_between"] = DAYS_BETWEEN
        brain.judge_choice = "days_between"
        brain.will_write("days_between", *[draft() for _ in range(5)])
        settings = Settings(provider="scripted", embedder="hashing", db_path=db_path, fuzz_cases=10,
                            differential=differential)
        agent = Toolforge(settings, llm=brain.llm(), embedder=HashingEmbedder(), registry=Registry(db_path), rag=rag)
        if interrupt_on:
            real_run = agent.run

            def run(task, on_event=None):
                if interrupt_on in task:
                    raise KeyboardInterrupt  # the user closed the window mid-benchmark
                return real_run(task, on_event)

            agent.run = run
        return agent

    return factory


@pytest.mark.parametrize("mode", ["fresh", "library"])
def test_benchmark_resumes_after_interruption(tmp_path, mode):
    tasks = _date_tasks()
    with pytest.raises(KeyboardInterrupt):
        run_mode(mode, tasks, _factory(interrupt_on="2000-02-28"), log=lambda _: None, checkpoint_dir=tmp_path)
    saved = [line for line in (tmp_path / f"{mode}.jsonl").read_text().splitlines() if line]
    assert len(saved) == 2  # the two finished tasks survived the interruption

    logs = []
    rows = run_mode(mode, tasks, _factory(), log=logs.append, checkpoint_dir=tmp_path)
    task_rows = [r for r in rows if "id" in r]
    assert [r["id"] for r in task_rows] == ["d0", "d1", "d2"] and all(r["correct"] for r in task_rows)
    assert "resuming: 2/3" in logs[0] and sum("✓" in line for line in logs) == 1  # only the missing task ran
    if mode == "library":  # the library survived too, so the resumed task reused instead of re-forging
        assert task_rows[2]["reused"] == ["days_between"] and task_rows[2]["created"] == []


def test_half_written_progress_line_is_ignored(tmp_path):
    (tmp_path / "fresh.jsonl").write_text('{"id": "d0", "correct": true}\n{"id": "d1", "corr')
    from evals.benchmark import _load_progress
    assert [r["id"] for r in _load_progress(tmp_path / "fresh.jsonl")] == ["d0"]


def test_verified_accuracy_and_reuse_comparison():
    from evals.benchmark import reuse_comparison

    def row(i, correct=True, calls=1, created=(), reused=(), tokens=100, latency=1.0):
        return {"id": i, "correct": correct, "tool_calls": calls, "created": list(created), "reused": list(reused),
                "tokens": tokens, "latency_s": latency, "llm_calls": 3, "failed": []}

    fresh = [row("a", created=["t"], tokens=1000, latency=10), row("b", calls=0, tokens=900, latency=9),
             row("c", correct=False, created=["u"])]
    library = [row("a", created=["t"], tokens=800), row("b", reused=["t"], tokens=200, latency=1),
               row("c", created=["u"]), {"_library": {}}]
    s = summarize(fresh)
    assert s["accuracy"] == round(2 / 3, 3) and s["verified_accuracy"] == round(1 / 3, 3)
    assert s["unverified_answers"] == 1  # "b" was right, but answered without any tool
    r = reuse_comparison({"fresh": fresh, "library": library})
    assert r["tasks"] == ["b"] and r["token_saving"] == round(1 - 200 / 900, 3)
    assert r["latency_saving"] == round(1 - 1 / 9, 3)


def test_grading_is_exact_to_the_cent():
    assert grade("14176.25", 14176.25) and not grade("14176.24", 14176.25)


def test_running_one_ablation_keeps_the_earlier_modes(tmp_path, monkeypatch):
    import json as _json

    import evals.benchmark as B

    monkeypatch.setattr(B, "RESULTS", tmp_path)
    meta = {"tasks": 1, "model": "nvidia:m", "embedder": "hashing", "date": "2026-10-06"}
    row = {"id": "t1", "correct": True, "tool_calls": 1, "created": ["x"], "reused": [], "tokens": 10,
           "latency_s": 1.0, "llm_calls": 3, "failed": []}
    (tmp_path / "benchmark.json").write_text(_json.dumps({"meta": meta, "rows": {"fresh": [row], "library": [row]}}))
    merged = B.merge_saved({"library-norag": [dict(row, failed=["y"])]}, meta, ["t1"])
    assert list(merged) == ["fresh", "library", "library-norag"]
    other_model = B.merge_saved({"library-norag": [row]}, dict(meta, model="groq:z"), ["t1"])
    assert list(other_model) == ["library-norag"]  # never mix results from a different model
    md = B.render_markdown({m: B.summarize(r) for m, r in merged.items()}, meta)
    assert "Ablation, RAG (retrieved examples + lessons) off" in md
