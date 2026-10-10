import json

import pytest

from evals.bbh_benchmark import (
    TASKS,
    demo_items,
    grade,
    load_saved,
    load_task,
    make_prompt,
    render,
    run_all,
    sample,
    summarize,
)
from toolforge.llm import ScriptedLLM


@pytest.mark.parametrize("task", TASKS)
def test_grader_accepts_every_official_answer_and_rejects_a_wrong_one(task):
    examples = load_task(task)
    assert len(examples) == 250
    for ex in examples:
        assert grade(task, ex["target"], ex["input"], ex["target"]), ex["target"]
        assert grade(task, f"Let me think.\nAnswer: {ex['target']}", ex["input"], ex["target"])
    ex = examples[0]
    wrong = {"word_sorting": " ".join(reversed(ex["target"].split())) + " extra",
             "dyck_languages": ex["target"] + " )"}.get(task, "(A)" if ex["target"] != "(A)" else "(B)")
    assert not grade(task, wrong, ex["input"], ex["target"])


def test_grader_tolerates_harmless_formatting():
    q = "Sort the following words alphabetically: List: pear apple"
    assert grade("word_sorting", "Sorted: apple, pear", q, "apple pear")
    assert not grade("word_sorting", "pear apple", q, "apple pear")
    dq = "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: ( [ ["
    assert grade("dyck_languages", "] ] )", dq, "] ] )")
    assert grade("dyck_languages", "( [ [ ] ] )", dq, "] ] )")  # echoed the full sequence
    assert not grade("dyck_languages", "] )", dq, "] ] )")
    mq = "Which?\nOptions:\n(A) goalkeeper\n(B) left midfielder\n(C) right winger"
    assert grade("tracking_shuffled_objects_five_objects", "The answer is (C).", mq, "(C)")
    assert grade("tracking_shuffled_objects_five_objects", "Answer: C", mq, "(C)")
    assert grade("tracking_shuffled_objects_five_objects", "Eve ends up as right winger.", mq, "(C)")
    assert not grade("tracking_shuffled_objects_five_objects", "(A) or maybe (B)", mq, "(C)")  # last letter wins


def test_seeds_draw_different_reproducible_samples():
    a, b = sample("word_sorting", 15, 0), sample("word_sorting", 15, 1)
    assert [i for i, _ in a] == [i for i, _ in sample("word_sorting", 15, 0)]
    assert {i for i, _ in a} != {i for i, _ in b} and len(a) == 15


LIBRARIES: dict[str, set] = {}  # db path -> tools built, standing in for the SQLite library
ROLES: list[str] = []


class FakeAgent:
    """Stands in for Toolforge: reuses one 'tool' once its library has it, like the real library."""

    def __init__(self, path, role="maker"):
        self.lib = LIBRARIES.setdefault(path, set())
        self.role = role
        ROLES.append(role)
        self.closed = False

    def run(self, task):
        from toolforge.models import RunResult

        words = task.split("List:", 1)[1].split("\n")[0].split()
        if self.role == "latm-make":
            assert "Correct answer:" in task  # demonstrations, not a test item
        created, reused = ([], ["sort_words"]) if "sort_words" in self.lib else (["sort_words"], [])
        self.lib.add("sort_words")
        user = 40 if self.role == "latm-use" else 0
        return RunResult(task=task, answer=" ".join(sorted(words)), created=created, reused=reused,
                         tool_calls=[{"call": "sort_words"}], llm_calls=2, input_tokens=50, output_tokens=10,
                         user_tokens=user, latency_s=0.1)

    def close(self):
        self.closed = True


def test_run_all_resumes_and_reports(tmp_path):
    def brain(system, prompt):
        words = prompt.split("List:", 1)[1].split("\n")[0].split()
        return "thinking...\nAnswer: " + " ".join(sorted(words))

    kw = dict(tasks=["word_sorting"], seeds=[0, 1], n=3, modes=["direct", "toolforge"], checkpoint=tmp_path,
              make_llm=lambda role: ScriptedLLM(brain), make_agent=FakeAgent, log=lambda _: None)
    first = run_all(**kw)
    assert sorted(first) == ["direct__word_sorting__s0", "direct__word_sorting__s1",
                             "toolforge__word_sorting__s0", "toolforge__word_sorting__s1"]
    assert all(r["correct"] for rows in first.values() for r in rows)
    logs = []
    again = run_all(**{**kw, "log": logs.append})  # everything saved: nothing re-runs
    assert logs == [] and {k: len(v) for k, v in again.items()} == {k: 3 for k in first}

    s = summarize(again)
    assert s["overall"]["direct"]["acc_mean"] == 1.0 and s["overall"]["direct"]["complete_seeds"] == 2
    assert s["cells"]["toolforge|word_sorting"]["tool_frac"] == 1.0
    md = render(s, {"model": "m", "n": 3, "seeds": [0, 1], "date": "d"})
    assert "100% ± 0%" in md and "macro average" in md


def test_demos_never_overlap_the_test_items():
    for task in TASKS:
        for seed in (0, 1, 2):
            test = {i for i, _ in sample(task, 15, seed)}
            demos = demo_items(task, seed, test)
            assert len(demos) == 3 and test.isdisjoint(i for i, _ in demos)
    prompt = make_prompt("word_sorting", demo_items("word_sorting", 0, set()))
    assert prompt.count("Correct answer:") == 3 and "Build ONE general, reusable tool" in prompt


def test_latm_mode_makes_tools_once_then_the_user_model_reuses_them(tmp_path):
    def brain(system, prompt):
        words = prompt.split("List:", 1)[1].split("\n")[0].split()
        return "Answer: " + " ".join(sorted(words))

    ROLES.clear()
    big, small = tmp_path / "big", tmp_path / "big" / "user_x"
    kw = dict(tasks=["word_sorting"], seeds=[0], n=3, modes=["direct", "direct-user", "latm"], checkpoint=big,
              make_llm=lambda role: ScriptedLLM(brain), make_agent=FakeAgent, log=lambda _: None,
              user_checkpoint=small)
    out = run_all(**kw)
    assert ROLES == ["latm-make", "latm-use"]  # one build from demos, then one agent for all test items
    assert all(r["reused"] == ["sort_words"] and r["user_tokens"] == 40 for r in out["latm__word_sorting__s0"])
    assert (small / "latm-make__word_sorting__s0.json").exists()
    assert (small / "direct-user__word_sorting__s0.jsonl").exists()
    assert not (big / "latm__word_sorting__s0.jsonl").exists()  # user-model results never mix with the big model's

    run_all(**kw)
    assert ROLES == ["latm-make", "latm-use"]  # resumed: nothing rebuilt or re-run

    rows, makes = load_saved(big, small)
    s = summarize(rows, makes)
    assert s["make"]["word_sorting"] == {"seeds": 1, "built": 1, "tokens_mean": 60, "demo_correct": 1}
    assert s["cells"]["latm|word_sorting"]["built_during_use"] == 0
    md = render(s, {"model": "nvidia:big/model-120b", "user_model": "nvidia:meta/small-8b", "n": 3, "seeds": [0],
                    "date": "d"})
    assert "small + big's tools" in md and "small = `small-8b`" in md and "(40 small + 20 big)" in md


def test_network_errors_are_retried_not_graded(tmp_path):
    import httpx

    calls = {"n": 0}

    def brain(system, prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise httpx.ReadTimeout("The read operation timed out")
        words = prompt.split("List:", 1)[1].split("\n")[0].split()
        return "Answer: " + " ".join(sorted(words))

    kw = dict(tasks=["word_sorting"], seeds=[0], n=3, modes=["direct"], checkpoint=tmp_path,
              make_llm=lambda role: ScriptedLLM(brain), make_agent=FakeAgent, log=lambda _: None)
    first = run_all(**kw)["direct__word_sorting__s0"]
    assert len(first) == 2 and all(r["correct"] for r in first)  # the timed-out item was skipped, not failed
    path = tmp_path / "direct__word_sorting__s0.jsonl"
    path.write_text(path.read_text() + '{"index": 999, "answer": "ERROR ReadTimeout: x", "correct": false}\n')
    again = run_all(**kw)["direct__word_sorting__s0"]  # retries the missing item; drops the saved network error
    assert len(again) == 3 and all(r["correct"] for r in again) and 999 not in {r["index"] for r in again}


def test_a_dead_network_stops_the_run_instead_of_skipping_everything(tmp_path):
    import httpx
    import pytest

    from evals.bbh_benchmark import NetworkDown

    def brain(system, prompt):
        raise httpx.ConnectError("getaddrinfo failed")

    with pytest.raises(NetworkDown):
        run_all(tasks=["word_sorting"], seeds=[0], n=5, modes=["direct"], checkpoint=tmp_path,
                make_llm=lambda role: ScriptedLLM(brain), make_agent=FakeAgent, log=lambda _: None)
    assert not (tmp_path / "direct__word_sorting__s0.jsonl").exists()  # nothing was graded as wrong


def test_an_item_that_keeps_timing_out_is_eventually_graded_wrong(tmp_path):
    import httpx

    def brain(system, prompt):
        if "List: zebra apple" in prompt:
            raise httpx.ReadTimeout("model too slow")
        words = prompt.split("List:", 1)[1].split("\n")[0].split()
        return "Answer: " + " ".join(sorted(words))

    examples = load_task("word_sorting")
    slow = next(i for i, _ in sample("word_sorting", 3, 0))
    real = examples[slow]["input"]
    import evals.bbh_benchmark as B

    original = B.load_task
    B.load_task = lambda task: [dict(ex, input="Sort the following words alphabetically: List: zebra apple")
                                if j == slow else ex for j, ex in enumerate(original(task))]
    try:
        kw = dict(tasks=["word_sorting"], seeds=[0], n=3, modes=["direct"], checkpoint=tmp_path,
                  make_llm=lambda role: ScriptedLLM(brain), make_agent=FakeAgent, log=lambda _: None)
        for _ in range(B.MAX_TIMEOUTS):
            assert len(run_all(**kw)["direct__word_sorting__s0"]) == 2  # skipped, retried next run
        rows = run_all(**kw)["direct__word_sorting__s0"]
    finally:
        B.load_task = original
    assert len(rows) == 3 and real
    stuck = next(r for r in rows if r["index"] == slow)
    assert not stuck["correct"] and "no reply after 3 attempts" in stuck["answer"]


SOLVE_SCHEMA = {"type": "object", "required": ["problem"], "properties": {"problem": {"type": "string"}}}
NAIVE = ('def solve(problem: str) -> str:\n    """Sort the words after List:."""\n'
         '    return " ".join(sorted(problem.split("List:", 1)[1].split()))\n')
FIXED = ('def solve(problem: str) -> str:\n    """Sort the words after List:, up to the end of that line."""\n'
         '    return " ".join(sorted(problem.split("List:", 1)[1].split("\\n")[0].split()))\n')
OWN_TESTS = [{"kwargs": {"problem": "List: b a"}, "expected": "a b"},
             {"kwargs": {"problem": "List: c a b"}, "expected": "a b c"},
             {"kwargs": {"problem": "List: z"}, "expected": "z"}]


DEDUP = ('def solve(problem: str) -> str:\n    """Sort the distinct words after List:."""\n'
         '    return " ".join(sorted(set(problem.split("List:", 1)[1].split("\\n")[0].split())))\n')


def _protocol2_setup(tmp_path, *drafts):
    from fakes import Brain, draft

    from toolforge.agent import Toolforge
    from toolforge.config import Settings
    from toolforge.embeddings import HashingEmbedder
    from toolforge.registry import Registry

    brain = Brain().plan("Build ONE general, reusable tool", ("solve", "Parse and solve a word sorting problem."))
    brain.will_write("solve", *[draft(code, name="solve", parameters=SOLVE_SCHEMA, tests=OWN_TESTS,
                                      description="Parse and solve a word sorting problem.") for code in drafts])
    brain._executor = lambda prompt: ({"action": "final", "answer": "done"} if "result" in prompt.split(
        "Previous steps:", 1)[1] else {"action": "call", "tool": "solve", "args": {"problem": "<<TASK>>"}})

    def make_agent(db, role):
        s = Settings(provider="scripted", embedder="hashing", sandbox="process", db_path=db, differential=False,
                     user_model=None, min_needs=1)
        return Toolforge(s, llm=brain.llm(), embedder=HashingEmbedder(), registry=Registry(db))

    return brain, make_agent


def test_protocol_2_adds_the_demos_as_exact_tests_so_a_wrong_parser_never_registers(tmp_path):
    import evals.bbh_benchmark as B
    from toolforge.registry import Registry

    brain, make_agent = _protocol2_setup(tmp_path, NAIVE, FIXED)
    items = B.sample("word_sorting", 3, 0)
    B._make_tools(make_agent, "word_sorting", 0, items, tmp_path, log=lambda _: None, protocol=2)
    rec = json.loads((tmp_path / "latm-make__word_sorting__s0.json").read_text())
    assert rec["created"] == ["solve"] and rec["demos_before"] == rec["checked"] == 3 + B.VALIDATION
    tool = Registry(str(tmp_path / "latm__word_sorting__s0.db")).get("solve")
    assert tool.version == 1 and "split(\"\\n\")" in tool.code  # NAIVE failed the demo tests in verification
    assert len(tool.tests) == 3 + 3  # its own tests + the 3 demos, added by code
    assert "Correct answer:" in brain.prompts["planner"][0] and "<<<" in brain.prompts["planner"][0]


def test_protocol_2_repairs_a_tool_that_fits_the_demos_but_fails_held_out_examples(tmp_path, monkeypatch):
    import evals.bbh_benchmark as B
    from toolforge.registry import Registry

    held = [(900 + i, {"input": f"Sort the following words alphabetically: List: b a {w} a", "target": t})
            for i, (w, t) in enumerate([("c", "a a b c"), ("d", "a a b d")])]
    monkeypatch.setattr(B, "validation_items", lambda task, seed, exclude, k=B.VALIDATION: held)
    _, make_agent = _protocol2_setup(tmp_path, DEDUP, FIXED)
    items = B.sample("word_sorting", 3, 0)
    B._make_tools(make_agent, "word_sorting", 0, items, tmp_path, log=lambda _: None, protocol=2)
    rec = json.loads((tmp_path / "latm-make__word_sorting__s0.json").read_text())
    assert rec["checked"] == 5 and rec["demos_before"] == 3 and rec["demos_after"] == 5 and rec["repair_ok"]
    tool = Registry(str(tmp_path / "latm__word_sorting__s0.db")).get("solve")
    assert tool.version == 2 and {"a a b c", "a a b d"} <= {t.expected for t in tool.tests}


def test_protocols_keep_separate_latm_results_but_share_direct_user(tmp_path):
    big, user = tmp_path / "big", tmp_path / "big" / "user_x"
    p2 = user / "p2"
    for folder, name in ((user, "direct-user__word_sorting__s0"), (user, "latm__word_sorting__s0"),
                         (p2, "latm__word_sorting__s0")):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.jsonl").write_text(json.dumps({"index": 1, "correct": folder == p2, "answer": "x",
                                                          "tokens": 1, "latency_s": 0, "tool_calls": 0}) + "\n")
    rows, _ = load_saved(big, user, p2)
    assert rows["latm__word_sorting__s0"][0]["correct"] is True  # protocol 2's own run
    assert "direct-user__word_sorting__s0" in rows
    rows1, _ = load_saved(big, user)
    assert rows1["latm__word_sorting__s0"][0]["correct"] is False  # protocol 1 unchanged


def test_a_slow_make_step_is_skipped_for_now_and_rebuilt_clean_next_run(tmp_path):
    import httpx

    class SlowMaker(FakeAgent):
        def __init__(self, path, role="maker"):
            super().__init__(path, role)
            self.path = path

        def run(self, task):
            if self.role == "latm-make":
                open(self.path, "w").close()  # a half-built library file now exists on disk
                raise httpx.ReadTimeout("model too slow")
            return super().run(task)

    def brain(system, prompt):
        words = prompt.split("List:", 1)[1].split("\n")[0].split()
        return "Answer: " + " ".join(sorted(words))

    small = tmp_path / "user_x"
    lines: list[str] = []
    kw = dict(tasks=["word_sorting"], seeds=[0], n=3, modes=["direct-user", "latm"], checkpoint=tmp_path,
              make_llm=lambda role: ScriptedLLM(brain), log=lines.append, user_checkpoint=small)
    out = run_all(make_agent=SlowMaker, **kw)  # does not raise NetworkDown: a slow model is not a dead network
    assert len(out["direct-user__word_sorting__s0"]) == 3 and not out.get("latm__word_sorting__s0")
    assert any("next run builds it again" in line for line in lines)
    assert not (small / "latm-make__word_sorting__s0.json").exists()  # no "built nothing" record
    assert not (small / "latm__word_sorting__s0.db").exists()  # the half-built library is gone
