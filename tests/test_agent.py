"""End-to-end tests of the agent graph with a scripted LLM."""
import json

from fakes import DAYS_BETWEEN, DAYS_BETWEEN_BUGGY, Brain, draft

NEED = ("days_between", "Number of days between two ISO dates.")


def brain_with_reference() -> Brain:
    b = Brain().plan("How many days", NEED)
    b.references["days_between"] = DAYS_BETWEEN
    return b


def test_creates_verifies_registers_and_executes(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain)

    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    assert result.created == ["days_between"]
    assert result.reused == []
    assert "46" in result.answer
    tool = forge.registry.get("days_between")
    assert tool.verification.differential == "passed"
    assert tool.verification.fuzz_cases > 0
    assert tool.uses == 1 and tool.successes == 1
    assert forge.registry.stats()["runs"] == 1


def test_second_task_reuses_without_writing_code(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    brain.judge_choice = "days_between"  # grey-zone similarity: the judge confirms the match
    forge = make_forge(brain)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")
    synth_calls = brain.roles.count("synth")

    result = forge.run("How many days between 2023-12-25 and 2024-02-14?")

    assert result.reused == ["days_between"] and result.created == []
    assert "51" in result.answer
    assert brain.roles.count("synth") == synth_calls  # no new code was written
    assert brain.prompts["judge"][0].startswith("Task: How many days between 2023-12-25")
    assert forge.registry.stats()["reuse_rate"] == 0.5


def test_high_similarity_reuses_without_asking_the_judge(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain, reuse_threshold=0.5)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")

    result = forge.run("How many days between 2023-12-25 and 2024-02-14?")

    assert result.reused == ["days_between"]
    assert "judge" not in brain.roles
    match = next(e for e in result.trace if e["node"] == "match")
    assert match["reason"].startswith("cosine")


def test_judge_rejection_forges_a_new_tool(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    brain.plan("weekdays", ("business_days_between", "Number of business days between two dates."))
    forge = make_forge(brain, differential=False)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")

    result = forge.run("How many weekdays between 2024-01-15 and 2024-03-01?")

    assert "judge" in brain.roles
    assert result.reused == [] and result.failed_needs == ["business_days_between"]


def test_failed_tests_trigger_repair_and_store_a_lesson(make_forge):
    lesson = {"mistake": "counted both endpoints", "fix": "use abs((end - start).days)"}
    brain = brain_with_reference().will_write(
        "days_between", draft(DAYS_BETWEEN_BUGGY), draft(lesson=lesson))
    forge = make_forge(brain)

    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    assert result.created == ["days_between"]
    assert brain.roles.count("repair") == 1
    assert "got 47, expected 46" in brain.prompts["repair"][0]
    assert forge.registry.get("days_between").verification.repair_rounds == 1
    assert [x.mistake for x in forge.registry.list_lessons()] == ["counted both endpoints"]


def test_lessons_are_retrieved_into_later_synthesis(make_forge):
    lesson = {"mistake": "counted both endpoints of a date range", "fix": "use abs((end - start).days)"}
    brain = brain_with_reference().will_write(
        "days_between", draft(DAYS_BETWEEN_BUGGY), draft(lesson=lesson))
    brain.plan("weekdays", ("business_days_between", "Number of business days between two dates."))
    forge = make_forge(brain)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")

    forge.run("How many weekdays between 2024-01-15 and 2024-03-01?")

    rag_prompt = brain.prompts["synth"][-1]
    assert "Capability needed: business_days_between" in rag_prompt
    assert "Lessons learned" in rag_prompt and "counted both endpoints" in rag_prompt
    assert "example: days_between (verified)" in rag_prompt


def test_rag_off_removes_examples_and_lessons(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    brain.plan("weekdays", ("business_days_between", "Number of business days between two dates."))
    forge = make_forge(brain, rag=False)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")
    forge.run("How many weekdays between 2024-01-15 and 2024-03-01?")
    assert "verified" not in brain.prompts["synth"][-1]


def test_unsafe_code_is_rejected_statically_then_repaired(make_forge):
    evil = "import os\n\ndef days_between(start: str, end: str) -> int:\n    os.system('id')\n    return 0\n"
    brain = brain_with_reference().will_write("days_between", draft(evil), draft())
    forge = make_forge(brain)

    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    assert result.created == ["days_between"]
    assert "module 'os' is not on the allow-list" in brain.prompts["repair"][0]
    stages = [e.get("stage") for e in result.trace if e["node"] == "verify"]
    assert stages == ["static", None]


def test_gives_up_after_repair_budget(make_forge):
    bad = draft(DAYS_BETWEEN_BUGGY)
    brain = brain_with_reference().will_write("days_between", bad, bad, bad)
    forge = make_forge(brain, max_repairs=2)

    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    assert result.failed_needs == ["days_between"] and result.created == []
    assert forge.registry.get("days_between") is None
    assert "could not" in result.answer


def test_differential_catches_bug_the_unit_tests_missed(make_forge):
    # Tests only ever use start <= end, so the missing abs() passes them…
    sneaky = DAYS_BETWEEN.replace("abs((b - a).days)", "(b - a).days")
    forward_only = [t for t in draft()["tests"] if t["kwargs"]["start"] <= t["kwargs"]["end"]]
    forward_only.append({"kwargs": {"start": "2020-01-01", "end": "2020-12-31"}, "expected": 365})

    def arbiter(payload):  # …but the independent reference disagrees on reversed ranges
        from datetime import date
        return {"verdicts": [{"input": d["input"], "expected": abs(
            (date.fromisoformat(d["input"]["end"]) - date.fromisoformat(d["input"]["start"])).days)}
            for d in payload]}

    brain = brain_with_reference().will_write(
        "days_between", draft(sneaky, tests=forward_only), draft(tests=forward_only))
    brain.arbiter = arbiter
    forge = make_forge(brain)

    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    verify_events = [e for e in result.trace if e["node"] == "verify"]
    assert verify_events[0]["stage"] == "differential"
    assert "An independent implementation disagreed" in brain.prompts["repair"][0]
    tool = forge.registry.get("days_between")
    assert len(tool.tests) > len(forward_only)  # disagreements became regression tests
    assert tool.verification.differential == "passed"


def test_name_collision_does_not_clobber_unrelated_tool(make_forge):
    other_schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    other = draft("def days_between(n: int) -> int:\n    return n * 7\n", name="days_between",
                  parameters=other_schema, tests=[{"kwargs": {"n": i}, "expected": 7 * i} for i in (1, 2, 3)])
    brain = Brain().plan("weeks", ("weeks_to_days", "Convert weeks to days.")).plan("How many days", NEED)
    brain.will_write("weeks_to_days", other).will_write("days_between", draft())
    forge = make_forge(brain, differential=False, reuse_threshold=1.01, consider_threshold=1.01, field_repair=False)

    forge.run("How many days in 3 weeks?")
    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    assert result.created == ["days_between_2"]
    names = sorted(t.name for t in forge.registry.list_tools())
    assert names == ["days_between", "days_between_2"]


def test_executor_survives_unusable_model_replies(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain, max_exec_steps=3)
    real = brain._executor
    replies = iter([None, None])

    def flaky(prompt):
        if next(replies, "real") is None:
            from toolforge.llm import LLMGenerationError
            raise LLMGenerationError("provider rejected the generation")
        return real(prompt)

    brain._executor = flaky
    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")
    assert result.created == ["days_between"]  # the run completes instead of crashing


def test_verify_event_reports_arbitrated_tests(make_forge, capsys):
    from toolforge.cli import _event

    sneaky = DAYS_BETWEEN.replace("abs((b - a).days)", "(b - a).days")
    forward = [{"kwargs": {"start": "2024-01-15", "end": "2024-03-01"}, "expected": 46},
               {"kwargs": {"start": "2024-02-28", "end": "2024-03-01"}, "expected": 2},
               {"kwargs": {"start": "2020-01-01", "end": "2020-12-31"}, "expected": 365}]

    def arbiter(cases):
        from datetime import date
        return {"verdicts": [{"input": c["input"], "expected": abs((date.fromisoformat(c["input"]["end"])
                              - date.fromisoformat(c["input"]["start"])).days)} for c in cases]}

    brain = brain_with_reference().will_write("days_between", draft(sneaky, tests=forward), draft(tests=forward))
    brain.arbiter = arbiter
    make_forge(brain).run("How many days between 2024-01-15 and 2024-03-01?", on_event=_event)
    out = capsys.readouterr().out
    assert "differential passed on" in out and "regression tests" not in out.split("register")[-1]
    tool_line = [line for line in out.splitlines() if line.startswith("verify") and "✓" in line][0]
    assert "disagreements" not in tool_line or "regression tests added" in tool_line


def test_executor_must_use_a_tool_before_answering(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain)
    real = brain._executor
    state = {"lazy": True}

    def lazy_then_real(prompt):
        if state.pop("lazy", False):
            return {"action": "final", "answer": "I think it is 46."}  # skipped the tool
        return real(prompt)

    brain._executor = lazy_then_real
    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")

    assert len(result.tool_calls) == 1 and result.tool_calls[0]["result"] == 46
    assert "answered without calling any tool" in brain.prompts["executor"][1]


def test_executor_answers_directly_when_no_tool_was_needed(make_forge):
    brain = Brain()  # planner returns no needs
    result = make_forge(brain).run("Say hello")
    assert result.tool_calls == [] and brain.roles.count("executor") == 1


def _two_forges(make_forge, second):
    brain = Brain().plan("How many days", NEED).plan("How many weeks", NEED)
    brain.will_write("days_between", draft(), second)
    forge = make_forge(brain, differential=False, reuse_threshold=1.01, consider_threshold=1.01, field_repair=False)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")
    return forge, forge.run("How many weeks between 2024-01-15 and 2024-03-01?")


def test_same_name_different_job_never_replaces_the_tool(make_forge):
    weeks = DAYS_BETWEEN.replace("return abs((b - a).days)", "return abs((b - a).days) // 7")
    weeks_tests = [{"kwargs": {"start": "2024-01-01", "end": "2024-01-15"}, "expected": 2},
                   {"kwargs": {"start": "2024-01-15", "end": "2024-01-01"}, "expected": 2},
                   {"kwargs": {"start": "2024-01-01", "end": "2024-03-01"}, "expected": 8}]
    forge, result = _two_forges(make_forge, draft(weeks, tests=weeks_tests))

    assert result.created == ["days_between_2"]  # renamed: it fails the original's tests
    original = forge.registry.get("days_between")
    assert original.version == 1 and original.status == "active" and "// 7" not in original.code
    register = next(e for e in result.trace if e["node"] == "register")
    assert register["how"] == "renamed"


def test_compatible_improvement_becomes_next_version_and_inherits_tests(make_forge):
    better = DAYS_BETWEEN.replace('"""Absolute', '"""(v2) Absolute')
    own_tests = [{"kwargs": {"start": "2020-01-01", "end": "2020-12-31"}, "expected": 365},
                 {"kwargs": {"start": "2021-01-01", "end": "2021-01-01"}, "expected": 0},
                 {"kwargs": {"start": "1999-12-31", "end": "2000-01-01"}, "expected": 1}]
    forge, result = _two_forges(make_forge, draft(better, tests=own_tests))

    assert result.created == ["days_between"]
    tool = forge.registry.get("days_between")
    assert tool.version == 2 and "(v2)" in tool.code
    assert len(tool.tests) == 6 and tool.verification.tests_total == 6  # 3 own + 3 inherited from v1
    assert [v.status for v in forge.registry.versions("days_between")] == ["superseded", "active"]


def test_maker_user_split_routes_roles_and_counts_user_tokens(tmp_path):
    """LATM split: the user model plans and executes; the maker writes and verifies tools."""
    from toolforge.agent import Toolforge
    from toolforge.config import Settings
    from toolforge.embeddings import HashingEmbedder
    from toolforge.registry import Registry

    maker = brain_with_reference().will_write("days_between", draft())
    user = Brain().plan("How many days", NEED)
    s = Settings(provider="scripted", embedder="hashing", sandbox="process", db_path=str(tmp_path / "l.db"),
                 fuzz_cases=10, plan_with_library=True)
    forge = Toolforge(s, llm=maker.llm(), user_llm=user.llm(), embedder=HashingEmbedder(),
                      registry=Registry(s.db_path))
    first = forge.run("How many days between 2024-01-15 and 2024-03-01?")
    assert first.created == ["days_between"] and "46" in first.answer
    assert set(user.roles) == {"planner", "executor"}
    assert "planner" not in maker.roles and "executor" not in maker.roles and "synth" in maker.roles
    assert 0 < first.user_tokens < first.total_tokens

    maker_calls = len(maker.roles)
    second = forge.run("How many days between 2023-12-25 and 2024-02-14?")
    assert second.reused == ["days_between"] and "51" in second.answer
    assert "Verified tools already in the library" in user.prompts["planner"][-1]
    assert len(maker.roles) == maker_calls  # reuse picked by name: the maker was never called
    assert second.user_tokens == second.total_tokens
    match = next(e for e in second.trace if e["node"] == "match")
    assert match["reason"] == "planner chose this library tool by name"


def test_without_a_user_model_one_llm_does_everything(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain)
    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")
    assert forge.user_llm is None and result.user_tokens == 0
    assert {"planner", "synth", "executor"} <= set(brain.roles)


def test_building_run_leaves_a_usage_example_for_later_callers(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    brain.judge_choice = "days_between"
    forge = make_forge(brain)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")
    example = forge.registry.get("days_between").example
    assert example == {"args": {"start": "2024-01-15", "end": "2024-03-01"}, "result": 46}
    forge.run("How many days between 2023-12-25 and 2024-02-14?")
    assert '"example_call"' in brain.prompts["executor"][-1]  # the reusing run is shown the worked call
    assert forge.registry.get("days_between").example == example  # first example is kept


def test_repeating_a_failed_call_is_refused_and_the_last_step_asks_for_an_answer(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain, max_exec_steps=4)
    brain._executor = lambda prompt: {"action": "call", "tool": "days_between", "args": {"start": "bad", "end": "x"}}
    result = forge.run("How many days between 2024-01-15 and 2024-03-01?")
    prompts = brain.prompts["executor"]
    assert "this exact call already failed" in prompts[2] and "last step" in prompts[-1]
    tool = forge.registry.get("days_between")
    assert tool.failures == 1  # the repeats never reached the sandbox
    assert result.answer.startswith("(no final answer")


def test_min_needs_reasks_a_planner_that_declined(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    asked = []

    def planner(prompt):
        asked.append(prompt)
        return {"needs": [{"name_hint": "days_between", "description": NEED[1]}]} if len(asked) > 1 else {"needs": []}

    brain._planner = planner
    result = make_forge(brain, min_needs=1).run("How many days between 2024-01-15 and 2024-03-01?")
    assert len(asked) == 2 and "explicitly requires building a reusable tool" in asked[1]
    assert result.created == ["days_between"]


def test_repeating_a_successful_call_is_answered_from_memory(make_forge):
    brain = brain_with_reference().will_write("days_between", draft())
    forge = make_forge(brain, max_exec_steps=4)
    brain._executor = lambda prompt: {"action": "call", "tool": "days_between",
                                      "args": {"start": "2024-01-15", "end": "2024-03-01"}}
    forge.run("How many days between 2024-01-15 and 2024-03-01?")
    assert "You already have this result" in brain.prompts["executor"][2]
    assert forge.registry.get("days_between").uses == 1  # the repeats never reached the sandbox


COUNT_SCHEMA = {"type": "object", "required": ["s"], "properties": {"s": {"type": "string"}}}
STRICT = ('def count_open(s: str) -> int:\n    """Count opening brackets."""\n    if " " in s:\n'
          '        raise ValueError("Invalid character \' \' in bracket string.")\n'
          '    return sum(c in "([{<" for c in s)\n')
TOLERANT = 'def count_open(s: str) -> int:\n    """Count opening brackets."""\n    return sum(c in "([{<" for c in s)\n'
COUNT_TESTS = [{"kwargs": {"s": "(("}, "expected": 2}, {"kwargs": {"s": "([<"}, "expected": 3},
               {"kwargs": {"s": ""}, "expected": 0}]


def _bracket_brain(*drafts):
    b = Brain().plan("Count the opening brackets", ("count_open", "Count opening brackets in a string."))
    b.will_write("count_open", *drafts)

    def executor(prompt):
        steps = prompt.split("Previous steps:\n", 1)[1].strip()
        history = [] if steps == "(none yet)" else [json.loads(line) for line in steps.splitlines()]
        ok = [h for h in history if "result" in h]
        if ok:
            return {"action": "final", "answer": f"There are {ok[-1]['result']}."}
        if any("error" in h for h in history):
            return {"action": "final", "answer": "The tool kept failing."}
        return {"action": "call", "tool": "count_open", "args": {"s": "( ( ["}}  # as the task writes it

    b._executor = executor
    return b


def test_tool_that_fails_in_real_use_is_repaired_reverified_and_retried(make_forge):
    fixed = draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA,
                  tests=[*COUNT_TESTS, {"kwargs": {"s": "( ( ["}, "expected": 3}],
                  description="Count opening brackets in a string.")
    brain = _bracket_brain(draft(STRICT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."), fixed)
    forge = make_forge(brain, differential=False)
    result = forge.run("Count the opening brackets in: ( ( [")

    assert result.created == ["count_open"] and result.repaired == ["count_open"]
    assert "3" in result.answer
    assert [c.get("error", "").split(":")[0] for c in result.tool_calls] == ["ValueError", ""]
    tool = forge.registry.get("count_open")
    assert tool.version == 2 and len(tool.tests) == 4  # old tests kept, failing input added
    assert tool.example == {"args": {"s": "( ( ["}, "result": 3}
    assert "this tool passed verification, but it failed in real use" in brain.prompts["repair"][-1].lower()
    assert [e["node"] for e in result.trace].count("execute") == 2


def test_a_failed_field_repair_keeps_the_first_answer_and_stops(make_forge):
    brain = _bracket_brain(draft(STRICT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))  # no fix available
    forge = make_forge(brain, differential=False, max_repairs=1)
    result = forge.run("Count the opening brackets in: ( ( [")
    assert result.answer == "The tool kept failing." and result.repaired == []
    assert forge.registry.get("count_open").version == 1
    assert [e["node"] for e in result.trace].count("execute") == 1


def test_field_repair_is_capped_per_tool(make_forge):
    brain = _bracket_brain(draft(STRICT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))  # repair never succeeds
    forge = make_forge(brain, differential=False, max_repairs=0)
    forge.run("Count the opening brackets in: ( ( [")
    repairs = brain.roles.count("repair")
    forge.run("Count the opening brackets in: ( ( [")  # same broken tool, but its repair budget is spent
    assert repairs == 1 and brain.roles.count("repair") == repairs
    assert forge.registry.get_meta("field_repairs:count_open") == "1"


def test_known_answer_repair_makes_solved_examples_permanent_tests(make_forge):
    from toolforge.models import TestCase

    fixed = draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                  description="Count opening brackets in a string.")
    brain = _bracket_brain(draft(STRICT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."), fixed)
    forge = make_forge(brain, differential=False, field_repair=False)
    forge.run("Count the opening brackets in: ( ( [")
    known = [TestCase(kwargs={"s": "( ( ["}, expected=3), TestCase(kwargs={"s": "{ <"}, expected=2)]
    out = forge.repair_tool("count_open", known, "Fails on the examples' spaced format.")
    tool = forge.registry.get("count_open")
    assert out["ok"] and out["version"] == 2 and tool.version == 2 and out["tokens"] > 0
    assert {"s": "( ( ["} in [t.kwargs for t in tool.tests] and {"s": "{ <"} in [t.kwargs for t in tool.tests]
    assert {"s": "(("} in [t.kwargs for t in tool.tests]  # the old tests are still there


def test_known_answer_repair_that_cannot_pass_leaves_the_tool_alone(make_forge):
    from toolforge.models import TestCase

    brain = _bracket_brain(draft(STRICT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))
    forge = make_forge(brain, differential=False, field_repair=False, max_repairs=1)
    forge.run("Count the opening brackets in: ( ( [")
    out = forge.repair_tool("count_open", [TestCase(kwargs={"s": "( ( ["}, expected=3)], "spaced input")
    assert not out["ok"] and forge.registry.get("count_open").version == 1


def test_task_reference_passes_the_problem_text_verbatim(make_forge):
    brain = _bracket_brain(draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))
    real = brain._executor

    def executor(prompt):
        step = real(prompt)
        if step.get("action") == "call":
            step["args"] = {"s": "<<TASK>>"}
        return step

    brain._executor = executor
    result = make_forge(brain, differential=False).run("Count the opening brackets in: ( ( [")
    assert result.tool_calls[0]["args"] == {"s": "<<TASK>>"} and result.tool_calls[0]["result"] == 3
    from toolforge import prompts

    assert "<<TASK>>" in prompts.EXECUTOR  # the executor is told about the reference


def test_result_reference_copies_the_tool_output_into_the_answer(make_forge):
    brain = _bracket_brain(draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))
    real = brain._executor
    brain._executor = lambda prompt: ({"action": "final", "answer": "<<RESULT>>"} if "result" in prompt.split(
        "Previous steps:", 1)[1] else real(prompt))
    assert make_forge(brain, differential=False).run("Count the opening brackets in: ( ( [").answer == "3"


def test_double_escaped_code_is_unescaped_once():
    from toolforge.models import ToolDraft

    flat = 'def f(x: str) -> str:\\n    """Doc."""\\n    return x + "\\\\n" + \\"!\\"\\n'
    code = ToolDraft(name="f", description="d", code=flat).code
    ns: dict = {}
    exec(code, ns)
    assert ns["f"]("a") == "a\n!" and code.count("\n") == 3


def test_partly_flattened_code_is_unescaped_and_valid_code_is_left_alone():
    from toolforge.models import ToolDraft

    partly = 'def f(x: str) -> str:\\n    """Doc."""\\n    y = x.strip()\n    return y + "!"\n'
    code = ToolDraft(name="f", description="d", code=partly).code
    ns: dict = {}
    exec(code, ns)
    assert ns["f"](" a ") == "a!"
    fine = 'def f(x: str) -> str:\n    return x.replace("\\\\n", "\\n")\n'  # a literal "\\n" the code needs
    assert ToolDraft(name="f", description="d", code=fine).code == fine
    broken = 'def f(x):\\n    return (x\n'  # does not compile either way: left as sent, for the static check
    assert ToolDraft(name="f", description="d", code=broken).code == broken


def test_answer_from_task_tool_ignores_a_garbled_retyping(make_forge):
    brain = _bracket_brain(draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))

    def executor(prompt):
        if "result" in prompt.split("Previous steps:", 1)[1]:
            return {"action": "final", "answer": "There are 7."}  # retyped wrong
        return {"action": "call", "tool": "count_open", "args": {"s": "<<TASK>>"}}

    brain._executor = executor
    garbled = make_forge(brain, differential=False).run("Count the opening brackets in: ( ( [")
    assert garbled.answer == "There are 7."  # default: the model's own answer stands
    brain2 = _bracket_brain(draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                  description="Count opening brackets in a string."))
    brain2._executor = executor
    fixed = make_forge(brain2, db="b.db", differential=False, answer_from_task_tool=True).run(
        "Count the opening brackets in: ( ( [")
    assert fixed.answer == "3"


def test_a_retyped_task_is_snapped_back_to_the_exact_text():
    from toolforge.graph import _is_task_copy

    task = ("Complete the rest of the sequence, making sure that the parentheses are closed properly. "
            "Input: [ < > ] [ {\n\nReply with only the closing brackets needed, separated by spaces.")
    assert _is_task_copy(task.split("\n\n")[0], task)  # the instruction line was dropped
    assert _is_task_copy(task.replace("\n\n", " ").replace("properly.", "properly"), task)
    assert not _is_task_copy("[ < > ] [ {", task)  # just the extracted brackets: a real argument
    assert not _is_task_copy("Complete the rest of the sequence. " * 6, task)
    opening_dropped = "<<<" + task.split(", ", 1)[1] + ">>>"  # wrapped, the first clause left out
    assert _is_task_copy(opening_dropped, task)
    assert not _is_task_copy("Reply with only the closing brackets needed, separated by spaces. " * 2, task)


PROBLEM_SCHEMA = {"type": "object", "required": ["problem"], "properties": {"problem": {"type": "string"}}}
SOLVE_OPEN = ('def count_open(problem: str) -> int:\n    """Count opening brackets after Input:."""\n'
              '    return sum(c in "([{<" for c in problem.split("Input:", 1)[-1])\n')


def bracket_task(seq: str) -> str:
    return ("You are given a sequence of brackets and have to count some of them. "
            f"Count the opening brackets in the sequence. Input: {seq}\nReply with a number only.")


def _solve_brain(tests, executor):
    b = Brain().plan("Count the opening brackets", ("count_open", "Count opening brackets in a problem."))
    b.will_write("count_open", draft(SOLVE_OPEN, name="count_open", parameters=PROBLEM_SCHEMA, tests=tests,
                                     description="Count opening brackets in a problem."))
    b.judge_choice = "count_open"
    b._executor = executor
    return b


def _retyping_executor(prompt):
    """A small model that wraps the problem in <<< >>>, drops its opening sentence, then misreports."""
    task = prompt.split("Task:", 1)[1].split("Tools:", 1)[0].strip()
    if "result" in prompt.split("Previous steps:", 1)[1]:
        return {"action": "final", "answer": "There are 9."}
    return {"action": "call", "tool": "count_open", "args": {"problem": "<<<" + task.split(". ", 1)[1] + ">>>"}}


def test_a_retyped_whole_problem_is_snapped_and_teaches_the_task_reference(make_forge):
    whole = [{"kwargs": {"problem": bracket_task(q)}, "expected": n} for q, n in [("( (", 2), ("[ < >", 2), ("", 0)]]
    forge = make_forge(_solve_brain(whole, _retyping_executor), differential=False, answer_from_task_tool=True)
    result = forge.run(bracket_task("( ( [ ] <"))
    assert result.answer == "4"  # the tool got the exact task, and its answer was not retyped
    assert forge.registry.get("count_open").example == {"args": {"problem": "<<TASK>>"}, "result": 4}


def test_a_tool_verified_on_short_inputs_is_never_handed_the_whole_task(make_forge):
    short = [{"kwargs": {"problem": f"Input: {q}"}, "expected": n} for q, n in [("( (", 2), ("[ < >", 2), ("", 0)]]
    forge = make_forge(_solve_brain(short, _retyping_executor), differential=False, answer_from_task_tool=True)
    assert forge.run(bracket_task("( ( [ ] <")).answer == "There are 9."  # no snapping: the model's answer stands


def test_a_cut_off_argument_is_never_kept_as_the_usage_example(make_forge):
    whole = [{"kwargs": {"problem": bracket_task(q)}, "expected": n} for q, n in [("( (", 2), ("[ < >", 2), ("", 0)]]
    runs = {"n": 0}

    def executor(prompt):
        if "result" in prompt.split("Previous steps:", 1)[1]:
            return {"action": "final", "answer": "<<RESULT>>"}
        runs["n"] += 1
        problem = "<<<Input: " + "( " * 200 + ">>>" if runs["n"] == 1 else "<<TASK>>"  # a long demo, then the task
        return {"action": "call", "tool": "count_open", "args": {"problem": problem}}

    forge = make_forge(_solve_brain(whole, executor), differential=False)
    first = forge.run(bracket_task("( ( [ ] <"))
    assert first.created == ["count_open"] and forge.registry.get("count_open").example is None
    second = forge.run(bracket_task("< < ("))
    assert second.reused == ["count_open"] and second.answer == "3"
    assert forge.registry.get("count_open").example == {"args": {"problem": "<<TASK>>"}, "result": 3}


def test_known_answers_outrank_a_wrong_self_written_test(make_forge):
    from toolforge.models import TestCase

    wrong_own = {"kwargs": {"s": "(\\n("}, "expected": 5}  # the model's own guess, mis-escaped and wrong
    brain = _bracket_brain(draft(TOLERANT, name="count_open", parameters=COUNT_SCHEMA,
                                 tests=[*COUNT_TESTS, wrong_own], description="Count opening brackets in a string."))
    forge = make_forge(brain, differential=False, field_repair=False)
    result = forge.run("Count the opening brackets in: ( ( [",
                       known_tests=[TestCase(kwargs={"s": "( ( ["}, expected=3),
                                    {"kwargs": {"s": "{ <"}, "expected": 2}])

    assert result.created == ["count_open"] and "repair" not in brain.roles  # good code was not "fixed"
    tool = forge.registry.get("count_open")
    kept = [t.kwargs for t in tool.tests]
    assert wrong_own["kwargs"] not in kept and {"s": "( ( ["} in kept and {"s": "{ <"} in kept
    verify = [e for e in result.trace if e["node"] == "verify" and e.get("ok")][0]
    assert verify["dropped_own_tests"] == 1


def test_a_failing_known_answer_is_never_dropped(make_forge):
    from toolforge.models import TestCase

    wrong_own = {"kwargs": {"s": "(\\n("}, "expected": 5}
    brain = _bracket_brain(draft(STRICT, name="count_open", parameters=COUNT_SCHEMA,
                                 tests=[*COUNT_TESTS, wrong_own], description="Count opening brackets in a string."))
    forge = make_forge(brain, differential=False, field_repair=False, max_repairs=1)
    result = forge.run("Count the opening brackets in: ( ( [",
                       known_tests=[TestCase(kwargs={"s": "( ( ["}, expected=3)])
    assert result.created == [] and forge.registry.get("count_open") is None  # the strict tool fails a known answer


def test_a_tool_with_a_demo_main_block_is_registered_without_it(make_forge):
    demo = TOLERANT + '\nif __name__ == "__main__":\n    print(count_open("(("))\n'
    brain = _bracket_brain(draft(demo, name="count_open", parameters=COUNT_SCHEMA, tests=COUNT_TESTS,
                                 description="Count opening brackets in a string."))
    forge = make_forge(brain, differential=False)
    result = forge.run("Count the opening brackets in: ( ( [")
    assert result.created == ["count_open"] and "repair" not in brain.roles
    assert "__main__" not in forge.registry.get("count_open").code
