"""End-to-end tests of the agent graph with a scripted LLM."""

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
    forge = make_forge(brain, differential=False, reuse_threshold=1.01, consider_threshold=1.01)

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
