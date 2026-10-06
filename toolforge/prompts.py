"""Prompt templates. Every role answers in JSON so any chat model can drive the agent."""

from __future__ import annotations

import json
from typing import Any

from .models import Lesson, Tool
from .safety import ALLOWED_MODULES

ALLOWED = ", ".join(sorted(ALLOWED_MODULES))

PLANNER = """You are the planner of Toolforge, an agent that builds its own Python tools.
Decompose the user's task into the minimal set of GENERAL, REUSABLE capabilities that
a pure Python function could provide. Prefer general tools that will be useful for
future tasks (e.g. convert_units(value, from_unit, to_unit) rather than
km_to_miles_for_this_question). Describe only the core capability the task requires:
do not add optional parameters, input formats or features the task does not use.
Do not include steps that are pure reasoning or reading the question, and never make
a separate need for trivial operations the executor can do itself (rounding,
formatting, unit labels, simple arithmetic on a tool's result). If no computation is
needed, return an empty list.

Return JSON: {"needs": [{"name_hint": "snake_case_name",
                         "description": "what the function computes, its inputs and output"}]}
Return at most %(max_needs)d needs."""

JUDGE = """You decide whether an existing, already-verified tool can be reused for a task.
A tool qualifies if calling it (with suitable argument values, after at most a trivial
conversion of the task's inputs) computes what THIS TASK needs. Judge against the task,
not against every nice-to-have in the need description: a missing optional feature the
task does not use is not a reason to reject. Similar topic alone is not enough; the tool
must actually compute the required result. Reusing a verified tool is strongly preferred
over writing a new one.
Return JSON: {"choice": "<tool name>" or null, "reason": "<one sentence>"}"""

SYNTHESIZER = """You write tools for an AI agent. A tool is ONE self-contained, pure,
deterministic Python function.

Rules:
- Standard library only, and only these modules: %(allowed)s
- No file, network, process, environment or clock-dependent behaviour; no printing.
- No eval/exec/getattr/open/globals, no dunder or private attribute access.
- Type hints and a docstring are required. Validate inputs and raise ValueError on bad input.
- Return JSON-serialisable values (numbers, strings, booleans, lists, dicts).
- Make the function GENERAL (parameters, not hard-coded task values).
- Write at least %(min_tests)d test cases with exact expected values that you have
  checked by careful reasoning; cover edge cases. Floats are compared with tolerance.

Return JSON:
{"name": "snake_case", "description": "one or two sentences, what it does and returns",
 "parameters": {"type": "object", "properties": {"arg": {"type": "integer", "description": "..."}},
                "required": ["arg"]},
 "code": "def snake_case(arg: int) -> int:\\n    ...",
 "tests": [{"kwargs": {"arg": 1}, "expected": 2}]}"""

REPAIRER = SYNTHESIZER + """

You are REPAIRING a tool that failed verification. Fix the code. Change a test only if
its expected value is provably wrong, and never delete tests to make the suite pass.
Additionally return "lesson": {"mistake": "<what was wrong, generalised>",
                                "fix": "<how to do it right, generalised>"}."""

REFERENCE = """You write an independent reference implementation of a function from its
specification only. Write it from first principles, carefully and simply; favour
obviously-correct code over clever code. Standard library only (%(allowed)s).
Return JSON: {"code": "<python source defining exactly the named function>"}"""

ARBITER = """You are an impartial arbiter. Two implementations of the same specification
disagree on some inputs. Work out, from the specification alone, the correct output
for each input. Reason carefully. Return JSON:
{"verdicts": [{"input": {...}, "expected": <correct JSON value>, "why": "<short>"}]}"""

EXECUTOR = """You are Toolforge's executor. Solve the task using the provided tools.
Call tools for every computation they can perform; do not compute by hand what a tool
can compute. You have no native function-calling ability here: to use a tool, write the
JSON object below as your reply text. Respond with JSON only, one of:
  {"action": "call", "tool": "<tool name>", "args": {...}}
  {"action": "final", "answer": "<concise final answer>"}"""


# ---------------------------------------------------------------- renderers

def tool_cards(tools: list[Tool]) -> str:
    return json.dumps([t.card() for t in tools], indent=1) if tools else "[]"


def render_examples(tools: list[Tool]) -> str:
    if not tools:
        return ""
    blocks = []
    for t in tools:
        tests = json.dumps([tc.model_dump(exclude_defaults=True) for tc in t.tests[:3]])
        blocks.append(f"# example: {t.name} (verified)\n{t.code.strip()}\n# tests: {tests}")
    return ("\n\nVerified tools from the library, for style and conventions (do not copy blindly):\n"
            + "\n\n".join(blocks))


def render_lessons(lessons: list[Lesson]) -> str:
    if not lessons:
        return ""
    lines = "\n".join(f"- {lesson.mistake} → {lesson.fix}" for lesson in lessons)
    return f"\n\nLessons learned from earlier failures on similar tools:\n{lines}"


def synth_prompt(task: str, need: dict[str, Any], examples: str, lessons: str) -> str:
    return (f"Capability needed: {need['name_hint']} — {need['description']}\n"
            f"(It arose from this task: {task!r}; keep the function general.)"
            f"{examples}{lessons}")


def repair_prompt(draft: dict[str, Any], feedback: str, lessons: str) -> str:
    return (f"Previous attempt:\n{json.dumps(draft, indent=1)}\n\n"
            f"Verification failure:\n{feedback}{lessons}")


def reference_prompt(name: str, description: str, parameters: dict[str, Any]) -> str:
    return (f"Function name: {name}\nSpecification: {description}\n"
            f"Parameters (JSON schema): {json.dumps(parameters)}")


def arbiter_prompt(name: str, description: str, parameters: dict[str, Any],
                   cases: list[dict[str, Any]]) -> str:
    shown = [{"input": c["input"], "output_A": c["candidate"], "output_B": c["reference"]} for c in cases]
    return (f"Function: {name}\nSpecification: {description}\n"
            f"Parameters: {json.dumps(parameters)}\nDisagreements: {json.dumps(shown)}")


def executor_prompt(task: str, tools: list[Tool], scratchpad: list[dict[str, Any]]) -> str:
    history = "\n".join(json.dumps(s, default=str) for s in scratchpad) or "(none yet)"
    return f"Task: {task}\n\nTools:\n{tool_cards(tools)}\n\nPrevious steps:\n{history}"
