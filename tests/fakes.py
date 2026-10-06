"""A scripted 'brain' that plays every LLM role deterministically, so the whole agent
graph can be tested offline, with no API key, in a couple of seconds."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from toolforge.llm import ScriptedLLM

DAYS_BETWEEN = '''from datetime import date

def days_between(start: str, end: str) -> int:
    """Absolute number of days between two ISO dates (YYYY-MM-DD)."""
    try:
        a, b = date.fromisoformat(start), date.fromisoformat(end)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid date: {e}") from e
    return abs((b - a).days)
'''

DAYS_BETWEEN_BUGGY = DAYS_BETWEEN.replace("abs((b - a).days)", "(b - a).days + 1")

DAYS_SCHEMA = {"type": "object", "required": ["start", "end"], "properties": {
    "start": {"type": "string", "format": "date", "description": "ISO date"},
    "end": {"type": "string", "format": "date", "description": "ISO date"}}}

DAYS_TESTS = [
    {"kwargs": {"start": "2024-01-15", "end": "2024-03-01"}, "expected": 46},
    {"kwargs": {"start": "2024-03-01", "end": "2024-01-15"}, "expected": 46},
    {"kwargs": {"start": "2024-02-28", "end": "2024-03-01"}, "expected": 2},
]


def draft(code: str = DAYS_BETWEEN, tests: list | None = None, name: str = "days_between",
          description: str = "Absolute number of days between two ISO dates.",
          parameters: dict | None = None, **extra: Any) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": parameters or DAYS_SCHEMA,
            "code": code, "tests": DAYS_TESTS if tests is None else tests, **extra}


class Brain:
    def __init__(self) -> None:
        self.needs: list[tuple[str, list[dict[str, str]]]] = []
        self.drafts: dict[str, deque] = defaultdict(deque)
        self.references: dict[str, str] = {}
        self.arbiter: Callable[[dict], dict] = lambda payload: {"verdicts": []}
        self.judge_choice: str | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.roles: list[str] = []
        self.prompts: dict[str, list[str]] = defaultdict(list)

    # configuration helpers
    def plan(self, task_fragment: str, *needs: tuple[str, str]) -> Brain:
        self.needs.append((task_fragment, [{"name_hint": n, "description": d} for n, d in needs]))
        return self

    def will_write(self, hint: str, *drafts: dict[str, Any]) -> Brain:
        self.drafts[hint].extend(drafts)
        return self

    def llm(self) -> ScriptedLLM:
        return ScriptedLLM(self)

    # dispatcher
    def __call__(self, system: str, prompt: str) -> str | dict:
        role = self._role(system)
        self.roles.append(role)
        self.prompts[role].append(prompt)
        return getattr(self, f"_{role}")(prompt)

    @staticmethod
    def _role(system: str) -> str:
        for prefix, role in (("You are the planner", "planner"), ("You decide whether", "judge"),
                             ("You write an independent", "reference"), ("You are an impartial", "arbiter"),
                             ("You are Toolforge's executor", "executor")):
            if system.startswith(prefix):
                return role
        return "repair" if "You are REPAIRING" in system else "synth"

    def _planner(self, prompt: str):
        for fragment, needs in self.needs:
            if fragment in prompt:
                return {"needs": needs}
        return {"needs": []}

    def _hint(self, prompt: str) -> str:
        m = re.search(r"Capability needed: (\w+)", prompt) or re.search(r'"name": "(\w+)"', prompt)
        return m.group(1) if m else ""

    def _synth(self, prompt: str):
        queue = self.drafts[self._hint(prompt)]
        return queue.popleft() if queue else "not json at all"

    _repair = _synth

    def _judge(self, prompt: str):
        return {"choice": self.judge_choice, "reason": "scripted"}

    def _reference(self, prompt: str):
        name = re.search(r"Function name: (\w+)", prompt).group(1)
        return {"code": self.references.get(name, "def nothing():\n    pass\n")}

    def _arbiter(self, prompt: str):
        payload = json.loads(prompt.split("Disagreements: ", 1)[1])
        return self.arbiter(payload)

    def _executor(self, prompt: str):
        steps = prompt.split("Previous steps:\n", 1)[1].strip()
        history = [] if steps == "(none yet)" else [json.loads(line) for line in steps.splitlines()]
        if not any("call" in h for h in history):  # no tool used yet: call one, like a real model would
            tools = json.loads(prompt.split("Tools:\n", 1)[1].split("\n\nPrevious steps:")[0])
            if not tools:
                return {"action": "final", "answer": "I could not build a tool for this."}
            name = tools[0]["name"]
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", prompt.split("Tools:")[0])
            args = {"start": dates[0], "end": dates[1]} if len(dates) >= 2 else {}
            self.calls.append((name, args))
            return {"action": "call", "tool": name, "args": args}
        last = next(h for h in reversed(history) if "call" in h)
        return {"action": "final", "answer": f"The answer is {last.get('result', last.get('error'))}."}
