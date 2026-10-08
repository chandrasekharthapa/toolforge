"""Data models shared across the package."""

from __future__ import annotations

import json
import keyword
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TestCase(BaseModel):
    __test__ = False  # stop pytest from collecting this class

    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None

    @model_validator(mode="before")
    @classmethod
    def _accept_common_shapes(cls, data: Any) -> Any:
        """Models describe tests in many ways; normalise them instead of failing a round."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for alias in ("input", "inputs", "arguments", "params", "parameters"):
            if alias in data and "args" not in data and "kwargs" not in data:
                value = data.pop(alias)
                data["kwargs" if isinstance(value, dict) else "args"] = value
        for alias in ("output", "expected_output", "result", "returns"):
            if alias in data and "expected" not in data:
                data["expected"] = data.pop(alias)
        if isinstance(data.get("args"), dict):  # {"args": {"a": 1}} means keyword arguments
            data["kwargs"] = {**data.pop("args"), **(data.get("kwargs") or {})}
        elif data.get("args") is not None and not isinstance(data["args"], list):
            data["args"] = [data["args"]]
        return data


class ToolDraft(BaseModel):
    """What the synthesizer proposes: one pure Python function plus its contract."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    code: str
    tests: list[TestCase] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        v = re.sub(r"[^a-zA-Z0-9_]", "_", v.strip()).strip("_").lower()
        if not v or v[0].isdigit() or keyword.iskeyword(v):
            raise ValueError(f"invalid tool name {v!r}")
        return v

    @field_validator("parameters")
    @classmethod
    def _object_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        v = dict(v or {})
        v.setdefault("type", "object")
        v.setdefault("properties", {})
        return v

    @field_validator("code")
    @classmethod
    def _unescape_flattened_code(cls, v: str) -> str:
        """Some models JSON-escape the source twice, so it arrives as ONE line full of literal ``\\n``.
        A real function spans several lines, so a newline-free body with ``\\n`` in it is unescaped."""
        if "\n" not in v.strip() and "\\n" in v:
            try:  # undo exactly one level of JSON string escaping
                return json.loads('"' + v.replace('"', '\\"').replace('\\\\"', '\\"') + '"')
            except ValueError:
                return v.replace("\\n", "\n").replace("\\t", "\t")
        return v

    @model_validator(mode="after")
    def _unwrap_dict_args(self) -> ToolDraft:
        """``args: [{"start": .., "end": ..}]`` for a multi-parameter function is almost always
        keyword arguments wrapped in a list by mistake."""
        names = set(self.parameters.get("properties", {}))
        for t in self.tests:
            if (len(t.args) == 1 and isinstance(t.args[0], dict) and not t.kwargs and t.args[0]
                    and set(t.args[0]) <= names and (len(names) > 1 or set(t.args[0]) == names)):
                first = next(iter(self.parameters["properties"].values()), {})
                if len(names) == 1 and first.get("type") == "object":
                    continue  # a single dict-typed parameter really does take a dict
                t.kwargs, t.args = t.args[0], []
        return self

    def signature_text(self) -> str:
        props = self.parameters.get("properties", {})
        return ", ".join(f"{k}: {p.get('type', 'any')}" for k, p in props.items())

    def search_text(self) -> str:
        """The text that represents this tool in the retrieval index."""
        return f"{self.name.replace('_', ' ')}. {self.description} Inputs: {self.signature_text()}"


class Verification(BaseModel):
    """Evidence attached to every registered tool."""

    tests_passed: int = 0
    tests_total: int = 0
    differential: Literal["passed", "skipped", "failed"] = "skipped"
    fuzz_cases: int = 0
    disagreements: int = 0
    arbitrated_tests: int = 0  # regression tests added from arbiter verdicts on disagreements
    repair_rounds: int = 0


class Tool(ToolDraft):
    id: int
    version: int = 1
    status: Literal["active", "superseded", "retired"] = "active"
    uses: int = 0
    successes: int = 0
    failures: int = 0
    created_at: str = ""
    origin_task: str = ""
    verification: Verification = Field(default_factory=Verification)
    #: a worked call ({"args", "result"}) from the run that built the tool, shown to callers
    example: dict[str, Any] | None = None

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.uses if self.uses else None

    def card(self) -> dict[str, Any]:
        """Compact view the planner/executor sees."""
        card = {"name": self.name, "description": self.description, "parameters": self.parameters}
        if self.example:
            card["example_call"] = self.example
        return card


class Need(BaseModel):
    name_hint: str
    description: str

    def query(self) -> str:
        return f"{self.name_hint.replace('_', ' ')}. {self.description}"


class Lesson(BaseModel):
    id: int | None = None
    need: str
    mistake: str
    fix: str
    created_at: str = ""


class RunResult(BaseModel):
    task: str
    answer: str
    created: list[str] = Field(default_factory=list)
    reused: list[str] = Field(default_factory=list)
    failed_needs: list[str] = Field(default_factory=list)
    #: tools sent back to the maker after failing in real use, and re-verified as a new version
    repaired: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: of the tokens above, those spent on the tool-USER model (0 when there is no maker/user split)
    user_tokens: int = 0
    latency_s: float = 0.0
    trace: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
