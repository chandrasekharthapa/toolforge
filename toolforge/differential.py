"""Differential verification.

Unit tests written by the same model that wrote the code share its blind spots: if
the model misunderstands "business days", its tests encode the same misunderstanding.
Toolforge therefore asks for a *second, independent* implementation from the spec
alone (it never sees the candidate's code), fuzzes both with schema-driven inputs,
and compares outputs.

* value vs. value mismatch  → hard disagreement (blocks registration until resolved)
* error vs. value mismatch  → soft disagreement (reported; usually input-validation
                              strictness, not a wrong answer)
* both error                → agreement

Hard disagreements are sent to an arbiter that sees only the spec and the inputs
and decides the correct outputs. Those become new regression tests, so every
disagreement permanently strengthens the tool's test-suite.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .models import TestCase, ToolDraft
from .sandbox import Sandbox


@dataclass
class DiffReport:
    inputs: int = 0
    agreements: int = 0
    hard: list[dict[str, Any]] = field(default_factory=list)
    soft: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def disagreement_rate(self) -> float:
        return len(self.hard) / self.inputs if self.inputs else 0.0


# ----------------------------------------------------------------------- inputs

def kwargs_from_tests(draft: ToolDraft) -> list[dict[str, Any]]:
    """Normalise every test case to keyword arguments using the schema's order."""
    names = list(draft.parameters.get("properties", {}))
    out = []
    for t in draft.tests:
        kw = dict(zip(names, t.args))
        kw.update(t.kwargs)
        out.append(kw)
    return out


def _random_date(rng: random.Random) -> str:
    return (date(1970, 1, 1) + timedelta(days=rng.randint(0, 60 * 365))).isoformat()


def _value(spec: dict[str, Any], name: str, pool: list[Any], rng: random.Random, depth: int = 0) -> Any:
    if "enum" in spec and spec["enum"]:
        return rng.choice(spec["enum"])
    if pool and rng.random() < 0.5:  # values the tests used are realistic seeds
        v = rng.choice(pool)
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v + rng.choice([-2, -1, 0, 1, 2, 10])
        if isinstance(v, float):
            return round(v * rng.uniform(0.5, 1.5), 4)
        return v
    kind = spec.get("type", "string")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    lo, hi = spec.get("minimum"), spec.get("maximum")
    if kind == "integer":
        lo = int(lo) if lo is not None else -1000
        hi = int(hi) if hi is not None else 1000
        edges = [x for x in (0, 1, 2, -1, 7, 12, 100, lo, hi) if lo <= x <= hi]
        return rng.choice(edges) if rng.random() < 0.4 else rng.randint(lo, hi)
    if kind == "number":
        lo = float(lo) if lo is not None else -1e4
        hi = float(hi) if hi is not None else 1e4
        return round(rng.uniform(lo, hi), 3) if rng.random() < 0.7 else rng.choice([0.0, 1.0, 0.5])
    if kind == "boolean":
        return rng.random() < 0.5
    if kind == "array":
        item = spec.get("items", {"type": "integer"})
        return [_value(item, name, [], rng, depth + 1) for _ in range(rng.randint(0, 6))] if depth < 2 else []
    if kind == "object":
        return {}
    # strings: honour formats and obvious name hints
    if spec.get("format") == "date" or "date" in name.lower():
        return _random_date(rng)
    alphabet = string.ascii_letters + string.digits + "  ,.!?-"
    return rng.choice(["", "a", "Hello, World", "ÄÖÜ ß", "  spaced  ",
                       "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24)))])


def generate_inputs(draft: ToolDraft, n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    props: dict[str, Any] = draft.parameters.get("properties", {})
    required = set(draft.parameters.get("required", list(props)))
    seen = kwargs_from_tests(draft)
    pools = {k: [kw[k] for kw in seen if k in kw] for k in props}
    inputs: list[dict[str, Any]] = [dict(kw) for kw in seen]
    while len(inputs) < n + len(seen):
        kw = {}
        for name, spec in props.items():
            if name in required or rng.random() < 0.5:
                kw[name] = _value(spec or {}, name, pools.get(name, []), rng)
        inputs.append(kw)
    return inputs


# ------------------------------------------------------------------- compare

def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def compare(sandbox: Sandbox, candidate: ToolDraft, reference_code: str, inputs: list[dict[str, Any]],
            reference_func: str | None = None) -> DiffReport:
    a = sandbox.batch(candidate.code, candidate.name, inputs)
    b = sandbox.batch(reference_code, reference_func or candidate.name, inputs)
    if not a.ok:
        return DiffReport(error=f"candidate failed under fuzzing: {a.error}")
    if not b.ok:
        return DiffReport(error=f"reference failed to run: {b.error}")
    report = DiffReport(inputs=len(inputs))
    for kw, x, y in zip(inputs, a.outputs, b.outputs):
        if x["ok"] and y["ok"]:
            if _same(x["value"], y["value"]):
                report.agreements += 1
            else:
                report.hard.append({"input": kw, "candidate": x["value"], "reference": y["value"]})
        elif not x["ok"] and not y["ok"]:
            report.agreements += 1
        else:
            report.soft.append({"input": kw,
                                "candidate": x.get("value", f"raises {x.get('error')}"),
                                "reference": y.get("value", f"raises {y.get('error')}")})
    return report


def adjudicated_tests(verdicts: list[dict[str, Any]]) -> list[TestCase]:
    """Turn arbiter verdicts into regression tests."""
    tests = []
    for v in verdicts:
        if isinstance(v, dict) and isinstance(v.get("input"), dict) and "expected" in v:
            tests.append(TestCase(kwargs=v["input"], expected=v["expected"]))
    return tests
