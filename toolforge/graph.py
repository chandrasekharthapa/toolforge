"""The Toolforge agent as a LangGraph state machine.

    analyze ─► match ─┬─► reuse ───────────────────────────┐
                      └─► synthesize ─► verify ─┬─► register ┤
                             ▲                  ├─► give_up ─┤
                             └──── repair ◄─────┘            ▼
                                             (next need) match … ─► execute ─► END

* analyze     – planner decomposes the task into reusable capability needs
* match       – hybrid retrieval over the library; auto-reuse above a calibrated
                threshold, LLM judge in the grey zone, create below it
* synthesize  – RAG-augmented code generation (similar verified tools + lessons)
* verify      – static policy → sandboxed unit tests → differential fuzzing against
                an independent reference implementation, with an arbiter that turns
                disagreements into new regression tests
* register    – versioned insert into the library; lessons from repairs are committed
* execute     – JSON tool-calling loop that runs every call in the sandbox
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from . import prompts as P
from .config import Settings
from .differential import adjudicated_tests, compare, generate_inputs
from .knowledge import Knowledge
from .llm import LLM
from .models import Need, TestCase, ToolDraft, Verification
from .safety import check_code
from .sandbox import Sandbox


class ForgeState(TypedDict, total=False):
    task: str
    needs: list[dict[str, Any]]
    idx: int
    decision: str
    chosen: str | None
    draft: dict[str, Any] | None
    reference: dict[str, Any] | None
    attempts: int
    feedback: str
    verified: bool
    verification: dict[str, Any]
    pending_lessons: list[dict[str, str]]
    adjudicated: list[dict[str, Any]]
    bound: list[str]
    created: list[str]
    reused: list[str]
    failed_needs: list[str]
    tool_calls: list[dict[str, Any]]
    answer: str
    trace: list[dict[str, Any]]


_PER_NEED_RESET = {"draft": None, "reference": None, "attempts": 0, "feedback": "",
                   "verified": False, "verification": {}, "pending_lessons": [], "adjudicated": [],
                   "chosen": None}


def _clip(value: Any, limit: int = 1500) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "…(truncated)"


class Forge:
    def __init__(self, llm: LLM, knowledge: Knowledge, sandbox: Sandbox, settings: Settings) -> None:
        self.llm = llm
        self.knowledge = knowledge
        self.registry = knowledge.registry
        self.sandbox = sandbox
        self.s = settings
        emb = knowledge.embedder
        self.reuse_t = settings.reuse_threshold if settings.reuse_threshold is not None else emb.reuse_threshold
        self.consider_t = (settings.consider_threshold if settings.consider_threshold is not None
                           else emb.consider_threshold)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _log(state: ForgeState, node: str, **data: Any) -> list[dict[str, Any]]:
        return [*state.get("trace", []), {"node": node, **data}]

    def _need(self, state: ForgeState) -> Need:
        return Need(**state["needs"][state["idx"]])

    # ------------------------------------------------------------------ nodes
    def analyze(self, state: ForgeState) -> dict[str, Any]:
        system = P.PLANNER % {"max_needs": self.s.max_needs}
        needs: list[dict[str, Any]] = []
        error = None
        try:
            data = self.llm.complete_json(system, f"Task: {state['task']}")
            for raw in (data.get("needs") or [])[: self.s.max_needs]:
                try:
                    needs.append(Need(**raw).model_dump())
                except (ValidationError, TypeError):
                    continue
        except ValueError as e:
            error = str(e)
        return {"needs": needs, "idx": 0, "bound": [], "created": [], "reused": [],
                "failed_needs": [], "tool_calls": [], **_PER_NEED_RESET,
                "trace": self._log(state, "analyze", needs=[n["name_hint"] for n in needs], error=error)}

    def match(self, state: ForgeState) -> dict[str, Any]:
        need = self._need(state)
        hits = self.knowledge.find_tools(need.query(), self.s.top_k)
        seen = [{"tool": h.key, "cosine": round(h.cosine, 3), "bm25": round(h.bm25, 2),
                 "rrf": round(h.rrf, 4)} for h in hits]
        candidates = [h for h in hits if h.cosine >= self.consider_t or h.key == need.name_hint]
        decision, chosen, reason = "create", None, "no sufficiently similar tool"

        if candidates:
            top = max(candidates, key=lambda h: h.cosine)
            if top.cosine >= self.reuse_t:
                decision, chosen, reason = "reuse", top.key, f"cosine {top.cosine:.2f} ≥ {self.reuse_t}"
            else:
                prompt = (f"Task: {state['task']}\n"
                          f"Need: {need.name_hint} — {need.description}\n\n"
                          f"Candidate tools:\n{P.tool_cards([h.payload for h in candidates])}")
                try:
                    verdict = self.llm.complete_json(P.JUDGE, prompt)
                    pick = verdict.get("choice")
                    if pick in {h.key for h in candidates}:
                        decision, chosen = "reuse", pick
                    reason = f"judge: {verdict.get('reason', '')}"
                except ValueError:
                    reason = "judge reply unparseable; creating"

        return {"decision": decision, "chosen": chosen,
                "trace": self._log(state, "match", need=need.name_hint, decision=decision,
                                   chosen=chosen, reason=reason, candidates=seen)}

    def reuse(self, state: ForgeState) -> dict[str, Any]:
        name = state["chosen"]
        bound = state["bound"] + ([name] if name not in state["bound"] else [])
        return {"bound": bound, "reused": state["reused"] + [name], "idx": state["idx"] + 1,
                **_PER_NEED_RESET, "trace": self._log(state, "reuse", tool=name)}

    def synthesize(self, state: ForgeState) -> dict[str, Any]:
        need = self._need(state)
        lessons = P.render_lessons(self.knowledge.lessons_for(need.query()))
        fmt = {"allowed": P.ALLOWED, "min_tests": self.s.min_tests}
        repairing = bool(state.get("feedback")) and state.get("draft") is not None
        if repairing:
            system = P.REPAIRER % fmt
            prompt = P.repair_prompt(state["draft"], state["feedback"], lessons)
        else:
            system = P.SYNTHESIZER % fmt
            examples = P.render_examples(self.knowledge.examples_for(need.query()))
            prompt = P.synth_prompt(state["task"], need.model_dump(), examples, lessons)
            if state.get("feedback"):  # previous reply was not even valid JSON
                prompt += f"\n\nYour previous reply was rejected: {state['feedback']}"

        attempts = state.get("attempts", 0) + 1
        try:
            data = self.llm.complete_json(system, prompt)
        except ValueError as e:
            return {"draft": None, "attempts": attempts, "feedback": f"reply was not valid JSON ({e})",
                    "trace": self._log(state, "synthesize", attempt=attempts, error="invalid JSON")}

        pending = list(state.get("pending_lessons", []))
        lesson = data.pop("lesson", None)
        if repairing and isinstance(lesson, dict) and lesson.get("mistake") and lesson.get("fix"):
            pending.append({"mistake": str(lesson["mistake"]), "fix": str(lesson["fix"])})
        return {"draft": data, "attempts": attempts, "pending_lessons": pending,
                "trace": self._log(state, "synthesize", attempt=attempts, repair=repairing,
                                   name=data.get("name"), rag_lessons=lessons.count("\n- ") if lessons else 0)}

    def _reference(self, draft: ToolDraft) -> dict[str, Any]:
        """An independent implementation written from the spec alone."""
        try:
            data = self.llm.complete_json(
                P.REFERENCE % {"allowed": P.ALLOWED},
                P.reference_prompt(draft.name, draft.description, draft.parameters),
                temperature=0.7,
            )
            code = str(data.get("code", ""))
        except ValueError:
            return {"ok": False, "why": "unparseable reply"}
        violations = check_code(code, draft.name)
        if violations:
            return {"ok": False, "why": "; ".join(map(str, violations[:3]))}
        return {"ok": True, "code": code, "name": draft.name}

    def verify(self, state: ForgeState) -> dict[str, Any]:
        raw = state.get("draft")
        if raw is None:
            return {"verified": False, "trace": self._log(state, "verify", ok=False, stage="parse")}

        def fail(stage: str, feedback: str, **extra: Any) -> dict[str, Any]:
            return {"verified": False, "feedback": feedback, **extra,
                    "trace": self._log(state, "verify", ok=False, stage=stage, detail=feedback[:400])}

        try:
            draft = ToolDraft(**raw)
        except (ValidationError, TypeError) as e:
            return fail("schema", f"The JSON did not match the required shape: {e}")

        # arbiter-decided tests are sticky: a repair cannot silently drop them
        known = {json.dumps(t.model_dump(), sort_keys=True, default=str) for t in draft.tests}
        for raw_test in state.get("adjudicated", []):
            if json.dumps(raw_test, sort_keys=True, default=str) not in known:
                draft.tests.append(TestCase(**raw_test))

        violations = check_code(draft.code, draft.name)
        if violations:
            return fail("static", "The static safety policy rejected the code:\n"
                        + "\n".join(f"- {v}" for v in violations))
        if len(draft.tests) < self.s.min_tests:
            return fail("tests", f"Provide at least {self.s.min_tests} test cases (got {len(draft.tests)}).")

        result = self.sandbox.run_tests(draft.code, draft.name, draft.tests)
        if not result.ok:
            return fail("tests", "Sandboxed tests failed:\n" + result.failure_report())

        verification = Verification(tests_passed=len(draft.tests), tests_total=len(draft.tests),
                                    repair_rounds=state.get("attempts", 1) - 1)
        updates: dict[str, Any] = {}
        if self.s.differential:
            reference = state.get("reference") or self._reference(draft)
            updates["reference"] = reference
            if reference.get("ok"):
                inputs = generate_inputs(draft, self.s.fuzz_cases, seed=state.get("attempts", 0))
                report = compare(self.sandbox, draft, reference["code"], inputs,
                                 reference_func=reference.get("name"))
                if report.error and report.error.startswith("candidate"):
                    return fail("differential", f"{report.error}. Make the function robust: it must "
                                "return or raise ValueError quickly for any input.", **updates)
                if report.error:  # reference broken: cannot judge, do not block
                    verification.differential = "skipped"
                else:
                    verification.fuzz_cases = report.inputs
                    verification.disagreements = len(report.hard)
                    if report.disagreement_rate > self.s.max_disagreement:
                        try:
                            verdicts = self.llm.complete_json(P.ARBITER, P.arbiter_prompt(
                                draft.name, draft.description, draft.parameters, report.hard[:4]))
                            new_tests = adjudicated_tests(verdicts.get("verdicts", []))
                        except ValueError:
                            new_tests = []
                        known = {json.dumps(t.model_dump(), sort_keys=True, default=str) for t in draft.tests}
                        new_tests = [t for t in new_tests
                                     if json.dumps(t.model_dump(), sort_keys=True, default=str) not in known]
                        draft.tests.extend(new_tests)
                        updates["adjudicated"] = [*state.get("adjudicated", []),
                                                  *(t.model_dump() for t in new_tests)]
                        recheck = self.sandbox.run_tests(draft.code, draft.name, draft.tests)
                        if not recheck.ok:
                            return fail(
                                "differential",
                                "An independent implementation disagreed with yours on some inputs. An "
                                "arbiter determined the correct outputs from the specification; they were "
                                "added to your tests. Failing now:\n" + recheck.failure_report(),
                                draft=draft.model_dump(), verification=verification.model_dump(), **updates,
                            )
                        verification.tests_total = verification.tests_passed = len(draft.tests)
                    verification.arbitrated_tests = len(updates.get("adjudicated", state.get("adjudicated", [])))
                    verification.differential = "passed"
            else:
                verification.differential = "skipped"

        return {"verified": True, "draft": draft.model_dump(), "verification": verification.model_dump(),
                **updates, "trace": self._log(state, "verify", ok=True, **verification.model_dump())}

    def _resolve_name(self, draft: ToolDraft) -> tuple[ToolDraft, str]:
        """Decide whether a draft may become the next version of a same-named tool.

        A draft only supersedes the active tool if it has the same parameters AND still passes every
        test the active version carries: it must still do the old tool's job. Otherwise (a temperature
        converter that happens to be called convert_units, say) it is stored under a fresh name, so
        the tools other tasks already rely on are never silently replaced.
        """
        existing = self.registry.get(draft.name)
        if existing is None:
            return draft, "new"
        same_signature = set(existing.parameters.get("properties", {})) == set(draft.parameters.get("properties", {}))
        if same_signature and existing.tests:
            check = self.sandbox.run_tests(draft.code, draft.name, existing.tests)
            if check.ok:
                known = {json.dumps(t.model_dump(), sort_keys=True, default=str) for t in draft.tests}
                inherited = [t for t in existing.tests
                             if json.dumps(t.model_dump(), sort_keys=True, default=str) not in known]
                return draft.model_copy(update={"tests": [*draft.tests, *inherited]}), "new_version"
        elif same_signature:
            return draft, "new_version"
        n = 2
        while self.registry.get(f"{draft.name}_{n}"):
            n += 1
        new = f"{draft.name}_{n}"
        code = re.sub(rf"\b{re.escape(draft.name)}\b", new, draft.code)
        return draft.model_copy(update={"name": new, "code": code}), "renamed"

    def register(self, state: ForgeState) -> dict[str, Any]:
        draft, how = self._resolve_name(ToolDraft(**state["draft"]))
        verification = Verification(**state.get("verification", {}))
        if how == "new_version":
            verification.tests_total = verification.tests_passed = len(draft.tests)
        tool = self.knowledge.add_tool(draft, origin_task=state["task"], verification=verification)
        need = self._need(state)
        for lesson in state.get("pending_lessons", []):
            self.knowledge.add_lesson(need.description, lesson["mistake"], lesson["fix"])
        bound = state["bound"] + ([tool.name] if tool.name not in state["bound"] else [])
        return {"bound": bound, "created": state["created"] + [tool.name], "idx": state["idx"] + 1,
                **_PER_NEED_RESET,
                "trace": self._log(state, "register", tool=tool.name, version=tool.version, how=how,
                                   lessons=len(state.get("pending_lessons", [])))}

    def give_up(self, state: ForgeState) -> dict[str, Any]:
        need = self._need(state)
        return {"failed_needs": state["failed_needs"] + [need.name_hint], "idx": state["idx"] + 1,
                **_PER_NEED_RESET, "trace": self._log(state, "give_up", need=need.name_hint,
                                                       last_feedback=state.get("feedback", "")[:400])}

    def execute(self, state: ForgeState) -> dict[str, Any]:
        tools = [t for n in state.get("bound", []) if (t := self.registry.get(n))]
        by_name = {t.name: t for t in tools}
        scratch: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        answer = "(no final answer within the step budget)"
        nudged = False
        for _ in range(self.s.max_exec_steps):
            try:
                step = self.llm.complete_json(P.EXECUTOR, P.executor_prompt(state["task"], tools, scratch))
            except ValueError:
                scratch.append({"error": "reply was not valid JSON; respond with one JSON object"})
                continue
            action = step.get("action")
            if action == "final":
                if tools and not calls and not nudged:
                    # A verified tool was forged or reused for this task; an answer computed "in
                    # the model's head" bypasses it and would make the run's accuracy meaningless.
                    nudged = True
                    scratch.append({"error": "You answered without calling any tool. Call the provided "
                                             "tool(s) to compute the result, then give the final answer."})
                    continue
                answer = str(step.get("answer", "")).strip()
                break
            if action != "call":
                scratch.append({"error": f"unknown action {action!r}"})
                continue
            name, args = step.get("tool"), step.get("args") or {}
            tool = by_name.get(name)
            if tool is None:
                scratch.append({"call": name, "error": f"unknown tool; available: {sorted(by_name)}"})
                continue
            if not isinstance(args, dict):
                scratch.append({"call": name, "error": "args must be a JSON object"})
                continue
            res = self.sandbox.call(tool.code, tool.name, args)
            self.registry.record_use(tool.id, res.ok)
            entry = ({"call": name, "args": args, "result": _clip(res.result)} if res.ok
                     else {"call": name, "args": args, "error": res.error})
            scratch.append(entry)
            calls.append({**entry, "version": tool.version, "ms": round(res.duration_s * 1000)})
        return {"answer": answer, "tool_calls": calls,
                "trace": self._log(state, "execute", calls=len(calls), answer=answer[:300])}

    # ---------------------------------------------------------------- routing
    def _after_verify(self, state: ForgeState) -> str:
        if state.get("verified"):
            return "register"
        return "synthesize" if state.get("attempts", 0) <= self.s.max_repairs else "give_up"

    @staticmethod
    def _next_need(state: ForgeState) -> str:
        return "match" if state["idx"] < len(state["needs"]) else "execute"

    def build(self):
        g = StateGraph(ForgeState)
        for name in ("analyze", "match", "reuse", "synthesize", "verify", "register", "give_up", "execute"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("analyze")
        g.add_conditional_edges("analyze", self._next_need, {"match": "match", "execute": "execute"})
        g.add_conditional_edges("match", lambda s: s["decision"], {"reuse": "reuse", "create": "synthesize"})
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self._after_verify,
                                {"register": "register", "synthesize": "synthesize", "give_up": "give_up"})
        for node in ("reuse", "register", "give_up"):
            g.add_conditional_edges(node, self._next_need, {"match": "match", "execute": "execute"})
        g.add_edge("execute", END)
        return g.compile()
