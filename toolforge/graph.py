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
* field_repair– a verified tool that raised on every real call this run is sent back to the
                maker with those inputs; the fix must pass the old tests plus new ones for the
                failing inputs, becomes the next version, and the task is executed again (once)
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
from .safety import check_code, strip_demo_code
from .sandbox import Sandbox


class ForgeState(TypedDict, total=False):
    task: str
    needs: list[dict[str, Any]]
    offered: list[str]
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
    field_failures: list[dict[str, Any]]
    known_tests: list[dict[str, Any]]
    field_repair: str | None
    repaired: list[str]
    answer: str
    trace: list[dict[str, Any]]


TASK_REF = "<<TASK>>"
RESULT_REF = "<<RESULT>>"

_PER_NEED_RESET = {"draft": None, "reference": None, "attempts": 0, "feedback": "",
                   "verified": False, "verification": {}, "pending_lessons": [], "adjudicated": [],
                   "chosen": None}


_WRAPPING = " \t\r\n\"'`<>"  # delimiters a model adds around copied text (<<< ... >>>, quotes)


def _is_task_copy(value: Any, task: str) -> bool:
    """True if ``value`` is the task text, retyped: most of the task, almost all of it copied in order.
    Small models drop an opening sentence or the options, wrap the text in ``<<< >>>`` or quotes, or
    change spacing; none of that makes it a different input."""
    if not isinstance(value, str) or len(value) < 40:
        return False
    a, b = " ".join(value.strip(_WRAPPING).split()), " ".join(task.split())
    if not a or not b or len(a) > 1.2 * len(b) or len(a) < 0.4 * len(b):
        return False
    if a in b:
        return True
    from difflib import SequenceMatcher

    m = SequenceMatcher(None, a, b, autojunk=False)
    copied = sum(block.size for block in m.get_matching_blocks())
    return copied >= 0.95 * len(a) or (len(a) >= 0.6 * len(b) and m.ratio() >= 0.9)


def _whole_task_params(tool: Any, task: str) -> set[str]:
    """Parameters this tool was verified on whole problem statements for (a test value at least half
    as long as the task). Only these may be snapped to the task: a tool that takes a word list or a
    bracket string must never be handed the whole problem."""
    return {k for t in tool.tests for k, v in t.kwargs.items()
            if isinstance(v, str) and len(v) >= 0.5 * len(task)}


def _clip(value: Any, limit: int = 1500) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "…(truncated)"


class Forge:
    def __init__(self, llm: LLM, knowledge: Knowledge, sandbox: Sandbox, settings: Settings,
                 user_llm: LLM | None = None) -> None:
        self.llm = llm  # tool MAKER: writes, verifies, judges and repairs tools
        self.user_llm = user_llm or llm  # tool USER: plans each query and executes it
        self.knowledge = knowledge
        self.registry = knowledge.registry
        self.sandbox = sandbox
        self.s = settings
        emb = knowledge.embedder
        self.reuse_t = settings.reuse_threshold if settings.reuse_threshold is not None else emb.reuse_threshold
        self.consider_t = (settings.consider_threshold if settings.consider_threshold is not None
                           else emb.consider_threshold)
        self.reranker = None
        if (settings.judge or "llm").lower() == "reranker":
            from .reranker import RerankerJudge

            self.reranker = RerankerJudge(settings.reranker_path)

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
        offered: list[str] = []
        prompt = f"Task: {state['task']}"
        if self.s.plan_with_library:
            hits = self.knowledge.find_tools(state["task"], 3)
            offered = [h.key for h in hits]
            if hits:
                prompt += P.library_for_planner([h.payload for h in hits])
        try:
            for attempt in range(2):
                data = self.user_llm.complete_json(system, prompt)
                for raw in (data.get("needs") or [])[: self.s.max_needs]:
                    try:
                        needs.append(Need(**raw).model_dump())
                    except (ValidationError, TypeError):
                        continue
                if len(needs) >= self.s.min_needs or attempt:
                    break
                prompt += (f"\n\nThis task explicitly requires building a reusable tool: return at least "
                           f"{self.s.min_needs} need(s).")
        except ValueError as e:
            error = str(e)
        return {"needs": needs, "offered": offered, "idx": 0, "bound": [], "created": [], "reused": [],
                "repaired": [], "field_failures": [], "field_repair": None,
                "failed_needs": [], "tool_calls": [], **_PER_NEED_RESET,
                "trace": self._log(state, "analyze", needs=[n["name_hint"] for n in needs], error=error)}

    def match(self, state: ForgeState) -> dict[str, Any]:
        need = self._need(state)
        hits = self.knowledge.find_tools(need.query(), self.s.top_k)
        seen = [{"tool": h.key, "cosine": round(h.cosine, 3), "bm25": round(h.bm25, 2),
                 "rrf": round(h.rrf, 4)} for h in hits]
        candidates = [h for h in hits if h.cosine >= self.consider_t or h.key == need.name_hint]
        decision, chosen, reason = "create", None, "no sufficiently similar tool"

        if need.name_hint in state.get("offered", []) and self.registry.get(need.name_hint):
            # the planner was shown this verified tool's card and asked for it by name
            decision, chosen, reason = "reuse", need.name_hint, "planner chose this library tool by name"
        elif candidates:
            top = max(candidates, key=lambda h: h.cosine)
            if top.cosine >= self.reuse_t:
                decision, chosen, reason = "reuse", top.key, f"cosine {top.cosine:.2f} ≥ {self.reuse_t}"
            elif self.reranker is not None:
                pick, reason = self.reranker.choose(need.query(), [h.payload for h in candidates])
                if pick:
                    decision, chosen = "reuse", pick
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
            code = strip_demo_code(str(data.get("code", "")))[0]
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

        # arbiter-decided tests are sticky: a repair cannot silently drop them. Known-answer cases the
        # caller supplied (solved examples) are added the same way, copied exactly by code, not by a model
        def key(t: Any) -> str:  # one spelling for a test case, whether it came as a dict or a TestCase
            return json.dumps((t if isinstance(t, TestCase) else TestCase(**t)).model_dump(),
                              sort_keys=True, default=str)

        known = {key(t) for t in draft.tests}
        params = set(draft.parameters.get("properties", {}))
        supplied = [t for t in state.get("known_tests", []) if set(t.get("kwargs", {})) and set(t["kwargs"]) <= params]
        for raw_test in [*state.get("adjudicated", []), *supplied]:
            if key(raw_test) not in known:
                draft.tests.append(TestCase(**raw_test))

        draft.code, stripped = strip_demo_code(draft.code)
        violations = check_code(draft.code, draft.name)
        if violations:
            return fail("static", "The static safety policy rejected the code:\n"
                        + "\n".join(f"- {v}" for v in violations))
        if len(draft.tests) < self.s.min_tests:
            return fail("tests", f"Provide at least {self.s.min_tests} test cases (got {len(draft.tests)}).")

        result = self.sandbox.run_tests(draft.code, draft.name, draft.tests)
        dropped = 0
        if not result.ok and supplied:
            # Known answers outrank the model's own guesses: if every failing test is one the model wrote
            # itself while every supplied known-answer case passes, the self-written tests are the ones
            # that are wrong (a typo, a mis-escaped "\\n") and are dropped instead of "fixing" good code.
            pinned = {key(t) for t in [*supplied, *state.get("adjudicated", [])]}
            failing = {r["i"] for r in result.results if not r.get("passed")}
            own = {i for i, t in enumerate(draft.tests) if key(t) not in pinned}
            if failing and failing <= own:
                kept = [t for i, t in enumerate(draft.tests) if i not in failing]
                if len(kept) >= self.s.min_tests:
                    dropped, draft.tests = len(failing), kept
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
                **updates, "trace": self._log(state, "verify", ok=True, dropped_own_tests=dropped,
                                              stripped_demo_lines=stripped, **verification.model_dump())}

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
        if state.get("field_repair"):
            return {"bound": bound, "repaired": [*state.get("repaired", []), tool.name], "idx": state["idx"] + 1,
                    **_PER_NEED_RESET,
                    "trace": self._log(state, "register", tool=tool.name, version=tool.version, how=how,
                                       field_repair=True, lessons=len(state.get("pending_lessons", [])))}
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
        failed: dict[str, str] = {}
        succeeded: dict[str, Any] = {}
        task_answer: str | None = None  # what a tool given the WHOLE task returned
        for step_no in range(self.s.max_exec_steps):
            last = step_no == self.s.max_exec_steps - 1
            if last and step_no:
                scratch.append({"note": "This is your last step: reply with the final answer now "
                                        "(\"<<RESULT>>\" copies the last tool result)."})
            try:
                step = self.user_llm.complete_json(P.EXECUTOR, P.executor_prompt(state["task"], tools, scratch))
            except ValueError:
                scratch.append({"error": "reply was not valid JSON; respond with one JSON object"})
                continue
            action = step.get("action")
            if action == "final":
                if tools and not calls and not nudged and not last:
                    # A verified tool was forged or reused for this task; an answer computed "in
                    # the model's head" bypasses it and would make the run's accuracy meaningless.
                    nudged = True
                    scratch.append({"error": "You answered without calling any tool. Call the provided "
                                             "tool(s) to compute the result, then give the final answer."})
                    continue
                answer = str(step.get("answer", "")).strip()
                if self.s.answer_from_task_tool and task_answer is not None:
                    answer = task_answer  # the tool solved the whole task; do not let the model retype it
                elif RESULT_REF in answer:  # the final answer IS a tool result: copy it exactly
                    last = next((c for c in reversed(calls) if "error" not in c), None)
                    if last is not None:
                        value = last["result"]
                        answer = answer.replace(RESULT_REF, value if isinstance(value, str)
                                                else json.dumps(value, default=str))
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
            signature = name + json.dumps(args, sort_keys=True, default=str)
            if signature in failed:  # small models retry the same broken call until the budget runs out
                scratch.append({"call": name, "args": args, "error": f"this exact call already failed "
                                f"({failed[signature][:200]}). Change the arguments to match the parameters "
                                "and example_call, or give your final answer."})
                continue
            if signature in succeeded:  # same call, same answer: the model is stuck, not computing
                scratch.append({"call": name, "args": args, "result": succeeded[signature],
                                "note": "You already have this result. Give your final answer now."})
                continue
            # "<<TASK>>" passes the task text verbatim: a long problem statement copied through a
            # small model's JSON gets mangled, a reference does not
            real_args = {k: (state["task"] if v == TASK_REF else v) for k, v in args.items()}
            whole_task = TASK_REF in args.values()
            if self.s.answer_from_task_tool and not whole_task:
                # a long argument that is clearly the task retyped (a line dropped, spacing changed) is
                # snapped back to the exact task text the tool was verified on
                whole_params = _whole_task_params(tool, state["task"])
                for k, v in real_args.items():
                    if k in whole_params and _is_task_copy(v, state["task"]):
                        real_args[k], whole_task = state["task"], True
            res = self.sandbox.call(tool.code, tool.name, real_args)
            self.registry.record_use(tool.id, res.ok)
            if not res.ok:
                failed[signature] = res.error or "failed"
            else:
                succeeded[signature] = _clip(res.result)
                if (whole_task and task_answer is None
                        and isinstance(res.result, (str, int, float)) and not isinstance(res.result, bool)):
                    task_answer = str(res.result)
            if res.ok and tool.example is None and (
                    whole_task or name in [*state.get("created", []), *state.get("repaired", [])]):
                # the run that built a tool leaves a worked call behind for later callers (LATM's
                # "wrapping": a usage demonstration travels with the tool). The task itself is shown as
                # "<<TASK>>". An example with a cut-off argument is not a call anyone can copy (small
                # models imitate its truncation and wrapping), so none is kept; the first call that hands
                # the tool the whole task records one later.
                shown = {k: TASK_REF if real_args.get(k) == state["task"] else v for k, v in args.items()}
                if all(len(json.dumps(v, default=str)) <= 300 for v in shown.values()):
                    tool.example = {"args": shown, "result": _clip(res.result, 300)}
                    self.registry.set_example(tool.id, tool.example)
            entry = ({"call": name, "args": args, "result": _clip(res.result)} if res.ok
                     else {"call": name, "args": args, "error": res.error})
            scratch.append(entry)
            calls.append({**entry, "version": tool.version, "ms": round(res.duration_s * 1000)})
        if self.s.answer_from_task_tool and task_answer is not None and answer.startswith("(no final answer"):
            answer = task_answer
        worked = {c["call"] for c in calls if "error" not in c}
        field = [{"tool": c["call"], "args": c["args"], "error": c["error"]} for c in calls
                 if "error" in c and c["call"] not in worked and "[sandbox]" not in str(c["error"])
                 and not str(c["error"]).startswith(("timed out", "process killed"))]
        previous = state.get("tool_calls", []) if state.get("field_repair") else []
        return {"answer": answer, "tool_calls": previous + calls,
                "field_failures": [] if state.get("field_repair") else field,
                "trace": self._log(state, "execute", calls=len(calls), answer=answer[:300],
                                   field_failures=len(field))}

    def field_repair(self, state: ForgeState) -> dict[str, Any]:
        failures = [f for f in state["field_failures"] if self._repairs_left(f["tool"])] or state["field_failures"]
        name = failures[0]["tool"]
        key = f"field_repairs:{name}"
        self.registry.set_meta(key, str(int(self.registry.get_meta(key) or 0) + 1))
        tool = self.registry.get(name)
        shown = [f for f in failures if f["tool"] == name][:3]
        feedback = ("This tool passed verification, but it failed in real use. Solving the task below, the "
                    "caller passed the inputs exactly as the task writes them, and every call raised:\n"
                    + "\n".join(f"- args={json.dumps(f['args'], default=str)[:400]} -> {f['error'][:200]}"
                                 for f in shown)
                    + f"\n\nThe task: {state['task'][:1500]!r}\n\nMake the function accept inputs written "
                      "this way (keep the same name and parameters), keep every existing test passing, and "
                      "add test cases for these inputs with their correct expected outputs.")
        need = {"name_hint": name, "description": tool.description}
        return {"needs": [*state["needs"], need], "idx": len(state["needs"]), **_PER_NEED_RESET,
                "draft": {k: v for k, v in tool.model_dump().items() if k in ToolDraft.model_fields},
                "feedback": feedback, "field_repair": name, "field_failures": [],
                "trace": self._log(state, "field_repair", tool=name, failed_calls=len(shown))}

    # ---------------------------------------------------------------- known-answer repair
    def repair_with_tests(self, name: str, tests: list[TestCase], reason: str, task: str = "") -> dict[str, Any]:
        """Repair a registered tool against cases whose answers are KNOWN (e.g. solved examples).

        The cases become sticky tests: the repair loop may not drop them, the full verification
        pipeline runs again, and the version gate still demands the old tests pass. Returns
        ``{"ok": bool, "tool": name, "version": int, "trace": [...]}``.
        """
        tool = self.registry.get(name)
        if tool is None:
            return {"ok": False, "tool": name, "trace": [{"node": "repair", "error": "no such tool"}]}
        draft = {k: v for k, v in tool.model_dump().items() if k in ToolDraft.model_fields}
        sticky = [t.model_dump() for t in tests]
        state: ForgeState = {
            "task": task or f"repair {name}", "needs": [{"name_hint": name, "description": tool.description}],
            "idx": 0, "bound": [], "created": [], "reused": [], "repaired": [], "failed_needs": [],
            "tool_calls": [], "trace": [], **_PER_NEED_RESET,
            "draft": draft, "feedback": reason, "adjudicated": sticky, "field_repair": name,
        }
        while True:
            state.update(self.synthesize(state))
            state.update(self.verify(state))
            if state.get("verified"):
                state.update(self.register(state))
                fixed = self.registry.get(state["repaired"][-1]) if state.get("repaired") else None
                return {"ok": True, "tool": fixed.name if fixed else name,
                        "version": fixed.version if fixed else tool.version, "trace": state["trace"]}
            if state.get("attempts", 0) > self.s.max_repairs:
                return {"ok": False, "tool": name, "version": tool.version, "trace": state["trace"]}

    # ---------------------------------------------------------------- routing
    def _after_verify(self, state: ForgeState) -> str:
        if state.get("verified"):
            return "register"
        return "synthesize" if state.get("attempts", 0) <= self.s.max_repairs else "give_up"

    @staticmethod
    def _next_need(state: ForgeState) -> str:
        return "match" if state["idx"] < len(state["needs"]) else "execute"

    def _after_execute(self, state: ForgeState) -> str:
        failures = [f for f in state.get("field_failures", []) if self._repairs_left(f["tool"])]
        return "field_repair" if self.s.field_repair and failures else END

    def _repairs_left(self, name: str) -> bool:
        """A tool may be sent back for field repair at most ``max_field_repairs`` times per library:
        a tool that keeps breaking needs a different design, not another round of big-model tokens."""
        return int(self.registry.get_meta(f"field_repairs:{name}") or 0) < self.s.max_field_repairs

    def _after_give_up(self, state: ForgeState) -> str:
        return END if state.get("field_repair") else self._next_need(state)  # keep the first answer

    def build(self):
        g = StateGraph(ForgeState)
        for name in ("analyze", "match", "reuse", "synthesize", "verify", "register", "give_up", "execute",
                     "field_repair"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("analyze")
        g.add_conditional_edges("analyze", self._next_need, {"match": "match", "execute": "execute"})
        g.add_conditional_edges("match", lambda s: s["decision"], {"reuse": "reuse", "create": "synthesize"})
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self._after_verify,
                                {"register": "register", "synthesize": "synthesize", "give_up": "give_up"})
        for node in ("reuse", "register"):
            g.add_conditional_edges(node, self._next_need, {"match": "match", "execute": "execute"})
        g.add_conditional_edges("give_up", self._after_give_up, {"match": "match", "execute": "execute", END: END})
        g.add_conditional_edges("execute", self._after_execute, {"field_repair": "field_repair", END: END})
        g.add_edge("field_repair", "synthesize")
        return g.compile()
