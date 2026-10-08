"""High-level entry point: ``Toolforge().run("task")``."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .config import Settings
from .embeddings import Embedder, make_embedder
from .graph import Forge, ForgeState
from .knowledge import Knowledge
from .llm import LLM, make_llm, make_user_llm
from .models import RunResult
from .registry import Registry
from .sandbox import Sandbox, make_sandbox

EventHandler = Callable[[dict[str, Any]], None]


class Toolforge:
    def __init__(self, settings: Settings | None = None, *, llm: LLM | None = None,
                 embedder: Embedder | None = None, registry: Registry | None = None,
                 sandbox: Sandbox | None = None, rag: bool = True, user_llm: LLM | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = llm or make_llm(self.settings)
        #: tool-USER model for planning and execution; the maker (self.llm) writes and checks tools
        # an injected maker LLM means the caller controls the models; only build one from settings otherwise
        self.user_llm = user_llm or (make_user_llm(self.settings) if llm is None else None)
        self.embedder = embedder or make_embedder(self.settings)
        self.registry = registry or Registry(self.settings.db_path)
        self.sandbox = sandbox or make_sandbox(self.settings)
        self.knowledge = Knowledge(self.registry, self.embedder, rag=rag,
                                   lexical_weight=self.settings.lexical_weight)
        self.forge = Forge(self.llm, self.knowledge, self.sandbox, self.settings, user_llm=self.user_llm)
        self.graph = self.forge.build()

    def close(self) -> None:
        """Release the database (Windows will not delete a SQLite file that is still open)."""
        self.registry.close()

    def __enter__(self) -> Toolforge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def repair_tool(self, name: str, tests: list, reason: str, task: str = "") -> dict[str, Any]:
        """Repair a library tool against known-answer cases (see ``Forge.repair_with_tests``).
        Adds the maker-model token cost of the repair as ``tokens``."""
        before = self.llm.usage.snapshot()
        out = self.forge.repair_with_tests(name, tests, reason, task)
        out["tokens"] = self.llm.usage.since(before).total_tokens
        return out

    def run(self, task: str, on_event: EventHandler | None = None, known_tests: list | None = None) -> RunResult:
        """Solve ``task``. ``known_tests`` are solved cases (TestCase) that any tool built for it must pass;
        they are added to its tests by code, exactly, so a model never has to copy them."""
        before = self.llm.usage.snapshot()
        user_before = self.user_llm.usage.snapshot() if self.user_llm is not None else None
        start = time.perf_counter()
        final: ForgeState = {}
        emitted = 0
        start_state = {"task": task, "trace": [],
                       "known_tests": [t.model_dump() if hasattr(t, "model_dump") else t for t in known_tests or []]}
        for state in self.graph.stream(start_state, config={"recursion_limit": 250},
                                       stream_mode="values"):
            final = state
            trace = state.get("trace", [])
            if on_event:
                for event in trace[emitted:]:
                    on_event(event)
            emitted = len(trace)

        usage = self.llm.usage.since(before)
        user = self.user_llm.usage.since(user_before) if self.user_llm is not None else None
        calls = usage.calls + (user.calls if user else 0)
        tokens_in = usage.input_tokens + (user.input_tokens if user else 0)
        tokens_out = usage.output_tokens + (user.output_tokens if user else 0)
        result = RunResult(
            task=task,
            answer=final.get("answer", ""),
            created=final.get("created", []),
            reused=final.get("reused", []),
            failed_needs=final.get("failed_needs", []),
            repaired=final.get("repaired", []),
            tool_calls=final.get("tool_calls", []),
            llm_calls=calls,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            user_tokens=user.total_tokens if user else 0,
            latency_s=round(time.perf_counter() - start, 3),
            trace=final.get("trace", []),
        )
        self.registry.log_run(result)
        return result
