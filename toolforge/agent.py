"""High-level entry point: ``Toolforge().run("task")``."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .config import Settings
from .embeddings import Embedder, make_embedder
from .graph import Forge, ForgeState
from .knowledge import Knowledge
from .llm import LLM, make_llm
from .models import RunResult
from .registry import Registry
from .sandbox import Sandbox

EventHandler = Callable[[dict[str, Any]], None]


class Toolforge:
    def __init__(self, settings: Settings | None = None, *, llm: LLM | None = None,
                 embedder: Embedder | None = None, registry: Registry | None = None,
                 sandbox: Sandbox | None = None, rag: bool = True) -> None:
        self.settings = settings or Settings()
        self.llm = llm or make_llm(self.settings)
        self.embedder = embedder or make_embedder(self.settings)
        self.registry = registry or Registry(self.settings.db_path)
        self.sandbox = sandbox or Sandbox(self.settings.sandbox_timeout, self.settings.sandbox_memory_mb)
        self.knowledge = Knowledge(self.registry, self.embedder, rag=rag,
                                   lexical_weight=self.settings.lexical_weight)
        self.forge = Forge(self.llm, self.knowledge, self.sandbox, self.settings)
        self.graph = self.forge.build()

    def close(self) -> None:
        """Release the database (Windows will not delete a SQLite file that is still open)."""
        self.registry.close()

    def __enter__(self) -> Toolforge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(self, task: str, on_event: EventHandler | None = None) -> RunResult:
        before = self.llm.usage.snapshot()
        start = time.perf_counter()
        final: ForgeState = {}
        emitted = 0
        for state in self.graph.stream({"task": task, "trace": []}, config={"recursion_limit": 250},
                                       stream_mode="values"):
            final = state
            trace = state.get("trace", [])
            if on_event:
                for event in trace[emitted:]:
                    on_event(event)
            emitted = len(trace)

        usage = self.llm.usage.since(before)
        result = RunResult(
            task=task,
            answer=final.get("answer", ""),
            created=final.get("created", []),
            reused=final.get("reused", []),
            failed_needs=final.get("failed_needs", []),
            tool_calls=final.get("tool_calls", []),
            llm_calls=usage.calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_s=round(time.perf_counter() - start, 3),
            trace=final.get("trace", []),
        )
        self.registry.log_run(result)
        return result
