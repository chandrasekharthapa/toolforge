"""The retrieval-augmented layer: tool library search, few-shot examples and lessons.

Three things are retrieved, all through the same hybrid (dense + BM25 + RRF) index:

* **tools**    – to decide reuse vs. create;
* **examples** – the closest verified tools, shown to the synthesizer as worked
                 examples of house style (signature, docstring, tests);
* **lessons**  – distilled "mistake → fix" notes written after every repair loop,
                 so the forge does not repeat errors it has already paid for.

``rag=False`` turns examples and lessons off (tool search stays on) for ablations.
"""

from __future__ import annotations

from .embeddings import Embedder
from .models import Lesson, Tool, ToolDraft, Verification
from .registry import Registry
from .retrieval import Doc, Hit, hybrid_search


class Knowledge:
    def __init__(self, registry: Registry, embedder: Embedder, *, rag: bool = True,
                 lexical_weight: float | None = None) -> None:
        self.registry = registry
        self.embedder = embedder
        self.rag = rag
        self.lexical_weight = embedder.lexical_weight if lexical_weight is None else lexical_weight
        self._reembed_if_embedder_changed()

    def _reembed_if_embedder_changed(self) -> None:
        """Vectors from different embedders are not comparable even at equal dimension."""
        stored = self.registry.get_meta("embedder")
        if stored == self.embedder.identity:
            return
        tools = [t for t, _ in self.registry.tool_embeddings()]
        if tools:
            for tool, vec in zip(tools, self.embedder.embed([t.search_text() for t in tools])):
                self.registry.set_embedding(tool.id, vec)
        lessons = self.registry.list_lessons()
        if lessons:
            texts = [f"{x.need} {x.mistake} {x.fix}" for x in lessons]
            for lesson, vec in zip(lessons, self.embedder.embed(texts)):
                self.registry.set_lesson_embedding(lesson.id, vec)
        self.registry.set_meta("embedder", self.embedder.identity)

    # ------------------------------------------------------------------ tools
    def _tool_docs(self) -> list[Doc]:
        docs = []
        for tool, vec in self.registry.tool_embeddings():
            if vec is None or vec.shape[0] != self._dim():
                vec = self.embedder.embed_one(tool.search_text())  # embedder changed: heal lazily
                self.registry.set_embedding(tool.id, vec)
            docs.append(Doc(tool.name, tool.search_text(), vec, tool))
        return docs

    def _dim(self) -> int:
        if not hasattr(self, "_cached_dim"):
            self._cached_dim = int(self.embedder.embed_one("probe").shape[0])
        return self._cached_dim

    def find_tools(self, query: str, k: int = 4) -> list[Hit]:
        docs = self._tool_docs()
        if not docs:
            return []
        return hybrid_search(query, self.embedder.embed_query(query), docs, k, lexical_weight=self.lexical_weight)

    def add_tool(self, draft: ToolDraft, *, origin_task: str, verification: Verification) -> Tool:
        vec = self.embedder.embed_one(draft.search_text())
        return self.registry.add_tool(draft, vec, origin_task=origin_task, verification=verification)

    def examples_for(self, query: str, k: int = 2) -> list[Tool]:
        if not self.rag:
            return []
        return [h.payload for h in self.find_tools(query, k)]

    # ---------------------------------------------------------------- lessons
    def add_lesson(self, need: str, mistake: str, fix: str) -> Lesson:
        lesson = Lesson(need=need, mistake=mistake, fix=fix)
        return self.registry.add_lesson(lesson, self.embedder.embed_one(f"{need} {mistake} {fix}"))

    def lessons_for(self, query: str, k: int = 3) -> list[Lesson]:
        if not self.rag:
            return []
        pairs = self.registry.lesson_embeddings()
        docs = [Doc(str(lesson.id), f"{lesson.need} {lesson.mistake} {lesson.fix}",
                    vec if vec is not None and vec.shape[0] == self._dim()
                    else self.embedder.embed_one(lesson.need), lesson)
                for lesson, vec in pairs]
        if not docs:
            return []
        hits = hybrid_search(query, self.embedder.embed_query(query), docs, k, lexical_weight=self.lexical_weight)
        floor = self.embedder.consider_threshold
        return [h.payload for h in hits if h.cosine >= floor or h.bm25 > 0]


