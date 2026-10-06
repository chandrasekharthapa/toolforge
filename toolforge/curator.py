"""Library hygiene: find near-duplicate tools and under-performing tools.

A self-growing library degrades unless something prunes it. The curator reports
(and with ``apply=True`` retires) tools whose real-world success rate is poor, and
flags pairs of tools so similar that one should probably be merged into the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .knowledge import Knowledge


@dataclass
class CurationReport:
    duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    underperforming: list[tuple[str, int, float]] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)


def curate(knowledge: Knowledge, *, dup_threshold: float | None = None, min_uses: int = 5,
           min_success_rate: float = 0.6, apply: bool = False) -> CurationReport:
    registry = knowledge.registry
    threshold = dup_threshold if dup_threshold is not None else knowledge.embedder.reuse_threshold
    report = CurationReport()

    tools = registry.list_tools()
    if len(tools) > 1:
        vecs = knowledge.embedder.embed([t.search_text() for t in tools])
        sims = vecs @ vecs.T
        for i, j in zip(*np.triu_indices(len(tools), k=1)):
            if sims[i, j] >= threshold:
                report.duplicates.append((tools[i].name, tools[j].name, round(float(sims[i, j]), 3)))

    for t in tools:
        rate = t.success_rate
        if t.uses >= min_uses and rate is not None and rate < min_success_rate:
            report.underperforming.append((t.name, t.uses, round(rate, 3)))
            if apply:
                registry.set_status(t.name, "retired")
                report.retired.append(t.name)
    return report
