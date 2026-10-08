"""A fine-tuned cross-encoder that replaces the LLM reuse-judge (see evals/train_reranker.py).

It scores (need, tool card) pairs; the best candidate is reused if its probability clears the
threshold that was chosen on validation tools, otherwise a new tool is built. Same inputs as the
LLM judge (the need's query text and each candidate's search text), zero LLM tokens per decision.

    TOOLFORGE_JUDGE=reranker TOOLFORGE_RERANKER_PATH=evals/distill/reranker toolforge run "..."

Needs the optional extras: ``pip install -e ".[reranker]"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RerankerJudge:
    def __init__(self, path: str, threshold: float | None = None, device: str | None = None,
                 max_len: int = 192) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on optional extras
            raise RuntimeError('The reranker judge needs PyTorch and transformers: pip install -e ".[reranker]"') from e
        self._torch = torch
        meta_path = Path(path) / "toolforge_reranker.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.threshold = threshold if threshold is not None else float(meta.get("threshold", 0.5))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.name = f"reranker:{Path(path).name}"

    def scores(self, query: str, cards: list[str]) -> list[float]:
        if not cards:
            return []
        with self._torch.no_grad():
            enc = self.tokenizer([query] * len(cards), cards, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            return self._torch.sigmoid(logits).float().cpu().tolist()

    def choose(self, query: str, candidates: list[Any]) -> tuple[str | None, str]:
        """``candidates`` are registry Tools (anything with ``.name`` and ``.search_text()``)."""
        probs = self.scores(query, [c.search_text() for c in candidates])
        if not probs:
            return None, "reranker: no candidates"
        best = max(range(len(probs)), key=probs.__getitem__)
        if probs[best] >= self.threshold:
            return candidates[best].name, f"reranker p={probs[best]:.2f} ≥ {self.threshold:.2f}"
        return None, f"reranker best p={probs[best]:.2f} < {self.threshold:.2f}; building a new tool"
