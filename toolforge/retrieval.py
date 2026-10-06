"""Hybrid retrieval: dense cosine + Okapi BM25, merged with Reciprocal Rank Fusion.

Dense vectors catch paraphrases ("days between two dates" ~ "date difference");
BM25 catches exact identifiers that embeddings blur ("sha256", "levenshtein").
RRF merges the two rankings without having to calibrate their raw scores against
each other (Cormack et al., 2009). The dense cosine is still reported per hit
because the reuse/create gate needs a calibrated similarity.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embeddings import tokenize


@dataclass
class Doc:
    key: str
    text: str
    vector: np.ndarray
    payload: Any = None


@dataclass
class Hit:
    key: str
    payload: Any
    cosine: float
    bm25: float
    rrf: float
    dense_rank: int
    lexical_rank: int | None


class BM25:
    def __init__(self, corpus: Sequence[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(toks) for toks in corpus]
        self.lengths = np.array([len(t) for t in corpus], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(corpus) else 0.0
        df: Counter[str] = Counter()
        for toks in corpus:
            df.update(set(toks))
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not self.docs:
            return out
        for i, tf in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
            s = 0.0
            for term in set(query):
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


def hybrid_search(query: str, query_vec: np.ndarray, docs: Sequence[Doc], k: int = 5,
                  rrf_k: int = 60, dense_weight: float = 1.0, lexical_weight: float = 1.0) -> list[Hit]:
    if not docs:
        return []
    matrix = np.stack([d.vector for d in docs])
    cosine = matrix @ query_vec
    lexical = BM25([tokenize(d.text) for d in docs]).scores(tokenize(query))

    dense_order = np.argsort(-cosine)
    dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}
    lex_order = [int(i) for i in np.argsort(-lexical) if lexical[i] > 0]
    lex_rank = {i: r + 1 for r, i in enumerate(lex_order)}

    hits = []
    for i, doc in enumerate(docs):
        score = dense_weight / (rrf_k + dense_rank[i])
        if i in lex_rank:
            score += lexical_weight / (rrf_k + lex_rank[i])
        hits.append(Hit(doc.key, doc.payload, float(cosine[i]), float(lexical[i]), score,
                        dense_rank[i], lex_rank.get(i)))
    hits.sort(key=lambda h: h.rrf, reverse=True)
    return hits[:k]
