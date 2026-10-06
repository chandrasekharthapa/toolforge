"""Embedding back-ends.

``HashingEmbedder`` needs no API key and no model download: it hashes words, word
bigrams and character trigrams into a fixed-size signed vector. It is weaker than a
neural embedder but deterministic, which makes it ideal for tests and offline evals.
Each embedder ships calibrated similarity thresholds because cosine scores are not
comparable across embedding models.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from .config import Settings

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; snake_case and camelCase are split."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    return _WORD.findall(text.lower().replace("_", " "))


class Embedder:
    name = "base"
    #: cosine at/above which a stored tool is reused without asking the LLM
    reuse_threshold = 0.9
    #: cosine below which a stored tool is not even shown to the LLM judge
    consider_threshold = 0.5

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    name = "hashing"
    reuse_threshold = 0.86
    consider_threshold = 0.22

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokenize(text)
        feats = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += weight if (h >> 63) & 1 else -weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class GeminiEmbedder(Embedder):
    name = "gemini"
    reuse_threshold = 0.92
    consider_threshold = 0.62

    def __init__(self, model: str | None = None) -> None:
        from google import genai

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.model = model or "gemini-embedding-001"

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.models.embed_content(model=self.model, contents=texts)
        vecs = np.array([e.values for e in r.embeddings], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class OpenAICompatEmbedder(Embedder):
    name = "openai"
    reuse_threshold = 0.88
    consider_threshold = 0.5

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        import httpx

        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "text-embedding-3-small"
        key = os.getenv("OPENAI_API_KEY") or "not-needed"
        self.http = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.http.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        vecs = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def make_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder.lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "gemini":
        return GeminiEmbedder(settings.embed_model)
    if kind in {"openai", "ollama", "openai-compatible"}:
        base = settings.base_url or ("http://localhost:11434/v1" if kind == "ollama" else None)
        return OpenAICompatEmbedder(settings.embed_model, base)
    raise ValueError(f"Unknown TOOLFORGE_EMBEDDER: {settings.embedder!r}")
