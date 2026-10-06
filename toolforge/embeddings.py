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


class EmbeddingError(RuntimeError):
    """The embedding endpoint rejected the request (wrong model, retired model, bad key...)."""


class Embedder:
    name = "base"
    #: cosine at/above which a stored tool is reused without asking the LLM
    reuse_threshold = 0.9
    #: cosine below which a stored tool is not even shown to the LLM judge
    consider_threshold = 0.5
    #: weight of the BM25 ranking in Reciprocal Rank Fusion (dense ranking weight is 1.0). A strong
    #: neural embedder should outvote keywords; a weak one benefits from them. Measured per embedder
    #: with ``python -m evals.retrieval`` (its fusion sweep).
    lexical_weight = 1.0

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:  # pragma: no cover - abstract
        """``kind`` is "passage" for stored documents and "query" for searches. Asymmetric
        retrieval models embed the two differently; symmetric ones ignore it."""
        raise NotImplementedError

    def embed_one(self, text: str, kind: str = "passage") -> np.ndarray:
        return self.embed([text], kind)[0]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_one(text, "query")

    @property
    def identity(self) -> str:
        """Stored vectors are only comparable with vectors from the same embedder + model."""
        return f"{self.name}:{getattr(self, 'model', '') or getattr(self, 'dim', '')}"


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

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
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

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        r = self.client.models.embed_content(model=self.model, contents=texts,
                                             config=types.EmbedContentConfig(task_type=task))
        vecs = np.array([e.values for e in r.embeddings], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class OpenAICompatEmbedder(Embedder):
    name = "openai"
    reuse_threshold = 0.88
    consider_threshold = 0.5
    #: send input_type=query|passage (NVIDIA retrieval models require it)
    asymmetric = False
    key_env = "OPENAI_API_KEY"
    default_base = "https://api.openai.com/v1"
    default_model = "text-embedding-3-small"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        import httpx

        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or self.default_base).rstrip("/")
        self.model = model or self.default_model
        key = os.getenv(self.key_env) or "not-needed"
        self.http = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        out = []
        for start in range(0, len(texts), 64):  # stay well under per-request input limits
            body: dict = {"model": self.model, "input": texts[start:start + 64], "encoding_format": "float"}
            if self.asymmetric:
                body.update({"input_type": "query" if kind == "query" else "passage", "truncate": "END"})
            r = self.http.post(f"{self.base_url}/embeddings", json=body)
            if r.status_code >= 400:
                raise EmbeddingError(
                    f"{self.name} embeddings failed for model {self.model!r} (HTTP {r.status_code}): {r.text[:300]}\n"
                    f"  Run `python -m evals.retrieval --embedder {self.name} --probe` to find an embedding model "
                    "your key can use, then set TOOLFORGE_EMBED_MODEL.")
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d.get("index", 0))]
        vecs = np.array(out, dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


    def probe(self, keyword: str = "embed") -> list[tuple[str, bool, str]]:
        """Try every listed model whose id contains ``keyword`` with a tiny request."""
        r = self.http.get(f"{self.base_url}/models")
        r.raise_for_status()
        ids = sorted({m["id"] for m in r.json().get("data", []) if keyword in m.get("id", "").lower()})
        results, original = [], self.model
        for model_id in ids:
            self.model = model_id
            try:
                dim = self.embed(["probe"], "query").shape[1]
                results.append((model_id, True, f"dim {dim}"))
            except Exception as e:  # noqa: BLE001 - report every failure mode
                results.append((model_id, False, str(e).splitlines()[0][:120]))
        self.model = original
        return results


class NvidiaEmbedder(OpenAICompatEmbedder):
    """NVIDIA NIM retrieval embeddings (free tier at build.nvidia.com). Query and passage are
    embedded differently, so query-to-tool cosines run lower than symmetric models'. With these
    thresholds every one of 50 labelled needs had its correct tool shown to the LLM judge and no
    need without a tool was auto-reused (``python -m evals.retrieval --embedder nvidia``)."""

    name = "nvidia"
    reuse_threshold = 0.80
    consider_threshold = 0.25
    # Measured (evals/retrieval.py, nemotron-3-embed-1b): dense alone 98% hit@1, equal-weight hybrid 92%,
    # hybrid with BM25 at 0.5 back to 98%. A strong embedder should outvote keyword lookalikes such as
    # roman_to_int vs int_to_roman; BM25 still breaks ties. (Weight chosen on the same 50 queries.)
    lexical_weight = 0.5
    asymmetric = True
    key_env = "NVIDIA_API_KEY"
    default_base = "https://integrate.api.nvidia.com/v1"
    default_model = "nvidia/nemotron-3-embed-1b"


def make_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder.lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "gemini":
        return GeminiEmbedder(settings.embed_model)
    if kind == "nvidia":
        return NvidiaEmbedder(settings.embed_model)
    if kind in {"openai", "ollama", "openai-compatible"}:
        base = settings.base_url or ("http://localhost:11434/v1" if kind == "ollama" else None)
        return OpenAICompatEmbedder(settings.embed_model, base)
    raise ValueError(f"Unknown TOOLFORGE_EMBEDDER: {settings.embedder!r}")
