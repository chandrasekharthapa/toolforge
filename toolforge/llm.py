"""Provider-agnostic LLM layer.

Every model sits behind one tiny interface, ``complete(system, prompt)``. The agent
never relies on provider-specific function calling; it speaks a JSON protocol, so any
chat model works (Gemini, Claude, or anything OpenAI-compatible such as Groq, Ollama,
OpenRouter, vLLM). Token usage is tracked uniformly so the eval harness can compare
cost across modes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .config import Settings


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def snapshot(self) -> Usage:
        return Usage(self.calls, self.input_tokens, self.output_tokens)

    def since(self, earlier: Usage) -> Usage:
        return Usage(
            self.calls - earlier.calls,
            self.input_tokens - earlier.input_tokens,
            self.output_tokens - earlier.output_tokens,
        )


class LLMConfigError(RuntimeError):
    """A setup problem (bad key, unknown model) that no retry will fix."""


class LLMAuthError(LLMConfigError):
    """The provider rejected the API key (HTTP 401/403)."""


class LLMModelError(LLMConfigError):
    """The provider does not serve the configured model (HTTP 404, or 400 naming the model)."""


class LLMQuotaError(LLMConfigError):
    """The provider's rate limit needs a long wait (typically a daily token quota)."""


#: rate-limit waits longer than this are reported instead of slept through
MAX_RATE_LIMIT_WAIT_S = 90.0
_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?")


def parse_wait(text: str) -> float | None:
    """Seconds from messages like 'Please try again in 7m12.5s' / '1h2m' / '850ms'."""
    m = re.search(r"try again in\s*([\dhms.]+)", text or "", re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).rstrip(".")
    if token.endswith("ms") and token[:-2].replace(".", "", 1).isdigit():
        return float(token[:-2]) / 1000
    d = _DURATION.fullmatch(token)
    if not d or not any(d.groups()):
        return None
    h, mins, secs = d.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(secs or 0)


def _notice(message: str) -> None:
    print(f"  … {message}", file=sys.stderr, flush=True)


class LLMBadRequest(RuntimeError):
    """HTTP 400 that is not a configuration problem, e.g. the provider's own JSON or
    tool-call validation rejecting what the model generated."""

    def __init__(self, message: str, code: str = "", failed_generation: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.failed_generation = failed_generation


class LLMGenerationError(ValueError):
    """The model could not produce a usable reply. Subclasses ValueError so the agent's
    nodes treat it exactly like an unparseable reply (repair / skip) instead of crashing."""


def _is_model_error(r) -> bool:
    if r.status_code in (404, 410):  # 410 Gone: the provider retired the model (NVIDIA NIM)
        return True
    if r.status_code != 400:
        return False
    try:
        err = r.json().get("error", {})
    except ValueError:
        return False
    if not isinstance(err, dict):
        return False
    if err.get("code") in {"model_not_found", "model_decommissioned"}:
        return True
    msg = str(err.get("message", "")).lower()
    return "model" in msg and any(w in msg for w in ("does not exist", "not found", "decommissioned",
                                                       "not supported", "no longer"))


def salvage_reply(text: str) -> dict[str, Any] | None:
    """Recover our JSON protocol from a generation the provider rejected.

    Models trained for native function calling (e.g. gpt-oss) sometimes emit a native
    tool call — ``{"name": ..., "arguments": ...}`` — instead of the requested JSON text.
    That is converted to ``{"action": "call", "tool": ..., "args": ...}``.
    """
    if not text:
        return None
    try:
        data = parse_json(text)
    except ValueError:
        return None
    if "action" not in data and isinstance(data.get("name"), str) and (
            "arguments" in data or "parameters" in data):
        args = data.get("arguments", data.get("parameters")) or {}
        if isinstance(args, str):
            try:
                args = parse_json(args)
            except ValueError:
                args = {}
        name = data["name"].split(".")[-1]  # "functions.foo" -> "foo"
        return {"action": "call", "tool": name, "args": args}
    return data


#: which environment variable holds the key for each OpenAI-compatible provider
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "ollama": None,  # local server, no key
}


def _error_detail(r) -> str:
    try:
        err = r.json().get("error", {})
        return str(err.get("message", err) if isinstance(err, dict) else err)
    except ValueError:
        return r.text[:200]


def _mask(key: str) -> str:
    return f"{key[:8]}…{key[-4:]} ({len(key)} chars)" if len(key) > 14 else "(short or empty)"


class LLM:
    """Base class: subclasses implement ``_complete``."""

    name = "base"
    #: the setting that chose this model, named in error messages
    model_env = "TOOLFORGE_MODEL"

    def __init__(self) -> None:
        self.usage = Usage()
        self._lock = threading.Lock()

    def complete(self, system: str, prompt: str, *, json_mode: bool = False,
                 temperature: float = 0.2) -> str:
        resp = self._complete(system, prompt, json_mode=json_mode, temperature=temperature)
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += resp.input_tokens
            self.usage.output_tokens += resp.output_tokens
        return resp.text

    def complete_json(self, system: str, prompt: str, *, temperature: float = 0.2) -> dict[str, Any]:
        text = self.complete(system, prompt, json_mode=True, temperature=temperature)
        return parse_json(text)

    def _complete(self, system: str, prompt: str, *, json_mode: bool,
                  temperature: float) -> LLMResponse:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- providers


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__()
        from google import genai  # pip install google-genai

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = model or "gemini-2.5-flash"

    def _complete(self, system, prompt, *, json_mode, temperature):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        r = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        meta = getattr(r, "usage_metadata", None)
        return LLMResponse(
            r.text or "",
            getattr(meta, "prompt_token_count", 0) or 0,
            getattr(meta, "candidates_token_count", 0) or 0,
        )


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__()
        import anthropic  # pip install anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or "claude-sonnet-5-5"

    def _complete(self, system, prompt, *, json_mode, temperature):
        if json_mode:
            prompt += "\n\nRespond with a single JSON object and nothing else."
        r = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in r.content)
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens)


class OpenAICompatLLM(LLM):
    """Any /chat/completions endpoint: Groq, Ollama, OpenRouter, vLLM, OpenAI…"""

    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, key_env: str | None = "OPENAI_API_KEY",
                 provider: str = "openai") -> None:
        super().__init__()
        import httpx

        self.provider = provider
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4o-mini"
        # Only the provider's own variable is consulted: a global OPENAI_API_KEY must never
        # be sent to Groq/OpenRouter (it would be rejected, and it leaks a secret to a third party).
        if api_key:
            key, self.key_source = api_key, "api_key argument"
        elif key_env and os.getenv(key_env):
            from .config import env_source

            origin = {"default": "environment"}.get(env_source(key_env), env_source(key_env))
            key, self.key_source = os.environ[key_env], f"{key_env} in your {origin}"
        else:
            key, self.key_source = "not-needed", "none"
        self._key_hint = _mask(key) if self.key_source != "none" else "(no key)"
        # reasoning models can think for minutes on one request; the read timeout must allow it
        read_s = float(os.getenv("TOOLFORGE_HTTP_TIMEOUT") or 300)
        self.http = httpx.Client(timeout=httpx.Timeout(read_s, connect=20.0),
                                 headers={"Authorization": f"Bearer {key}"})

    def _post_with_retry(self, url: str, body: dict[str, Any], attempts: int = 6):
        """POST with backoff on rate limits (429) and transient server errors (5xx).

        Free tiers (Groq, OpenRouter) rate-limit aggressively; honour Retry-After when sent.
        """
        import time

        import httpx

        delay = 2.0
        for attempt in range(attempts):
            try:
                r = self.http.post(url, json=body)
            except httpx.ReadTimeout:
                # the model is too slow on THIS request: one retry, not six (each waits the full timeout)
                if attempt >= 1:
                    raise
                _notice(f"{self.provider} did not answer within the read timeout; retrying once")
                continue
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                detail = _error_detail(r) if r.status_code == 429 else f"HTTP {r.status_code}"
                retry_after = r.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else None
                except ValueError:
                    wait = None
                wait = wait if wait is not None else (parse_wait(detail) if r.status_code == 429 else None)
                wait = wait if wait is not None else delay
                if r.status_code == 429 and wait > MAX_RATE_LIMIT_WAIT_S:
                    mins = wait / 60
                    raise LLMQuotaError(
                        f"{self.provider} rate limit needs a {mins:.0f}-minute wait for {self.model!r}: {detail}\n"
                        "  This is usually the free tier's daily token quota. Options: wait and re-run "
                        f"(the benchmark resumes where it stopped), or set {self.model_env} to another model "
                        "(each model has its own quota; run `toolforge models`)."
                    )
                if attempt == attempts - 1:
                    if r.status_code == 429:
                        raise LLMQuotaError(f"{self.provider} kept rate-limiting after {attempts} attempts: {detail}")
                    r.raise_for_status()
                _notice(f"{'rate-limited' if r.status_code == 429 else 'server error'} by {self.provider}; "
                        f"waiting {wait:.0f}s (attempt {attempt + 1}/{attempts - 1})")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            if r.status_code in (401, 403):
                detail = _error_detail(r)
                raise LLMAuthError(
                    f"{self.provider} rejected the API key (HTTP {r.status_code}): {detail}\n"
                    f"  key sent: {self._key_hint}, read from {self.key_source}\n"
                    "  Check that the key is current (a revoked or rotated key fails like this)."
                )
            if _is_model_error(r):
                detail = _error_detail(r)
                try:
                    models = self.list_models()
                    available = ", ".join(models[:30]) or "(none returned)"
                    if len(models) > 30:
                        available += f", … ({len(models)} in total)"
                except Exception:  # noqa: BLE001 - best effort, the original error matters more
                    available = "(could not fetch the model list)"
                raise LLMModelError(
                    f"{self.provider} does not serve model {self.model!r}"
                    f"{' any more (retired)' if r.status_code == 410 else ''} (HTTP {r.status_code}): {detail}\n"
                    f"  models available to your key: {available}\n"
                    "  A model can be listed and still not be served: `toolforge models <word> --probe` "
                    "tests each one.\n"
                    f"  Set {self.model_env} in .env to one that works."
                )
            if r.status_code == 400:
                try:
                    err = r.json().get("error", {})
                except ValueError:
                    err = {}
                err = err if isinstance(err, dict) else {"message": str(err)}
                raise LLMBadRequest(f"{self.provider} rejected the request (HTTP 400): "
                                    f"{err.get('message', r.text[:300])}",
                                    code=str(err.get("code", "")),
                                    failed_generation=str(err.get("failed_generation", "") or ""))
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")  # pragma: no cover

    def probe(self, model: str) -> str:
        """'ok', or a short reason a model id cannot be used for chat right now."""
        try:
            r = self.http.post(f"{self.base_url}/chat/completions", timeout=60, json={
                "model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK."}]})
        except Exception as e:  # noqa: BLE001
            return f"unreachable ({type(e).__name__})"
        if r.status_code == 200:
            return "ok"
        return f"HTTP {r.status_code}: {_error_detail(r)[:80]}"

    def list_models(self) -> list[str]:
        """Model ids the endpoint serves to this key (GET /models)."""
        r = self.http.get(f"{self.base_url}/models")
        if r.status_code in (401, 403):
            raise LLMAuthError(f"{self.provider} rejected the API key: {_error_detail(r)}\n"
                               f"  key sent: {self._key_hint}, read from {self.key_source}")
        r.raise_for_status()
        data = r.json().get("data", [])
        return sorted(m["id"] for m in data if isinstance(m, dict) and m.get("id")
                      and m.get("active", True) is not False)

    def _complete(self, system, prompt, *, json_mode, temperature):
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = f"{self.base_url}/chat/completions"
        try:
            r = self._post_with_retry(url, body)
        except LLMBadRequest as first:
            # 1) the rejected generation often already contains a usable answer
            salvaged = salvage_reply(first.failed_generation)
            if salvaged is not None:
                return LLMResponse(json.dumps(salvaged), 0, len(first.failed_generation) // 4)
            # 2) retry once without the provider's JSON enforcement; parse_json copes with prose
            retry = dict(body)
            retry.pop("response_format", None)
            retry["messages"] = [body["messages"][0], {"role": "user", "content": prompt + (
                "\n\nReply with the JSON object as plain text. Do not use native function or tool calls.")}]
            try:
                r = self._post_with_retry(url, retry)
            except LLMBadRequest as second:
                salvaged = salvage_reply(second.failed_generation)
                if salvaged is not None:
                    return LLMResponse(json.dumps(salvaged), 0, len(second.failed_generation) // 4)
                raise LLMGenerationError(f"{first} | retry without JSON mode: {second}") from second
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            data["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


class ScriptedLLM(LLM):
    """Deterministic stand-in used by the test-suite and offline demos.

    ``handler(system, prompt) -> str`` decides the reply. Token counts are
    estimated (≈4 chars/token) so cost metrics still move.
    """

    name = "scripted"

    def __init__(self, handler: Callable[[str, str], str | dict]) -> None:
        super().__init__()
        self.handler = handler
        self.transcript: list[tuple[str, str, str]] = []

    def _complete(self, system, prompt, *, json_mode, temperature):
        out = self.handler(system, prompt)
        text = out if isinstance(out, str) else json.dumps(out)
        self.transcript.append((system, prompt, text))
        return LLMResponse(text, len(system + prompt) // 4, len(text) // 4)


def make_llm(settings: Settings) -> LLM:
    provider = settings.provider.lower()
    if provider == "gemini":
        return GeminiLLM(settings.model)
    if provider in {"anthropic", "claude"}:
        return AnthropicLLM(settings.model)
    if provider in {"openai", "groq", "nvidia", "cerebras", "ollama", "openrouter", "openai-compatible"}:
        defaults = {
            "groq": "https://api.groq.com/openai/v1",
            "nvidia": "https://integrate.api.nvidia.com/v1",
            "cerebras": "https://api.cerebras.ai/v1",
            "ollama": "http://localhost:11434/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        return OpenAICompatLLM(settings.model, settings.base_url or defaults.get(provider),
                               key_env=PROVIDER_KEY_ENV.get(provider, "OPENAI_API_KEY"), provider=provider)
    raise ValueError(f"Unknown TOOLFORGE_PROVIDER: {settings.provider!r}")


def make_user_llm(settings: Settings) -> LLM | None:
    """The tool-USER model (planning + execution) when ``TOOLFORGE_USER_MODEL`` is set, else None."""
    if not settings.user_model:
        return None
    provider = settings.user_provider or settings.provider
    same = provider.lower() == settings.provider.lower()
    llm = make_llm(replace(settings, provider=provider, model=settings.user_model,
                           base_url=settings.base_url if same else None))
    llm.model_env = "TOOLFORGE_USER_MODEL"
    return llm


# --------------------------------------------------------------------------- JSON parsing

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json(text: str) -> dict[str, Any]:
    """Robustly pull one JSON object out of a model reply.

    Handles bare JSON, ```json fences, prose around the object, and reasoning models that
    emit ``<think>…</think>`` first (whose braces would otherwise confuse a naive scan).
    When several objects appear, the LAST complete one wins: models reason, then answer.
    """
    text = _THINK.sub("", text or "").strip()
    if text.lower().startswith("<think>"):  # unterminated think block: keep what follows it
        text = text.split("</think>")[-1]
    candidates = [text]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    for c in candidates:
        try:
            value = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i != -1:
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found = value
        i = text.find("{", end)  # skip past this object so a nested one never replaces it
    if found is not None:
        return found
    raise ValueError(f"Model did not return a JSON object: {text[:300]!r}")
