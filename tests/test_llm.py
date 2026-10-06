import httpx

from toolforge.llm import OpenAICompatLLM, parse_json


def _llm(handler) -> OpenAICompatLLM:
    llm = OpenAICompatLLM(base_url="https://api.groq.com/openai/v1", model="m", api_key="k")
    llm.http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def test_retries_rate_limits_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = []

    def handler(req):
        seen.append(req)
        if len(seen) < 3:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

    llm = _llm(handler)
    assert llm.complete_json("system", "Return JSON") == {"ok": True}
    assert len(seen) == 3 and llm.usage.calls == 1 and llm.usage.total_tokens == 13


def test_auth_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(401, json={"error": "bad key"})

    from toolforge.llm import LLMAuthError

    try:
        _llm(handler).complete("s", "p")
    except LLMAuthError:
        pass
    assert len(seen) == 1


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('noise {"b": [1, 2]} trailing') == {"b": [1, 2]}


def test_groq_never_sends_openai_key(monkeypatch):
    from toolforge.config import Settings
    from toolforge.llm import make_llm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-be-used")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_groq_key_that_should_be_used")
    llm = make_llm(Settings(provider="groq"))
    assert llm.http.headers["Authorization"] == "Bearer gsk_groq_key_that_should_be_used"
    assert llm.base_url == "https://api.groq.com/openai/v1"

    monkeypatch.delenv("GROQ_API_KEY")
    assert "sk-openai" not in make_llm(Settings(provider="groq")).http.headers["Authorization"]


def test_auth_failure_raises_actionable_error(monkeypatch):
    import pytest

    from toolforge.llm import LLMAuthError

    llm = _llm(lambda req: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    with pytest.raises(LLMAuthError, match="Invalid API Key") as info:
        llm.complete("s", "p")
    assert "key sent:" in str(info.value) and "api_key argument" in str(info.value)


def test_unknown_model_lists_available_models():
    import pytest

    from toolforge.llm import LLMModelError

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "model-b", "active": True}, {"id": "model-a"},
                                                      {"id": "retired", "active": False}]})
        return httpx.Response(404, json={"error": {"message": "The model `m` does not exist",
                                                   "code": "model_not_found"}})

    with pytest.raises(LLMModelError) as info:
        _llm(handler).complete("s", "p")
    msg = str(info.value)
    assert "does not exist" in msg and "model-a, model-b" in msg and "retired" not in msg


def test_native_tool_call_rejected_by_provider_is_salvaged():
    """gpt-oss on Groq: the model emits a native tool call, Groq answers 400 with failed_generation."""
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(400, json={"error": {
            "message": "Tool call validation failed: attempted to call tool 'date_diff' which was not in request.tools",
            "code": "tool_use_failed",
            "failed_generation": '{"name": "functions.date_diff", "arguments": "{\\"start\\": \\"2024-01-15\\"}"}'}})

    reply = _llm(handler).complete_json("s", "p")
    assert reply == {"action": "call", "tool": "date_diff", "args": {"start": "2024-01-15"}}
    assert len(seen) == 1  # salvaged without a second request


def test_json_validation_failure_retries_without_json_mode():
    import json as _json

    bodies = []

    def handler(req):
        body = _json.loads(req.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "Failed to validate JSON.",
                                                       "code": "json_validate_failed"}})
        return httpx.Response(200, json={"choices": [{"message": {
            "content": 'Here you go:\n```json\n{"action": "final", "answer": "46"}\n```'}}]})

    assert _llm(handler).complete_json("s", "p") == {"action": "final", "answer": "46"}
    assert "response_format" not in bodies[1] and "Do not use native" in bodies[1]["messages"][1]["content"]


def test_unrecoverable_bad_request_becomes_generation_error():
    import pytest

    from toolforge.llm import LLMGenerationError

    llm = _llm(lambda req: httpx.Response(400, json={"error": {"message": "nope", "code": "json_validate_failed"}}))
    with pytest.raises(LLMGenerationError):  # a ValueError, so graph nodes recover instead of crashing
        llm.complete_json("s", "p")


def test_bad_request_mentioning_model_is_not_misread_as_missing_model():
    import pytest

    from toolforge.llm import LLMGenerationError

    llm = _llm(lambda req: httpx.Response(400, json={"error": {
        "message": "The model produced invalid JSON", "code": "json_validate_failed"}}))
    with pytest.raises(LLMGenerationError):
        llm.complete_json("s", "p")
