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


def test_daily_quota_stops_immediately_with_a_clear_message(monkeypatch):
    import pytest

    from toolforge.llm import LLMConfigError, LLMQuotaError

    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    groq_429 = {"error": {"type": "tokens", "code": "rate_limit_exceeded", "message": (
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, "
        "Used 199500, Requested 1500. Please try again in 7m12.5s.")}}
    llm = _llm(lambda req: httpx.Response(429, json=groq_429))
    with pytest.raises(LLMQuotaError, match="7-minute wait") as info:
        llm.complete("s", "p")
    assert isinstance(info.value, LLMConfigError)  # so the benchmark stops (keeping its progress)
    assert "tokens per day" in str(info.value) and slept == []


def test_short_rate_limit_waits_visibly(monkeypatch, capsys):
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    replies = iter([httpx.Response(429, json={"error": {"message": "Please try again in 1.5s."}}),
                    httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    assert _llm(lambda req: next(replies)).complete("s", "p") == "ok"
    assert slept == [1.5] and "rate-limited by openai; waiting 2s" in capsys.readouterr().err


def test_parse_wait_formats():
    from toolforge.llm import parse_wait

    assert parse_wait("Please try again in 7m12.5s.") == 432.5
    assert parse_wait("try again in 1h2m") == 3720
    assert parse_wait("try again in 850ms") == 0.85
    assert parse_wait("try again in 30s") == 30
    assert parse_wait("no hint here") is None


def test_nvidia_preset_uses_its_own_key_and_endpoint(monkeypatch):
    from toolforge.config import Settings
    from toolforge.llm import make_llm

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key-0000000000")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_must_not_be_sent_to_nvidia")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-be-sent-either")
    llm = make_llm(Settings(provider="nvidia", model="openai/gpt-oss-120b"))
    assert llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert llm.http.headers["Authorization"] == "Bearer nvapi-test-key-0000000000"
    assert llm.model == "openai/gpt-oss-120b" and llm.provider == "nvidia"


def test_project_env_file_wins_over_stale_editor_copy(tmp_path):
    """VS Code copies .env into each new terminal; after .env is edited, that stale copy must not win."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / ".env").write_text("TOOLFORGE_PROVIDER=nvidia   # switch back: groq\nTOOLFORGE_MODEL=m-new\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("TOOLFORGE_")}
    env = {**clean, "TOOLFORGE_PROVIDER": "groq", "TOOLFORGE_MODEL": "m-old", "TOOLFORGE_TOP_K": "9",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    code = ("from toolforge.config import Settings, env_source as w; s = Settings(); "
            "print(s.provider, s.model, s.top_k, w('TOOLFORGE_PROVIDER'), w('TOOLFORGE_TOP_K'), w('TOOLFORGE_DB'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                         check=True).stdout.split()
    # .env wins for what it defines; the shell still supplies what .env does not define
    assert out == ["nvidia", "m-new", "9", ".env", "shell", "environment", "default"]


def test_retired_model_410_is_reported_with_alternatives():
    import pytest

    from toolforge.llm import LLMModelError

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": f"vendor/model-{i}"} for i in range(45)]})
        return httpx.Response(410, json={"error": {"message": "This model has reached its end of life."}})

    with pytest.raises(LLMModelError) as info:
        _llm(handler).complete("s", "p")
    msg = str(info.value)
    assert "any more (retired)" in msg and "end of life" in msg and "(45 in total)" in msg


def test_parse_json_survives_reasoning_models():
    think = ('<think>The user wants {days}. Maybe {"a": 1}? No, compute first.</think>\n'
             '{"action": "final", "answer": "46"}')
    assert parse_json(think) == {"action": "final", "answer": "46"}
    prose = 'I considered {"draft": 1} but the answer is {"needs": [{"name_hint": "x", "description": "y"}]}'
    assert parse_json(prose) == {"needs": [{"name_hint": "x", "description": "y"}]}
    nested = 'ok: {"code": "def f(d):\\n    return {\\"k\\": 1}", "tests": []} done'
    assert parse_json(nested)["tests"] == []


def test_probe_and_user_model_errors_name_the_right_setting():
    import pytest

    from toolforge.config import Settings
    from toolforge.llm import LLMModelError, make_user_llm

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "listed-but-dead"}, {"id": "live"}]})
        model = __import__("json").loads(req.content)["model"]
        if model == "live":
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return httpx.Response(404, json={})

    llm = _llm(handler)
    assert llm.probe("live") == "ok" and llm.probe("listed-but-dead").startswith("HTTP 404")

    user = make_user_llm(Settings(provider="groq", model="big", user_model="listed-but-dead"))
    user.http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMModelError, match="TOOLFORGE_USER_MODEL") as info:
        user.complete("s", "p")
    assert "--probe" in str(info.value)
    assert make_user_llm(Settings(provider="groq", model="big", user_model=None)) is None
