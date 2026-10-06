import pytest
from fakes import DAYS_BETWEEN, Brain, draft

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import toolforge.api as api  # noqa: E402


def test_rest_api_round_trip(make_forge, monkeypatch):
    brain = Brain().plan("How many days", ("days_between", "Number of days between two ISO dates."))
    brain.references["days_between"] = DAYS_BETWEEN
    brain.will_write("days_between", draft())
    agent = make_forge(brain)
    monkeypatch.setattr(api, "forge", lambda: agent)
    client = TestClient(api.app)

    run = client.post("/run", json={"task": "How many days between 2024-01-15 and 2024-03-01?"}).json()
    assert run["created"] == ["days_between"] and "46" in run["answer"]

    assert [t["name"] for t in client.get("/tools").json()] == ["days_between"]
    detail = client.get("/tools/days_between").json()
    assert detail["latest"]["verification"]["differential"] == "passed"
    call = client.post("/tools/days_between/call", json={"args": {"start": "2024-01-01", "end": "2024-01-31"}})
    assert call.json()["result"] == 30
    assert client.post("/tools/days_between/call", json={"args": {"start": "x", "end": "y"}}).status_code == 422
    assert client.get("/stats").json()["active_tools"] == 1
    assert client.get("/tools/nope").status_code == 404
