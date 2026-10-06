"""REST API (FastAPI). Run with ``toolforge serve`` and open http://localhost:8000/docs."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import Toolforge
from .curator import curate

app = FastAPI(title="Toolforge", version="0.1.0",
              description="An agent that writes, verifies and reuses its own tools.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def forge() -> Toolforge:
    return Toolforge()


class RunRequest(BaseModel):
    task: str = Field(min_length=3, max_length=4000)


class CallRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest) -> dict[str, Any]:
    result = await run_in_threadpool(forge().run, req.task)
    return result.model_dump()


@app.get("/tools")
def tools(status: str | None = "active") -> list[dict[str, Any]]:
    return [t.model_dump(exclude={"code", "tests"}) | {"success_rate": t.success_rate}
            for t in forge().registry.list_tools(status)]


@app.get("/tools/{name}")
def tool(name: str) -> dict[str, Any]:
    versions = forge().registry.versions(name)
    if not versions:
        raise HTTPException(404, f"no tool named {name!r}")
    return {"latest": versions[-1].model_dump(), "versions": [v.version for v in versions]}


@app.post("/tools/{name}/call")
async def call(name: str, req: CallRequest) -> dict[str, Any]:
    f = forge()
    t = f.registry.get(name)
    if t is None:
        raise HTTPException(404, f"no active tool named {name!r}")
    res = await run_in_threadpool(f.sandbox.call, t.code, t.name, req.args)
    f.registry.record_use(t.id, res.ok)
    if not res.ok:
        raise HTTPException(422, res.error)
    return {"result": res.result, "ms": round(res.duration_s * 1000)}


@app.post("/tools/{name}/retire")
def retire(name: str) -> dict[str, Any]:
    return {"retired": forge().registry.set_status(name, "retired")}


@app.get("/lessons")
def lessons() -> list[dict[str, Any]]:
    return [lesson.model_dump() for lesson in forge().registry.list_lessons()]


@app.get("/runs")
def runs(limit: int = 50) -> list[dict[str, Any]]:
    return forge().registry.runs(limit)


@app.get("/stats")
def stats() -> dict[str, Any]:
    return forge().registry.stats()


@app.post("/curate")
def curate_library(apply: bool = False) -> dict[str, Any]:
    report = curate(forge().knowledge, apply=apply)
    return report.__dict__
