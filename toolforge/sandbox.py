"""Process sandbox for generated code — the second security layer.

Each run is a fresh ``python -I`` subprocess (isolated mode: no env vars, no user
site-packages, no CWD on ``sys.path``) in an empty temp directory with a scrubbed
environment, rlimits on memory / CPU / file size, a wall-clock timeout, and the
in-process audit hook from ``_harness.py``.

This is strong enough for a single-user research tool. For multi-tenant use, run the
same harness inside a container or microVM (see README → Hardening).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import TestCase

HARNESS = Path(__file__).with_name("_harness.py")
MARKER = "<<<TOOLFORGE_RESULT>>>"


@dataclass
class SandboxResult:
    ok: bool
    stage: str = ""
    error: str | None = None
    result: Any = None
    results: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    trace: str = ""
    stdout: str = ""
    duration_s: float = 0.0

    @property
    def blocked_by_sandbox(self) -> bool:
        text = (self.error or "") + self.trace + " ".join(
            str(r.get("error", "")) for r in self.results)
        return "[sandbox]" in text

    def failure_report(self, max_chars: int = 2500) -> str:
        """Human/LLM readable summary of what went wrong."""
        if self.error:
            msg = f"{self.stage} failed: {self.error}"
            return (msg + ("\n" + self.trace if self.trace else ""))[:max_chars]
        lines = []
        for r in self.results:
            if r.get("passed"):
                continue
            if "error" in r:
                lines.append(f"test #{r['i']}: raised {r['error']} (expected {r.get('expected')!r})")
            else:
                lines.append(f"test #{r['i']}: got {r.get('got')!r}, expected {r.get('expected')!r}")
        return "\n".join(lines)[:max_chars] or "unknown failure"


def _limits(memory_mb: int, cpu_s: int):
    def apply() -> None:  # runs in the child between fork and exec (POSIX only)
        import resource

        for limit, value in (
            (resource.RLIMIT_AS, memory_mb * 1024 * 1024),
            (resource.RLIMIT_CPU, cpu_s),
            (resource.RLIMIT_FSIZE, 0),
            (resource.RLIMIT_CORE, 0),
        ):
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):  # e.g. RLIMIT_AS unsupported on macOS
                pass
        os.setsid()

    return apply


class Sandbox:
    def __init__(self, timeout: float = 5.0, memory_mb: int = 256) -> None:
        self.timeout = timeout
        self.memory_mb = memory_mb

    # public API ------------------------------------------------------------
    def run_tests(self, code: str, func: str, tests: list[TestCase]) -> SandboxResult:
        return self._run({"mode": "test", "code": code, "func": func,
                          "tests": [t.model_dump() for t in tests]})

    def call(self, code: str, func: str, kwargs: dict[str, Any]) -> SandboxResult:
        return self._run({"mode": "call", "code": code, "func": func, "kwargs": kwargs})

    def batch(self, code: str, func: str, inputs: list[dict[str, Any]]) -> SandboxResult:
        return self._run({"mode": "batch", "code": code, "func": func, "inputs": inputs},
                         timeout=self.timeout * 3)

    # internals -------------------------------------------------------------
    def _run(self, request: dict[str, Any], timeout: float | None = None) -> SandboxResult:
        timeout = timeout or self.timeout
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0",
               "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"}
        if os.name == "nt":  # Windows needs these to start Python at all
            env.update({k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP") if k in os.environ})
        preexec = _limits(self.memory_mb, int(timeout) + 1) if os.name == "posix" else None

        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="toolforge-") as workdir:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(HARNESS)],
                    input=json.dumps(request),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                    env=env,
                    preexec_fn=preexec,  # noqa: PLW1509 - intentional, sets rlimits
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(False, request["mode"], f"timed out after {timeout:.1f}s",
                                     duration_s=time.perf_counter() - start)
        duration = time.perf_counter() - start

        if MARKER not in proc.stdout:
            tail = (proc.stderr or "").strip().splitlines()[-5:]
            reason = "process killed (memory or CPU limit?)" if proc.returncode < 0 else "no result"
            if any("MemoryError" in t for t in tail):
                reason = "MemoryError: memory limit exceeded"
            return SandboxResult(False, request["mode"], f"{reason}; exit={proc.returncode}",
                                 trace="\n".join(tail), duration_s=duration)

        payload = json.loads(proc.stdout.split(MARKER, 1)[1].strip().splitlines()[0])
        return SandboxResult(
            ok=payload.get("ok", False),
            stage=payload.get("stage", ""),
            error=payload.get("error"),
            result=payload.get("result"),
            results=payload.get("results", []),
            outputs=payload.get("outputs", []),
            trace=payload.get("trace", ""),
            stdout=payload.get("stdout", ""),
            duration_s=duration,
        )
