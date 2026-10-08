"""Process sandbox for generated code — the second security layer.

Each run is a fresh ``python -I`` subprocess (isolated mode: no env vars, no user
site-packages, no CWD on ``sys.path``) in an empty temp directory with a scrubbed
environment, rlimits on memory / CPU / file size, a wall-clock timeout, and the
in-process audit hook from ``_harness.py``.

On Windows, which has no rlimits, the child is put in a Job Object with the same memory and
CPU caps, a one-process limit and kill-on-close.

That is a guard rail, not a security boundary: PEP 578 audit hooks were never meant to
contain hostile code. ``DockerSandbox`` runs the same harness
in a locked-down container (``TOOLFORGE_SANDBOX=docker``); for multi-tenant use, put that
container on gVisor or use a microVM (see README → Security).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
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


def _windows_job(memory_mb: int, cpu_s: int):
    """A Windows Job Object that does what the POSIX rlimits do: caps committed memory and CPU
    time, forbids child processes, and kills everything in it when the handle is closed.
    Returns ``(kernel32, handle)``, or ``None`` if the job cannot be created."""
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps",
                                                     "ReadBytes", "WriteBytes", "OtherBytes")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.argtypes, k32.CreateJobObjectW.restype = [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes, k32.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = (0x0002 | 0x0008 | 0x0100 | 0x0400 | 0x2000)
    # PROCESS_TIME | ACTIVE_PROCESS | PROCESS_MEMORY | DIE_ON_UNHANDLED_EXCEPTION | KILL_ON_JOB_CLOSE
    info.BasicLimitInformation.PerProcessUserTimeLimit = cpu_s * 10_000_000  # 100 ns units
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory_mb * 1024 * 1024
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # ExtendedLimitInformation
        k32.CloseHandle(job)
        return None
    return k32, job


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
        with tempfile.TemporaryDirectory(prefix="toolforge-", ignore_cleanup_errors=True) as workdir:
            if os.name == "nt":
                proc = self._run_windows(request, env, workdir, timeout)
                if proc is None:
                    return SandboxResult(False, request["mode"], f"timed out after {timeout:.1f}s",
                                         duration_s=time.perf_counter() - start)
            else:
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
        return _parse(proc, request["mode"], time.perf_counter() - start, killed=proc.returncode < 0)

    def _run_windows(self, request: dict[str, Any], env: dict[str, str], workdir: str,
                     timeout: float) -> subprocess.CompletedProcess | None:
        """Windows has no rlimits, so the child goes into a Job Object. It is assigned before
        the request is written: the harness reads stdin before it runs anything, so no
        generated code executes outside the job."""
        try:
            job = _windows_job(self.memory_mb, int(timeout) + 1)
        except (OSError, AttributeError, ValueError):
            job = None
        proc = subprocess.Popen([sys.executable, "-I", str(HARNESS)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=workdir, env=env)
        try:
            if job is not None and not job[0].AssignProcessToJobObject(job[1], int(proc._handle)):
                job[0].CloseHandle(job[1])
                job = None
            if job is None and not getattr(Sandbox, "_warned_no_job", False):
                Sandbox._warned_no_job = True
                print("toolforge: could not create a Windows Job Object; sandbox memory/CPU limits are off "
                      "(use TOOLFORGE_SANDBOX=docker)", file=sys.stderr)
            try:
                out, err = proc.communicate(json.dumps(request), timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return None
            return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
        finally:
            if job is not None:
                job[0].CloseHandle(job[1])  # KILL_ON_JOB_CLOSE: nothing outlives the call


class SandboxUnavailable(RuntimeError):
    """The configured backend cannot run here (e.g. Docker not installed or image not pulled)."""


class DockerSandbox(Sandbox):
    """The same harness, run inside a throwaway, locked-down container.

    What the container adds on top of the process backend:
      * no network at all (``--network none``), not just a denied ``socket`` audit event;
      * a read-only root filesystem, a non-root user, all capabilities dropped and
        ``no-new-privileges`` — host files are simply not there, rather than refused;
      * cgroup limits on memory (no swap), CPU and process count, enforced by the kernel
        rather than by the child process (and on macOS, where ``RLIMIT_AS`` is ignored);
      * only ``_harness.py`` is mounted, read-only. Nothing else from the host is visible.

    It is still a shared-kernel container, not a VM: for hostile multi-tenant code use
    gVisor (``runtime="runsc"``) or a microVM. Each call pays container start-up
    (~0.3 s on Linux, more on Docker Desktop), so this is for running untrusted tools,
    not for fast benchmark loops.
    """

    #: extra wall-clock allowance for container start-up and teardown
    STARTUP_GRACE_S = 10.0

    def __init__(self, timeout: float = 5.0, memory_mb: int = 256, image: str = "python:3.12-slim",
                 docker: str = "docker", python: str = "python", runtime: str | None = None,
                 audit_hook: bool = True) -> None:
        super().__init__(timeout, memory_mb)
        self.image, self.docker, self.python, self.runtime = image, docker, python, runtime
        self.audit_hook = audit_hook
        self._checked = False

    def command(self, name: str, timeout: float) -> list[str]:
        cpu = int(timeout) + 1
        cmd = [self.docker, "run", "--rm", "-i", "--name", name,
               "--network", "none",
               "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
               "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
               "--cpus", "1", "--pids-limit", "64",
               "--ulimit", f"cpu={cpu}:{cpu}", "--ulimit", "core=0:0", "--ulimit", "nofile=64:64",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", "65534:65534",
               "--mount", f"type=bind,source={HARNESS},target=/toolforge/_harness.py,readonly",
               "-w", "/tmp"]
        if self.runtime:
            cmd += ["--runtime", self.runtime]
        return cmd + [self.image, self.python, "-I", "-B", "/toolforge/_harness.py"]

    def check(self) -> None:
        """Fail fast, with instructions, if Docker or the image is missing."""
        if self._checked:
            return
        try:
            probe = subprocess.run([self.docker, "image", "inspect", self.image], capture_output=True,
                                   text=True, timeout=30)
        except FileNotFoundError as e:
            raise SandboxUnavailable(f"TOOLFORGE_SANDBOX=docker but {self.docker!r} is not on PATH. "
                                     "Install Docker Desktop, or set TOOLFORGE_SANDBOX=process.") from e
        except subprocess.TimeoutExpired as e:
            raise SandboxUnavailable("Docker did not answer within 30s - is Docker Desktop running?") from e
        if probe.returncode != 0:
            err = (probe.stderr or "").strip()
            if "No such image" in err or "not found" in err.lower():
                raise SandboxUnavailable(f"Docker image {self.image!r} is not available locally. "
                                         f"Run:  docker pull {self.image}")
            raise SandboxUnavailable(f"Docker is not usable ({err.splitlines()[-1] if err else 'no output'}). "
                                     "Is Docker Desktop running?")
        self._checked = True

    def _run(self, request: dict[str, Any], timeout: float | None = None) -> SandboxResult:
        self.check()
        timeout = timeout or self.timeout
        request = {**request, "audit_hook": self.audit_hook}
        name = f"toolforge-{uuid.uuid4().hex[:12]}"
        start = time.perf_counter()
        try:
            proc = subprocess.run(self.command(name, timeout), input=json.dumps(request), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout + self.STARTUP_GRACE_S)
        except subprocess.TimeoutExpired:
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, timeout=30)
            return SandboxResult(False, request["mode"], f"timed out after {timeout:.1f}s",
                                 duration_s=time.perf_counter() - start)
        duration = time.perf_counter() - start
        if proc.returncode == 125 and MARKER not in proc.stdout:  # docker itself failed
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            return SandboxResult(False, request["mode"], "docker run failed", trace="\n".join(tail),
                                 duration_s=duration)
        # 137 = SIGKILL (cgroup OOM killer), 152 = SIGXCPU (CPU ulimit)
        return _parse(proc, request["mode"], duration, killed=proc.returncode in (137, 152) or proc.returncode < 0)


def _parse(proc: subprocess.CompletedProcess, mode: str, duration: float, killed: bool) -> SandboxResult:
    if MARKER not in proc.stdout:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        reason = "process killed (memory or CPU limit?)" if killed else "no result"
        if any("MemoryError" in t for t in tail):
            reason = "MemoryError: memory limit exceeded"
        return SandboxResult(False, mode, f"{reason}; exit={proc.returncode}",
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


def make_sandbox(settings, *, timeout: float | None = None, audit_hook: bool = True) -> Sandbox:
    """Build the backend ``settings.sandbox`` names ("process" or "docker")."""
    kind = (getattr(settings, "sandbox", "process") or "process").lower()
    timeout = timeout if timeout is not None else settings.sandbox_timeout
    if kind == "docker":
        return DockerSandbox(timeout, settings.sandbox_memory_mb, image=settings.sandbox_image,
                             audit_hook=audit_hook)
    if kind != "process":
        raise SandboxUnavailable(f"unknown TOOLFORGE_SANDBOX={kind!r}; use 'process' or 'docker'")
    if not audit_hook:
        raise SandboxUnavailable("the process backend always installs the audit hook")
    return Sandbox(timeout, settings.sandbox_memory_mb)
