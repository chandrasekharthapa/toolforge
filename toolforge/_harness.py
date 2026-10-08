"""Runs inside the sandbox subprocess. Reads one JSON request on stdin, writes one
JSON result on stdout after a marker line. Stdlib only, no package imports.

Defence in depth inside the child process:
  * a PEP 578 audit hook (cannot be removed once installed) denies file writes,
    file reads outside the Python installation, sockets, subprocesses, ctypes,
    and other OS-level side effects;
  * the parent applies rlimits (memory, CPU, file size) and a wall-clock timeout.
"""

import contextlib
import io
import json
import math
import os
import sys
import sysconfig
import traceback

MARKER = "<<<TOOLFORGE_RESULT>>>"
_real_stdout = sys.stdout

_DENY_PREFIXES = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
    "os.fork", "os.forkpty", "os.kill", "os.killpg", "ctypes.", "shutil.", "os.remove",
    "os.unlink", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate",
    "os.symlink", "os.link", "os.putenv", "os.unsetenv", "os.chdir", "glob.glob",
    "webbrowser.", "pty.", "winreg.", "sqlite3.",
    "urllib.Request", "http.client.", "ftplib.", "smtplib.", "telnetlib.", "imaplib.",
    "poplib.", "nntplib.", "msvcrt.", "_winapi.", "mmap.", "fcntl.", "resource.setrlimit",
    "sys._current_frames",
)


def _read_roots():
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots.update(p for p in sysconfig.get_paths().values() if p)
    roots.update(p for p in sys.path if p and os.path.isabs(p))
    roots.add(os.path.dirname(os.path.abspath(__file__)))
    roots.update(("/usr/share/zoneinfo", "/usr/lib/zoneinfo", "/usr/share/lib/zoneinfo",
                  "/etc/zoneinfo", "/etc/localtime"))
    return tuple(os.path.normcase(os.path.realpath(r)) for r in roots if r)


_READ_ROOTS = _read_roots()


class SandboxViolation(PermissionError):
    pass


def _audit(event, args):
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else None
        if isinstance(path, int):
            raise SandboxViolation("[sandbox] opening file descriptors is blocked")
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
        mode = str(mode or "r")
        writing = any(c in mode for c in "wax+") or bool(flags & write_flags)
        real = os.path.normcase(os.path.realpath(os.fsdecode(path)))
        if writing or not real.startswith(_READ_ROOTS):
            raise SandboxViolation(f"[sandbox] file access blocked: {os.fsdecode(path)!r} ({mode})")
        return
    if event in ("os.listdir", "os.scandir"):  # the import system lists stdlib dirs
        target = os.path.normcase(os.path.realpath(os.fsdecode(args[0] if args and args[0] is not None else ".")))
        if not target.startswith(_READ_ROOTS):
            raise SandboxViolation(f"[sandbox] directory listing blocked: {target!r}")
        return
    if event.startswith(_DENY_PREFIXES):
        raise SandboxViolation(f"[sandbox] blocked operation: {event}")


def _jsonable(value):
    def default(o):
        if isinstance(o, (set, frozenset)):
            try:
                return sorted(o)
            except TypeError:
                return list(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if type(o).__name__ in ("Decimal", "Fraction"):
            return float(o)
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, bytes):
            return o.hex()
        return repr(o)

    return json.loads(json.dumps(value, default=default, allow_nan=True))


def _equal(got, expected):
    if isinstance(got, bool) or isinstance(expected, bool):
        return got == expected
    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
        if math.isnan(got) and math.isnan(expected):
            return True
        return math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(got, list) and isinstance(expected, list):
        return len(got) == len(expected) and all(_equal(a, b) for a, b in zip(got, expected))
    if isinstance(got, dict) and isinstance(expected, dict):
        return got.keys() == expected.keys() and all(_equal(got[k], expected[k]) for k in got)
    return got == expected


def _emit(obj):
    _real_stdout.write(MARKER + "\n" + json.dumps(obj, default=repr) + "\n")
    _real_stdout.flush()


def _err(e):
    return f"{type(e).__name__}: {e}"


# Modules whose import touches the filesystem or env in ways the audit hook would
# (correctly) refuse. Importing them before the hook goes up keeps them usable.
_PRELOAD = ("zoneinfo", "decimal", "statistics", "fractions", "hashlib", "datetime",
            "calendar", "unicodedata", "difflib", "random", "uuid", "ipaddress", "csv")


def main():
    request = json.loads(sys.stdin.read())
    for name in _PRELOAD:
        try:
            __import__(name)
        except ImportError:
            pass
    try:  # warm the tz cache for common zones so ZoneInfo() works after the hook
        import zoneinfo
        for tz in ("UTC", "Asia/Kolkata", "America/New_York", "Europe/London"):
            zoneinfo.ZoneInfo(tz)
    except Exception:  # noqa: BLE001
        pass
    # The parent may switch the hook off only to measure a container on its own (red-team
    # --no-audit-hook, Docker backend only). Generated code never controls the request.
    if request.get("audit_hook", True):
        sys.addaudithook(_audit)  # from here on, side effects are denied

    namespace = {"__name__": "toolforge_tool"}
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(request["code"], "<tool>", "exec"), namespace)
    except BaseException as e:  # noqa: BLE001 - report everything
        _emit({"ok": False, "stage": "load", "error": _err(e),
               "trace": traceback.format_exc(limit=4)})
        return

    fn = namespace.get(request["func"])
    if not callable(fn):
        _emit({"ok": False, "stage": "load", "error": f"function {request['func']!r} not defined"})
        return

    mode = request["mode"]
    if mode == "test":
        results = []
        for i, case in enumerate(request["tests"]):
            try:
                with contextlib.redirect_stdout(captured):
                    got = _jsonable(fn(*case.get("args", []), **case.get("kwargs", {})))
                results.append({"i": i, "passed": _equal(got, case.get("expected")),
                                "got": got, "expected": case.get("expected")})
            except BaseException as e:  # noqa: BLE001
                results.append({"i": i, "passed": False, "error": _err(e),
                                "expected": case.get("expected"),
                                "trace": traceback.format_exc(limit=3)})
        _emit({"ok": all(r["passed"] for r in results), "stage": "test",
               "results": results, "stdout": captured.getvalue()[-2000:]})
    elif mode == "call":
        try:
            with contextlib.redirect_stdout(captured):
                value = _jsonable(fn(**request.get("kwargs", {})))
            _emit({"ok": True, "stage": "call", "result": value, "stdout": captured.getvalue()[-2000:]})
        except BaseException as e:  # noqa: BLE001
            _emit({"ok": False, "stage": "call", "error": _err(e), "trace": traceback.format_exc(limit=3)})
    elif mode == "batch":
        outputs = []
        for kwargs in request["inputs"]:
            try:
                with contextlib.redirect_stdout(captured):
                    outputs.append({"ok": True, "value": _jsonable(fn(**kwargs))})
            except BaseException as e:  # noqa: BLE001
                outputs.append({"ok": False, "error": type(e).__name__})
        _emit({"ok": True, "stage": "batch", "outputs": outputs})
    else:
        _emit({"ok": False, "stage": "load", "error": f"unknown mode {mode!r}"})


if __name__ == "__main__":
    main()
