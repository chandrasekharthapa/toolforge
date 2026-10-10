"""Both security layers, tested independently: static policy and runtime sandbox."""

import pytest

from toolforge.models import TestCase
from toolforge.safety import check_code
from toolforge.sandbox import Sandbox

sandbox = Sandbox(timeout=3, memory_mb=256)


def rules(code: str, name: str = "f") -> set[str]:
    return {v.rule for v in check_code(code, name)}


@pytest.mark.parametrize("code, rule", [
    ("import os\ndef f(): return os.getcwd()", "import"),
    ("import subprocess\ndef f(): pass", "import"),
    ("from urllib import request\ndef f(): pass", "import"),
    ("def f(): return eval('1+1')", "builtin"),
    ("def f(): return open('/etc/passwd').read()", "builtin"),
    ("def f(): return __import__('os')", "builtin"),
    ("def f(): return ().__class__.__base__.__subclasses__()", "dunder"),
    ("def f(): return getattr(f, 'x')", "builtin"),
    ("import io\ndef f(): return io.open('x')", "attribute"),
    ("import time\ndef f(): time.sleep(100)", "attribute"),
    ("def f(): return '{0.__class__}'.format(1)", "dunder"),
    ("def f(x): return x._secret", "private"),
    ("async def f(): pass", "async"),
    ("print('side effect at import')\ndef f(): pass", "contract"),
    ("def g(): pass", "contract"),
    ("def f(:", "syntax"),
])
def test_static_policy_rejects(code, rule):
    assert rule in rules(code)


def test_static_policy_accepts_ordinary_code():
    code = ("import math, re\nfrom collections import Counter\nfrom datetime import date\n"
            "PI = math.pi\n\ndef f(text: str) -> dict:\n    \"\"\"doc\"\"\"\n"
            "    return dict(Counter(re.findall(r'\\w+', text.lower())))\n")
    assert check_code(code, "f") == []


@pytest.mark.parametrize("code", [
    "def f():\n    return open('/etc/passwd').read()",
    "def f():\n    open('pwned.txt', 'w').write('x')",
    "def f():\n    import socket\n    socket.create_connection(('example.com', 80))",
    "def f():\n    import os\n    os.system('id')",
    "def f():\n    import os\n    return os.listdir('/')",
    "def f():\n    import subprocess\n    subprocess.run(['id'])",
])
def test_sandbox_blocks_side_effects_even_if_static_check_is_bypassed(code):
    result = sandbox.call(code, "f", {})
    assert not result.ok and result.blocked_by_sandbox, result.error


def test_sandbox_enforces_timeout():
    result = Sandbox(timeout=1).call("def f():\n    while True:\n        pass", "f", {})
    assert not result.ok and "timed out" in result.error


def test_sandbox_enforces_memory_limit():
    result = sandbox.call("def f():\n    return len([0] * (10 ** 9))", "f", {})
    assert not result.ok


def test_sandbox_runs_tests_with_float_tolerance_and_reports_failures():
    code = "def add(a: float, b: float) -> float:\n    return a + b\n"
    tests = [TestCase(args=[0.1, 0.2], expected=0.3), TestCase(kwargs={"a": 1, "b": 1}, expected=3)]
    result = sandbox.run_tests(code, "add", tests)
    assert [r["passed"] for r in result.results] == [True, False]
    assert "test #1: got 2, expected 3" in result.failure_report()


def test_sandbox_allows_stdlib_that_reads_its_own_data():
    code = ("from zoneinfo import ZoneInfo\nfrom datetime import datetime\n"
            "def f(tz: str) -> str:\n    return datetime(2024, 1, 1, tzinfo=ZoneInfo(tz)).isoformat()\n")
    result = sandbox.call(code, "f", {"tz": "Asia/Tokyo"})
    assert result.ok and result.result.endswith("+09:00")


def test_batch_reports_per_input_errors():
    result = sandbox.batch("def f(n):\n    return 10 // n", "f", [{"n": 2}, {"n": 0}])
    assert result.outputs == [{"ok": True, "value": 5}, {"ok": False, "error": "ZeroDivisionError"}]


def test_windows_path_assigns_the_job_before_sending_code_and_always_closes_it(monkeypatch):
    """Runs the Windows code path on any OS with a fake kernel32, to pin the ordering."""
    import toolforge.sandbox as S

    calls = []

    class K32:
        def AssignProcessToJobObject(self, job, handle):  # noqa: N802 - Win32 name
            calls.append(("assign", job))
            return True

        def CloseHandle(self, job):  # noqa: N802
            calls.append(("close", job))
            return True

    monkeypatch.setattr(S, "_windows_job", lambda mb, cpu: (calls.append(("create", mb, cpu)) or (K32(), 7)))
    monkeypatch.setattr(S.subprocess.Popen, "_handle", 1, raising=False)
    sb = S.Sandbox(timeout=2, memory_mb=64)
    proc = sb._run_windows({"mode": "call", "code": "def f():\n    return 5\n", "func": "f", "kwargs": {}},
                           {"PATH": "/usr/bin:/bin"}, ".", 2)
    assert S._parse(proc, "call", 0.0, killed=False).result == 5
    assert calls == [("create", 64, 3), ("assign", 7), ("close", 7)]

    calls.clear()
    assert sb._run_windows({"mode": "call", "code": "def f():\n    while True:\n        pass\n", "func": "f",
                            "kwargs": {}}, {"PATH": "/usr/bin:/bin"}, ".", 1) is None  # timeout
    assert calls[-1] == ("close", 7)


def test_demo_code_after_the_tool_is_stripped_and_anything_else_still_rejected():
    from toolforge.safety import check_code, strip_demo_code

    tool = 'def solve(problem: str) -> str:\n    """Doc."""\n    return problem.upper()\n'
    demo = tool + '\nprint(solve("a"))\n\nif __name__ == "__main__":\n    import sys\n    print(solve(sys.argv[1]))\n'
    code, removed = strip_demo_code(demo)
    assert removed == 4 and check_code(code, "solve") == [] and "sys" not in code
    assert strip_demo_code('"""Module doc."""\n' + tool) == ('"""Module doc."""\n' + tool, 0)  # docstring kept
    loop = tool + "for i in range(3):\n    solve(str(i))\n"
    assert strip_demo_code(loop) == (loop, 0) and check_code(loop, "solve")  # not demo code: still rejected
    other_if = tool + "if True:\n    X = 1\n"
    assert strip_demo_code(other_if)[1] == 0 and check_code(other_if, "solve")
