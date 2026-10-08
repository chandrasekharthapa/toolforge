"""Docker backend: command construction always; real containers only when an image is present.

Set TOOLFORGE_TEST_IMAGE to use an image other than python:3.12-slim.
"""

import os
import shutil
import subprocess

import pytest

from toolforge.config import Settings
from toolforge.models import TestCase
from toolforge.sandbox import HARNESS, DockerSandbox, Sandbox, SandboxUnavailable, make_sandbox

IMAGE = os.environ.get("TOOLFORGE_TEST_IMAGE", "python:3.12-slim")


def test_container_is_locked_down():
    cmd = DockerSandbox(timeout=5, memory_mb=128, image="img:tag").command("toolforge-x", 5)
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--user 65534:65534", "--memory 128m", "--memory-swap 128m", "--pids-limit 64", "--rm"):
        assert flag in joined, flag
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a in ("--mount", "-v", "--volume")]
    assert mounts == [f"type=bind,source={HARNESS},target=/toolforge/_harness.py,readonly"]  # nothing else
    assert "-e" not in cmd and "--env" not in cmd  # no host environment crosses over
    assert cmd[-5:] == ["img:tag", "python", "-I", "-B", "/toolforge/_harness.py"]
    assert "--runtime" in DockerSandbox(runtime="runsc").command("n", 5)


def test_factory_picks_the_backend():
    assert type(make_sandbox(Settings(sandbox="process"))) is Sandbox
    sb = make_sandbox(Settings(sandbox="docker", sandbox_image="img:1", sandbox_memory_mb=64))
    assert isinstance(sb, DockerSandbox) and sb.image == "img:1" and sb.memory_mb == 64
    with pytest.raises(SandboxUnavailable):
        make_sandbox(Settings(sandbox="vm"))
    with pytest.raises(SandboxUnavailable):  # the hook is the only runtime layer the process backend has
        make_sandbox(Settings(sandbox="process"), audit_hook=False)


def test_missing_docker_gives_instructions_not_a_traceback():
    with pytest.raises(SandboxUnavailable, match="not on PATH"):
        DockerSandbox(docker="definitely-not-docker-xyz").call("def f():\n    return 1\n", "f", {})


def _image_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


needs_docker = pytest.mark.skipif(not _image_available(), reason=f"docker image {IMAGE} not available")


@needs_docker
def test_tools_run_and_verify_inside_the_container():
    sb = DockerSandbox(timeout=5, image=IMAGE)
    code = "def double(x):\n    return x * 2\n"
    assert sb.call(code, "double", {"x": 21}).result == 42
    assert sb.run_tests(code, "double", [TestCase(args=[2], expected=4), TestCase(args=[3], expected=6)]).ok
    outs = sb.batch(code, "double", [{"x": 1}, {"x": 2}]).outputs
    assert outs == [{"ok": True, "value": 2}, {"ok": True, "value": 4}]
    blocked = sb.call("def f():\n    return open('/etc/passwd').read()\n", "f", {})
    assert not blocked.ok and blocked.blocked_by_sandbox  # the audit hook still runs inside


@needs_docker
def test_container_alone_has_no_network_no_host_files_and_limits():
    sb = DockerSandbox(timeout=3, memory_mb=128, image=IMAGE, audit_hook=False)
    net = sb.call("def f():\n    import socket\n    socket.create_connection(('1.1.1.1', 53), 2)\n", "f", {})
    assert not net.ok and "unreachable" in net.error.lower()
    host = sb.call(f"def f():\n    return open({str(HARNESS.parent / 'sandbox.py')!r}).read()\n", "f", {})
    assert not host.ok and "No such file" in host.error  # only _harness.py is mounted
    ro = sb.call("def f():\n    open('/etc/x', 'w').write('x')\n", "f", {})
    assert not ro.ok
    env = sb.call("def f():\n    import os\n    return dict(os.environ)\n", "f", {})
    assert env.ok and not any("KEY" in k or "TOOLFORGE" in k for k in env.result)
    bomb = sb.call("def f():\n    x = []\n    while True:\n        x.append(bytearray(10**7))\n", "f", {})
    assert not bomb.ok
    spin = sb.call("def f():\n    while True:\n        pass\n", "f", {})
    assert not spin.ok
