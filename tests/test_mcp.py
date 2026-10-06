import anyio
import pytest
from fakes import draft

from toolforge.embeddings import HashingEmbedder
from toolforge.knowledge import Knowledge
from toolforge.models import ToolDraft, Verification
from toolforge.registry import Registry

mcp = pytest.importorskip("mcp")


def test_mcp_server_lists_and_calls_forged_tools(tmp_path):
    from mcp import Client

    from toolforge.mcp_server import build_server

    registry = Registry(str(tmp_path / "lib.db"))
    Knowledge(registry, HashingEmbedder()).add_tool(
        ToolDraft(**draft()), origin_task="t",
        verification=Verification(tests_passed=3, tests_total=3, differential="passed", fuzz_cases=40))
    server = build_server(registry=registry)

    async def scenario():
        async with Client(server) as client:
            listed = await client.list_tools()
            ok = await client.call_tool("days_between", {"start": "2024-01-15", "end": "2024-03-01"})
            bad = await client.call_tool("days_between", {"start": "garbage", "end": "2024-03-01"})
            return listed, ok, bad

    listed, ok, bad = anyio.run(scenario)

    tool = listed.tools[0]
    assert tool.name == "days_between" and "differentially fuzzed on 40 inputs" in tool.description
    assert ok.content[0].text == "46" and not ok.is_error
    assert bad.is_error and "invalid date" in bad.content[0].text
    assert registry.get("days_between").uses == 2
