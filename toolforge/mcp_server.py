"""Serve the forged tool library over the Model Context Protocol.

Any MCP client (Claude Desktop, Claude Code, Cursor, …) can then call the tools that
Toolforge wrote for itself. Every call still runs in the sandbox. The tool list is read
from the registry on each ``tools/list``, so newly forged tools appear without a restart.

With ``--forge`` an extra ``toolforge_solve`` tool is exposed: the client can hand
Toolforge a task, and Toolforge forges whatever tools it is missing.

    toolforge mcp                 # stdio transport (what desktop clients launch)
    toolforge mcp --forge         # also expose the forging agent itself
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server

from .config import Settings
from .registry import Registry
from .sandbox import Sandbox, make_sandbox

SOLVE_TOOL = "toolforge_solve"


def _describe(tool) -> str:
    v = tool.verification
    proof = f"{v.tests_passed} tests passed"
    if v.differential == "passed":
        proof += f", differentially fuzzed on {v.fuzz_cases} inputs"
    return f"{tool.description} [forged by Toolforge · v{tool.version} · {proof}]"


def build_server(settings: Settings | None = None, *, forge: bool = False,
                 registry: Registry | None = None, sandbox: Sandbox | None = None) -> Server:
    settings = settings or Settings()
    registry = registry or Registry(settings.db_path)
    sandbox = sandbox or make_sandbox(settings)
    agent = None

    async def list_tools(ctx, params) -> types.ListToolsResult:
        tools = [types.Tool(name=t.name, description=_describe(t), input_schema=t.parameters)
                 for t in registry.list_tools()]
        if forge:
            tools.append(types.Tool(
                name=SOLVE_TOOL,
                description="Solve a computational task with Toolforge. It reuses verified tools from its "
                            "library or forges, tests and fuzzes new ones, then answers.",
                input_schema={"type": "object", "required": ["task"],
                              "properties": {"task": {"type": "string"}}},
            ))
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        nonlocal agent
        args: dict[str, Any] = params.arguments or {}
        if forge and params.name == SOLVE_TOOL:
            if agent is None:
                from .agent import Toolforge

                agent = Toolforge(settings, registry=registry, sandbox=sandbox)
            result = await anyio.to_thread.run_sync(agent.run, str(args.get("task", "")))
            summary = {"answer": result.answer, "created": result.created, "reused": result.reused}
            return types.CallToolResult(content=[types.TextContent(text=json.dumps(summary))],
                                        structured_content=summary)

        tool = registry.get(params.name)
        if tool is None:
            return types.CallToolResult(content=[types.TextContent(text=f"Unknown tool {params.name!r}")],
                                        is_error=True)
        res = await anyio.to_thread.run_sync(sandbox.call, tool.code, tool.name, args)
        registry.record_use(tool.id, res.ok)
        if not res.ok:
            return types.CallToolResult(content=[types.TextContent(text=res.error or "tool failed")],
                                        is_error=True)
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(res.result))],
                                    structured_content={"result": res.result})

    return Server("toolforge", version="0.1.0",
                  instructions="Tools in this server were written, unit-tested and differentially "
                               "fuzzed by the Toolforge agent. Calls run in a sandbox.",
                  on_list_tools=list_tools, on_call_tool=call_tool)


def serve_stdio(settings: Settings | None = None, *, forge: bool = False) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(settings, forge=forge)

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
