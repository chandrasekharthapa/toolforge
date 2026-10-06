"""Command-line interface.

    toolforge run "How many days between 2024-01-15 and 2024-03-01?"
    toolforge tools | show <name> | call <name> '{"a": 1}' | lessons | stats | curate [--apply]
    toolforge serve            # REST API on :8000
    toolforge mcp [--forge]    # MCP server over stdio
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import Settings

USE_COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _event(e: dict[str, Any]) -> None:
    node = e["node"]
    if node == "analyze":
        print(c("plan    ", "36"), ", ".join(e["needs"]) or "(no tools needed)")
    elif node == "match":
        best = e["candidates"][0] if e["candidates"] else None
        hint = f" (best: {best['tool']} cos={best['cosine']})" if best else ""
        print(c("match   ", "36"), f"{e['need']} → {e['decision']}{hint} · {e['reason']}")
    elif node == "reuse":
        print(c("reuse   ", "32"), e["tool"])
    elif node == "synthesize":
        kind = "repair" if e.get("repair") else "write"
        print(c(f"{kind:<8}", "33"), f"attempt {e['attempt']}: {e.get('name') or e.get('error')}")
    elif node == "verify":
        if e["ok"]:
            diff = f"differential {e['differential']}"
            if e["differential"] == "passed":
                diff += f" on {e['fuzz_cases']} fuzz inputs"
                if e["disagreements"] and e.get("arbitrated_tests"):
                    diff += (f" ({e['disagreements']} disagreements with the reference, settled by the arbiter in"
                             f" this tool's favour → {e['arbitrated_tests']} regression tests added)")
                elif e["disagreements"]:
                    diff += f" ({e['disagreements']} disagreements, within tolerance)"
            print(c("verify  ", "32"), f"{e['tests_passed']} tests ✓ · {diff}")
        else:
            first = (e.get("detail") or "").splitlines()[:2]
            print(c("verify  ", "31"), f"✗ {e['stage']}: {' | '.join(first)}")
    elif node == "register":
        print(c("register", "32"), f"{e['tool']} v{e['version']}"
              + (f" (+{e['lessons']} lesson)" if e["lessons"] else ""))
    elif node == "give_up":
        print(c("give up ", "31"), e["need"])
    elif node == "execute":
        print(c("execute ", "36"), f"{e['calls']} tool call(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toolforge", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="library database path (default: $TOOLFORGE_DB or toolforge.db)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="solve a task, forging tools as needed")
    p.add_argument("task")
    p.add_argument("--no-rag", action="store_true", help="disable examples + lessons retrieval")
    p.add_argument("--json", action="store_true", help="print the full RunResult as JSON")
    sub.add_parser("tools", help="list active tools")
    p = sub.add_parser("show", help="show a tool's code, tests and provenance")
    p.add_argument("name")
    p = sub.add_parser("call", help="call a tool directly in the sandbox")
    p.add_argument("name")
    p.add_argument("args", nargs="?", default="{}", help="JSON object of arguments")
    p = sub.add_parser("retire", help="retire a tool")
    p.add_argument("name")
    sub.add_parser("lessons", help="list lessons learned from repairs")
    sub.add_parser("stats", help="library and run statistics")
    sub.add_parser("models", help="list the models your provider serves to your key")
    p = sub.add_parser("curate", help="find duplicate / under-performing tools")
    p.add_argument("--apply", action="store_true", help="retire under-performing tools")
    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("mcp", help="serve the tool library over MCP (stdio)")
    p.add_argument("--forge", action="store_true", help="also expose the forging agent as a tool")
    args = parser.parse_args(argv)

    settings = Settings()
    if args.db:
        settings.db_path = args.db

    if args.cmd == "serve":
        import os

        import uvicorn

        os.environ["TOOLFORGE_DB"] = settings.db_path
        uvicorn.run("toolforge.api:app", host=args.host, port=args.port)
        return 0
    if args.cmd == "mcp":
        from .mcp_server import serve_stdio

        serve_stdio(settings, forge=args.forge)
        return 0

    if args.cmd == "models":
        from .llm import LLMConfigError, make_llm

        llm = make_llm(settings)
        if not hasattr(llm, "list_models"):
            print(f"listing models is supported for OpenAI-compatible providers, not {settings.provider!r}")
            return 1
        try:
            for model_id in llm.list_models():
                mark = "  ← configured" if model_id == llm.model else ""
                print(model_id + mark)
        except LLMConfigError as e:
            print(c("error   ", "31") + str(e), file=sys.stderr)
            return 2
        return 0

    from .registry import Registry

    registry = Registry(settings.db_path)

    if args.cmd == "run":
        from .agent import Toolforge
        from .llm import LLMConfigError

        try:
            forge = Toolforge(settings, registry=registry, rag=not args.no_rag)
            result = forge.run(args.task, on_event=None if args.json else _event)
        except LLMConfigError as e:
            print(c("error   ", "31") + str(e), file=sys.stderr)
            return 2
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(c("\nanswer  ", "1"), result.answer)
            print(c(f"         {result.llm_calls} LLM calls · {result.total_tokens} tokens · "
                    f"{result.latency_s:.1f}s", "2"))
    elif args.cmd == "tools":
        tools = registry.list_tools()
        if not tools:
            print("Library is empty. Try: toolforge run \"How many days between 2024-01-15 and 2024-03-01?\"")
        for t in tools:
            rate = f"{t.success_rate:.0%}" if t.success_rate is not None else "–"
            print(f"{c(t.name, '1')} v{t.version}  uses={t.uses} success={rate}  "
                  f"tests={t.verification.tests_total} diff={t.verification.differential}\n    {t.description}")
    elif args.cmd == "show":
        versions = registry.versions(args.name)
        if not versions:
            print(f"no tool named {args.name!r}")
            return 1
        t = versions[-1]
        print(c(f"{t.name} v{t.version} [{t.status}]", "1"), "—", t.description)
        print(c("origin: ", "2") + t.origin_task)
        print(c("verification: ", "2") + t.verification.model_dump_json())
        print(c("schema: ", "2") + json.dumps(t.parameters))
        print("\n" + t.code.rstrip() + "\n")
        for tc in t.tests:
            print(c("  test ", "2") + json.dumps(tc.model_dump(exclude_defaults=True)))
    elif args.cmd == "call":
        from .sandbox import Sandbox

        t = registry.get(args.name)
        if t is None:
            print(f"no active tool named {args.name!r}")
            return 1
        res = Sandbox(settings.sandbox_timeout, settings.sandbox_memory_mb).call(
            t.code, t.name, json.loads(args.args))
        registry.record_use(t.id, res.ok)
        print(json.dumps(res.result) if res.ok else c(res.error or "failed", "31"))
        return 0 if res.ok else 1
    elif args.cmd == "retire":
        print(f"retired {registry.set_status(args.name, 'retired')} version(s)")
    elif args.cmd == "lessons":
        for lesson in registry.list_lessons():
            print(f"- {lesson.mistake} → {lesson.fix}  {c('(' + lesson.need + ')', '2')}")
    elif args.cmd == "stats":
        print(json.dumps(registry.stats(), indent=2))
    elif args.cmd == "curate":
        from .curator import curate
        from .embeddings import make_embedder
        from .knowledge import Knowledge

        report = curate(Knowledge(registry, make_embedder(settings)), apply=args.apply)
        for a, b, s in report.duplicates:
            print(f"possible duplicate: {a} ~ {b} (cos={s})")
        for name, uses, rate in report.underperforming:
            print(f"under-performing: {name} ({uses} uses, {rate:.0%} success)")
        if report.retired:
            print("retired:", ", ".join(report.retired))
        if not (report.duplicates or report.underperforming):
            print("library looks healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
