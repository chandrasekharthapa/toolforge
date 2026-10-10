"""Public benchmark: BIG-Bench Hard tasks used by LATM (Cai et al., 2023).

Four BBH tasks (data: github.com/suzgunmirac/BIG-Bench-Hard, MIT licence, vendored in evals/bbh/):
word_sorting, dyck_languages, logical_deduction_five_objects, tracking_shuffled_objects_five_objects.

Modes
  * direct       - TOOLFORGE_MODEL answers on its own, reasoning step by step
  * toolforge    - Toolforge on TOOLFORGE_MODEL with ONE library per (task, seed): it decides per item
                   whether to make, reuse or skip a tool
  * direct-user  - TOOLFORGE_USER_MODEL (a smaller, cheaper model) answers on its own
  * latm         - LATM's maker/user split. TOOLFORGE_MODEL first makes and verifies a tool from 3
                   held-out demonstration items (never test items); TOOLFORGE_USER_MODEL then plans and
                   answers every test item, seeing the library's tools. If it still needs a new tool, the
                   maker writes it, and those tokens are counted.

    TOOLFORGE_USER_MODEL=<small model> python -m evals.bbh_benchmark --tag latm   # all four modes

Each seed draws a different random sample of ``--n`` items per task, so the spread across seeds reflects
both item sampling and the model's own randomness. Every finished item is saved immediately; re-running
the same command resumes.

    python -m evals.bbh_benchmark                 # 4 tasks x 15 items x 3 seeds x 2 modes
    python -m evals.bbh_benchmark --report        # rebuild the report from saved results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolforge.config import Settings  # noqa: E402
from toolforge.llm import LLMConfigError  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "bbh"
RESULTS = HERE / "results"
TASKS = ("word_sorting", "dyck_languages", "logical_deduction_five_objects",
         "tracking_shuffled_objects_five_objects")
MODES = ("direct", "toolforge", "direct-user", "latm")
USER_MODES = {"direct-user", "latm"}  # results depend on TOOLFORGE_USER_MODEL as well
DEMOS = 3
#: protocol 2: extra solved items, never shown to the maker and never test items, used only to check a
#: tool before it is trusted (3 demos are too few: a tool can fit them and still be wrong)
VALIDATION = 7

FORMAT = {
    "word_sorting": "Reply with only the sorted words, separated by single spaces.",
    "dyck_languages": "Reply with only the closing brackets needed to complete the sequence, separated by spaces.",
    "logical_deduction_five_objects": "Reply with the letter of the correct option in parentheses, e.g. (A).",
    "tracking_shuffled_objects_five_objects": "Reply with the letter of the correct option in parentheses, e.g. (A).",
}

DIRECT_SYSTEM = ("Solve the problem carefully. Think step by step, then end your reply with a final line of the "
                 "form 'Answer: <answer>'.")

_BRACKETS = set("()[]{}<>")


# ----------------------------------------------------------------------------------------- grading

def final_segment(answer: str) -> str:
    """Text after the last 'Answer:' marker if there is one, else the whole reply."""
    parts = re.split(r"answer\s*[:：]", answer or "", flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else (answer or "").strip()


def grade_choice(answer: str, question: str, target: str) -> bool:
    seg = final_segment(answer)
    letters = re.findall(r"\(([A-E])\)", seg)
    if letters:
        return f"({letters[-1]})" == target
    bare = re.fullmatch(r"\s*([A-E])[.)]?\s*", seg)
    if bare:
        return f"({bare.group(1)})" == target
    options = dict(re.findall(r"^\(([A-E])\)\s*(.+?)\s*$", question, flags=re.MULTILINE))
    hits = [k for k, text in options.items() if text.lower().rstrip(".") in seg.lower()]
    return len(hits) == 1 and f"({hits[0]})" == target


def grade_words(answer: str, target: str) -> bool:
    want = target.split()
    got = re.findall(r"[a-z0-9][a-z0-9'&.\-]*", final_segment(answer).lower().replace(",", " "))
    got = [w.rstrip(".") for w in got]
    if got == want:
        return True
    n = len(want)  # accept the sorted list embedded in a short sentence ("Sorted: a b c")
    return any(got[i:i + n] == want for i in range(len(got) - n + 1)) and len(got) <= n + 6


def grade_dyck(answer: str, question: str, target: str) -> bool:
    got = [c for c in final_segment(answer) if c in _BRACKETS]
    want = [c for c in target if c in _BRACKETS]
    if got == want:
        return True
    prefix = [c for c in question.split("Input:", 1)[-1] if c in _BRACKETS]
    return got == prefix + want  # the model echoed the whole completed sequence


def grade(task: str, answer: str, question: str, target: str) -> bool:
    if task == "word_sorting":
        return grade_words(answer, target)
    if task == "dyck_languages":
        return grade_dyck(answer, question, target)
    return grade_choice(answer, question, target)


# ------------------------------------------------------------------------------------------ data

def load_task(task: str) -> list[dict[str, str]]:
    return json.loads((DATA / f"{task}.json").read_text(encoding="utf-8"))["examples"]


def sample(task: str, n: int, seed: int) -> list[tuple[int, dict[str, str]]]:
    examples = load_task(task)
    rng = random.Random(f"{task}:{seed}")
    idx = rng.sample(range(len(examples)), min(n, len(examples)))
    return [(i, examples[i]) for i in idx]


def prompt_for(task: str, question: str) -> str:
    return f"{question}\n\n{FORMAT[task]}"


def demo_items(task: str, seed: int, exclude: set[int], k: int = DEMOS) -> list[tuple[int, dict[str, str]]]:
    """Held-out demonstrations for tool making: never one of this seed's test items."""
    examples = load_task(task)
    pool = [i for i in range(len(examples)) if i not in exclude]
    idx = random.Random(f"{task}:{seed}:demos").sample(pool, min(k, len(pool)))
    return [(i, examples[i]) for i in idx]


#: protocol 1 = first published run (tool takes parsed parameters; the user model extracts them).
#: protocol 2 = the maker writes a parse-and-solve tool over the raw problem text, checked against the
#: demos' known answers and repaired against them before any test item is seen.
PROTOCOL = 2


def validation_items(task: str, seed: int, exclude: set[int], k: int = VALIDATION) -> list[tuple[int, dict[str, str]]]:
    examples = load_task(task)
    pool = [i for i in range(len(examples)) if i not in exclude]
    idx = random.Random(f"{task}:{seed}:validation").sample(pool, min(k, len(pool)))
    return [(i, examples[i]) for i in idx]


def make_prompt(task: str, demos: list[tuple[int, dict[str, str]]], protocol: int = PROTOCOL) -> str:
    if protocol == 1:
        shown = "\n\n".join(f"Example {j}:\n{ex['input']}\nCorrect answer: {ex['target']}"
                             for j, (_, ex) in enumerate(demos, 1))
        return ("The examples below are problems of one type, each with its correct answer.\n\n" + shown
                + "\n\nBuild ONE general, reusable tool that solves any problem of this type from its "
                  "parameters (not just these examples), using the examples as test cases. Then use the tool "
                  "to answer Example 1. " + FORMAT[task])
    shown = "\n\n".join(f"Example {j} (the exact text the tool will receive):\n<<<\n{prompt_for(task, ex['input'])}"
                         f"\n>>>\nCorrect answer: {ex['target']}" for j, (_, ex) in enumerate(demos, 1))
    return ("The examples below are problems of one type, each with its correct answer.\n\n" + shown
            + "\n\nBuild ONE general, reusable tool `solve(problem: str) -> str` for problems of this type. "
              "Its only parameter, `problem`, is the full problem text exactly as shown between <<< and >>> "
              "(including the instruction line at the end). The tool must parse that text itself and return "
              "the final answer as a string in exactly the format of the 'Correct answer' lines. These "
              "examples are added to its tests automatically, copied exactly, so do not retype them: write "
              "a few short tests of your own. Then call the tool on Example 1 and give its answer.")


def _problem_param(tool) -> str | None:
    props = (tool.parameters or {}).get("properties", {})
    strings = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"]
    return "problem" if "problem" in props else (strings[0] if len(props) == 1 and strings else None)


def check_demos(agent, task: str, tool_name: str, demos) -> list[dict[str, Any]]:
    """Run a raw-text tool on the solved demos, exactly as it will be called at test time."""
    tool = agent.registry.get(tool_name)
    param = _problem_param(tool) if tool else None
    if tool is None or param is None:
        return [{"ok": False, "why": "tool does not take the problem text"} for _ in demos]
    out = []
    for _, ex in demos:
        res = agent.sandbox.call(tool.code, tool.name, {param: prompt_for(task, ex["input"])})
        got = str(res.result) if res.ok else f"error: {res.error}"
        out.append({"ok": bool(res.ok) and grade(task, got, ex["input"], ex["target"]), "got": got[:200],
                    "expected": ex["target"], "param": param})
    return out


# ------------------------------------------------------------------------------------------- run

def _progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


#: network failures say nothing about the model, so they are retried on the next run, never graded
_TRANSIENT = ("ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError",
              "WriteError", "RemoteProtocolError")


def _transient(e: BaseException | None) -> bool:
    while e is not None:
        if type(e).__name__ in _TRANSIENT:
            return True
        e = e.__cause__ or e.__context__
    return False


#: errors that mean "cannot reach the server at all", as opposed to "this request was too slow"
_UNREACHABLE = ("ConnectError", "ConnectTimeout")
#: runs in which an item may time out before it is graded as a failure to answer in time
MAX_TIMEOUTS = 2


def _unreachable(e: BaseException | None) -> bool:
    while e is not None:
        if type(e).__name__ in _UNREACHABLE:
            return True
        e = e.__cause__ or e.__context__
    return False


class SkipForNow(RuntimeError):
    """The server is up but too slow for this step; leave it for the next run."""


class NetworkDown(RuntimeError):
    """Several network failures in a row: stop instead of skipping every remaining item."""


def _transient_row(row: dict[str, Any]) -> bool:
    return str(row.get("answer", "")).startswith(tuple(f"ERROR {n}" for n in _TRANSIENT))


def _bump_timeout(folder: Path, key: str, index: int) -> int:
    """Count, across runs, how often one item has timed out; returns the new count."""
    path = folder / ".timeouts.json"
    try:
        counts = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        counts = {}
    name = f"{key}#{index}"
    counts[name] = counts.get(name, 0) + 1
    path.write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return counts[name]


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def run_direct(llm, task: str, item: dict[str, str]) -> dict[str, Any]:
    before = llm.usage.snapshot()
    start = time.perf_counter()
    answer = llm.complete(DIRECT_SYSTEM, prompt_for(task, item["input"]), temperature=0.2)
    used = llm.usage.since(before)
    return {"answer": answer[-2000:], "tokens": used.total_tokens, "llm_calls": used.calls,
            "latency_s": round(time.perf_counter() - start, 2), "tool_calls": 0, "created": [], "reused": []}


def run_toolforge(agent, task: str, item: dict[str, str]) -> dict[str, Any]:
    r = agent.run(prompt_for(task, item["input"]))
    return {"answer": r.answer[-2000:], "tokens": r.total_tokens, "user_tokens": r.user_tokens,
            "llm_calls": r.llm_calls, "latency_s": r.latency_s, "tool_calls": len(r.tool_calls),
            "created": r.created, "reused": r.reused, "failed": r.failed_needs, "repaired": r.repaired,
            # the first calls, clipped, so a failure can be diagnosed from the saved rows alone
            "calls": [json.dumps({k: c.get(k) for k in ("call", "args", "result", "error")}, default=str)[:400]
                      for c in r.tool_calls[:4]]}


def _make_tools(make_agent: Callable[[str, str], Any], task: str, seed: int, items, folder: Path,
                log: Callable[[str], None], protocol: int = PROTOCOL) -> None:
    """LATM phase 1, once per (task, seed): the maker builds the library from held-out demos."""
    path = folder / f"latm-make__{task}__s{seed}.json"
    if path.exists():
        return
    demos = demo_items(task, seed, {i for i, _ in items})
    agent = make_agent(str(folder / f"latm__{task}__s{seed}.db"), "latm-make")
    check: dict[str, Any] = {}
    try:
        text = make_prompt(task, demos, protocol)
        if protocol >= 2 and hasattr(agent, "repair_tool"):
            from toolforge.models import TestCase

            known = [TestCase(kwargs={"problem": prompt_for(task, ex["input"])}, expected=ex["target"])
                     for _, ex in demos]
            r = agent.run(text, known_tests=known)
        else:
            r = agent.run(text)
        if protocol >= 2 and r.created and hasattr(agent, "repair_tool"):
            held = validation_items(task, seed, {i for i, _ in items} | {i for i, _ in demos})
            check = _check_and_repair(agent, task, r.created[0], demos, held, make_prompt(task, demos, protocol))
    except Exception as e:
        agent.close()
        if _transient(e):
            # a half-built library would be "reused" by the rebuild and its record would say nothing
            # was built, so the interrupted attempt is wiped and the next run starts clean
            for suffix in ("", "-wal", "-shm", "-journal"):
                Path(str(folder / f"latm__{task}__s{seed}.db") + suffix).unlink(missing_ok=True)
            cls = NetworkDown if _unreachable(e) else SkipForNow
            raise cls(f"{type(e).__name__} while building the {task} tool") from e
        raise
    agent.close()
    first = demos[0][1]
    record = {"demos": [i for i, _ in demos], "created": r.created, "failed": r.failed_needs,
              "tokens": r.total_tokens + check.get("repair_tokens", 0), "llm_calls": r.llm_calls,
              "latency_s": r.latency_s, "answer": r.answer[-500:], "protocol": protocol,
              "correct": grade(task, r.answer, first["input"], first["target"]), **check}
    path.write_text(json.dumps(record), encoding="utf-8")
    demo_note = (f", solved examples {check['demos_before']}/{check['checked']} -> "
                 f"{check['demos_after']}/{check['checked']}" if check else "")
    log(f"[latm-make__{task}__s{seed}] built {r.created or 'nothing'} from {len(demos)} demos "
        f"({record['tokens']} tokens{demo_note})")


def _check_and_repair(agent, task: str, name: str, demos, held, make_text: str) -> dict[str, Any]:
    """Known-answer gate: a tool must solve the demos it was built from AND held-out solved items it
    never saw. If it does not, the failures become sticky tests and the maker repairs it (once)."""
    from toolforge.models import TestCase

    items = [*demos, *held]
    before = check_demos(agent, task, name, items)
    out = {"checked": len(items), "demos_before": sum(c["ok"] for c in before),
           "demos_after": sum(c["ok"] for c in before), "repair_tokens": 0, "repair_ok": None}
    if all(c["ok"] for c in before) or not before or before[0].get("param") is None:
        return out
    param = before[0]["param"]
    failing = [(ex, c) for (_, ex), c in zip(items, before) if not c["ok"]]
    shown = {ex["input"] for _, ex in demos}
    cases = [ex for _, ex in demos] + [ex for ex, _ in failing if ex["input"] not in shown]
    tests = [TestCase(kwargs={param: prompt_for(task, ex["input"])}, expected=ex["target"]) for ex in cases]
    reason = (f"Checked against {len(items)} solved examples of this problem type, the tool got "
              f"{len(failing)} wrong:\n"
              + "\n".join(f"- {ex['input'][-160:]!r}: got {c['got']!r}, expected {ex['target']!r}"
                           for ex, c in failing[:5])
              + "\nThese examples are now test cases (full problem text in, exact answer string out). Fix the "
                "parsing and solving so it is correct for every problem of this type, not just these.")
    fixed = agent.repair_tool(name, tests, reason, task=make_text)
    after = check_demos(agent, task, fixed.get("tool", name), items)
    out.update(demos_after=sum(c["ok"] for c in after), repair_tokens=fixed.get("tokens", 0),
               repair_ok=bool(fixed.get("ok")))
    return out


def run_all(tasks: list[str], seeds: list[int], n: int, modes: list[str], checkpoint: Path,
            make_llm: Callable[[str], Any], make_agent: Callable[[str, str], Any], delay: float = 0.0,
            log: Callable[[str], None] = print, user_checkpoint: Path | None = None,
            protocol: int = PROTOCOL, latm_checkpoint: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Seed-major order, so complete seeds accumulate first and a partial run is already reportable.

    ``make_llm(role)`` gives the "maker" or "user" model; ``make_agent(db, role)`` gives a Toolforge for
    role "maker" (toolforge mode), "latm-make" or "latm-use". Modes that involve the user model are
    saved under ``user_checkpoint``, so changing the user model never mixes results."""
    checkpoint.mkdir(parents=True, exist_ok=True)
    if USER_MODES & set(modes):
        if user_checkpoint is None:
            raise ValueError("modes direct-user and latm need a user model (TOOLFORGE_USER_MODEL)")
        user_checkpoint.mkdir(parents=True, exist_ok=True)
    latm_checkpoint = latm_checkpoint or user_checkpoint
    if latm_checkpoint:
        latm_checkpoint.mkdir(parents=True, exist_ok=True)
    llms = {role: make_llm(role) for role, mode in (("maker", "direct"), ("user", "direct-user")) if mode in modes}
    out: dict[str, list[dict[str, Any]]] = {}
    streak = 0  # consecutive network failures
    for seed in seeds:
        for task in tasks:
            items = sample(task, n, seed)
            for mode in modes:
                folder = latm_checkpoint if mode == "latm" else user_checkpoint if mode in USER_MODES else checkpoint
                key = f"{mode}__{task}__s{seed}"
                path = folder / f"{key}.jsonl"
                rows = _progress(path)
                if any(_transient_row(r) for r in rows):  # an older run saved a network error: retry it
                    rows = [r for r in rows if not _transient_row(r)]
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                done = {r["index"] for r in rows}
                todo = [(i, ex) for i, ex in items if i not in done]
                if todo and rows:
                    log(f"[{key}] resuming: {len(rows)}/{len(items)} done")
                agent = None
                if todo and mode == "toolforge":
                    # one library per (task, seed); kept in the checkpoint folder so a resume keeps its tools
                    agent = make_agent(str(folder / f"{key}.db"), "maker")
                elif todo and mode == "latm":
                    try:
                        _make_tools(make_agent, task, seed, items, folder, log, protocol)
                    except SkipForNow as e:
                        log(f"[latm-make__{task}__s{seed}] {e}; not saved, the next run builds it again")
                        continue
                    agent = make_agent(str(folder / f"latm__{task}__s{seed}.db"), "latm-use")
                for i, ex in todo:
                    try:
                        if mode == "direct":
                            res = run_direct(llms["maker"], task, ex)
                        elif mode == "direct-user":
                            res = run_direct(llms["user"], task, ex)
                        else:
                            res = run_toolforge(agent, task, ex)
                    except LLMConfigError:
                        if agent:
                            agent.close()
                        raise
                    except Exception as e:  # noqa: BLE001 - one crashed item is one wrong item
                        if _transient(e) and _unreachable(e):
                            log(f"[{key}] item {i}: {type(e).__name__} (network), not saved; re-run to retry it")
                            streak += 1
                            if streak >= 3:
                                if agent:
                                    agent.close()
                                raise NetworkDown(f"{streak} connection failures in a row ({type(e).__name__})") from e
                            continue
                        if _transient(e):  # the server is up but this item is too slow, or the stream broke
                            tries = _bump_timeout(folder, key, i)
                            if tries <= MAX_TIMEOUTS:
                                log(f"[{key}] item {i}: {type(e).__name__} (attempt {tries}/{MAX_TIMEOUTS + 1}), "
                                    "not saved; re-run to retry it")
                                continue
                            e = TimeoutError(f"no reply after {tries} attempts ({type(e).__name__})")
                        res = {"answer": f"ERROR {type(e).__name__}: {e}"[:500], "tokens": 0, "llm_calls": 0,
                               "latency_s": 0.0, "tool_calls": 0, "created": [], "reused": []}
                    streak = 0
                    row = {"index": i, "target": ex["target"], **res,
                           "correct": grade(task, res["answer"], ex["input"], ex["target"])}
                    _append(path, row)
                    rows.append(row)
                    log(f"[{key}] {len(rows):>2}/{len(items)} {'✓' if row['correct'] else '✗'} "
                        f"tokens={row['tokens']:>6} tools={row['tool_calls']} created={row['created']} "
                        f"reused={row['reused']}")
                    if delay:
                        time.sleep(delay)
                if agent:
                    agent.close()
                out[key] = rows
    return out


def load_saved(checkpoint: Path, user_checkpoint: Path | None,
               latm_checkpoint: Path | None = None) -> tuple[dict[str, list], dict[str, dict]]:
    """Saved rows for every mode, plus the one-off LATM tool-making records. Each mode is read only
    from its own folder, so protocol-1 latm rows never leak into a protocol-2 report."""
    latm_checkpoint = latm_checkpoint or user_checkpoint
    owner = {"direct": checkpoint, "toolforge": checkpoint, "direct-user": user_checkpoint, "latm": latm_checkpoint}
    rows: dict[str, list] = {}
    makes: dict[str, dict] = {}
    for folder in {f for f in owner.values() if f is not None}:
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.jsonl")):
            if owner.get(p.stem.split("__")[0]) == folder:
                rows[p.stem] = [r for r in _progress(p) if not _transient_row(r)]
    if latm_checkpoint and latm_checkpoint.exists():
        for p in sorted(latm_checkpoint.glob("latm-make__*.json")):
            makes[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return rows, makes


# ---------------------------------------------------------------------------------------- report

def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarize(results: dict[str, list[dict[str, Any]]], makes: dict[str, dict] | None = None) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for key, rows in results.items():
        if not rows:
            continue
        mode, task, s = key.split("__")
        cells.setdefault((mode, task), {})[int(s[1:])] = {
            "acc": sum(r["correct"] for r in rows) / len(rows),
            "tokens": statistics.fmean(r["tokens"] for r in rows),
            "user_tokens": statistics.fmean(r.get("user_tokens", 0) for r in rows),
            "latency": statistics.fmean(r["latency_s"] for r in rows),
            "tool_frac": sum(1 for r in rows if r["tool_calls"] > 0) / len(rows),
            "tool_items": sum(1 for r in rows if r["tool_calls"] > 0),
            "tool_correct": sum(1 for r in rows if r["tool_calls"] > 0 and r["correct"]),
            "built": sum(len(r.get("created", [])) for r in rows),
            "repaired": sum(len(r.get("repaired", [])) for r in rows),
            "n": len(rows),
        }
    summary: dict[str, Any] = {"cells": {}, "overall": {}, "make": {}}
    for (mode, task), by_seed in cells.items():
        accs = [v["acc"] for v in by_seed.values()]
        summary["cells"][f"{mode}|{task}"] = {
            "seeds": sorted(by_seed), "items_per_seed": [by_seed[s]["n"] for s in sorted(by_seed)],
            "acc_mean": _mean_sd(accs)[0], "acc_sd": _mean_sd(accs)[1],
            "tokens_mean": statistics.fmean(v["tokens"] for v in by_seed.values()),
            "user_tokens_mean": statistics.fmean(v["user_tokens"] for v in by_seed.values()),
            "latency_mean": statistics.fmean(v["latency"] for v in by_seed.values()),
            "tool_frac": statistics.fmean(v["tool_frac"] for v in by_seed.values()),
            "tool_items": sum(v["tool_items"] for v in by_seed.values()),
            "tool_correct": sum(v["tool_correct"] for v in by_seed.values()),
            "built_during_use": sum(v["built"] for v in by_seed.values()),
            "repaired_during_use": sum(v["repaired"] for v in by_seed.values()),
        }
    modes = sorted({m for m, _ in cells})
    for mode in modes:
        seeds = sorted({s for (m, _), bs in cells.items() if m == mode for s in bs})
        per_seed = []
        for s in seeds:
            vals = [bs[s]["acc"] for (m, _), bs in cells.items() if m == mode and s in bs]
            if len(vals) == len({t for m, t in cells if m == mode}):
                per_seed.append(statistics.fmean(vals))  # macro average over tasks, only complete seeds
        tok = [bs[s]["tokens"] for (m, _), bs in cells.items() if m == mode for s in bs]
        summary["overall"][mode] = {"acc_mean": _mean_sd(per_seed)[0], "acc_sd": _mean_sd(per_seed)[1],
                                    "complete_seeds": len(per_seed),
                                    "tokens_mean": statistics.fmean(tok) if tok else 0.0}
    by_task: dict[str, list[dict]] = {}
    for key, rec in (makes or {}).items():
        by_task.setdefault(key.split("__")[1], []).append(rec)
    for task, recs in by_task.items():
        summary["make"][task] = {"seeds": len(recs), "built": sum(1 for r in recs if r["created"]),
                                 "tokens_mean": statistics.fmean(r["tokens"] for r in recs),
                                 "demo_correct": sum(1 for r in recs if r["correct"])}
    return summary


def _short(model: str | None) -> str:
    return (model or "?").split(":")[-1].split("/")[-1]


def mode_label(mode: str, meta: dict[str, Any]) -> str:
    if not meta.get("user_model"):
        return {"direct": "direct", "toolforge": "Toolforge"}.get(mode, mode)
    return {"direct": "big, direct", "toolforge": "big + Toolforge", "direct-user": "small, direct",
            "latm": "small + big's tools"}.get(mode, mode)


def render(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    tasks = [t for t in TASKS if any(k.endswith(f"|{t}") for k in summary["cells"])]
    modes = [m for m in MODES if m in summary["overall"]]
    who = (f"big = `{_short(meta['model'])}`, small = `{_short(meta['user_model'])}`" if meta.get("user_model")
           else f"`{meta['model']}`")
    lines = [f"## BIG-Bench Hard (LATM tasks): {who} · {meta['n']} items per task per seed · "
             f"seeds {meta['seeds']} · {meta['date']}", "",
             "Accuracy, mean ± sample SD across seeds (each seed is a different random sample of items).", "",
             "| task | " + " | ".join(mode_label(m, meta) for m in modes) + " |", "|---" * (len(modes) + 1) + "|"]
    for t in tasks:
        row = [t.replace("_", " ")]
        for m in modes:
            c = summary["cells"].get(f"{m}|{t}")
            row.append(f"{c['acc_mean']:.0%} ± {c['acc_sd']:.0%}" if c else "–")
        lines.append("| " + " | ".join(row) + " |")
    row = ["**macro average**"]
    for m in modes:
        o = summary["overall"][m]
        row.append(f"**{o['acc_mean']:.0%} ± {o['acc_sd']:.0%}** ({o['complete_seeds']} seeds)")
    lines.append("| " + " | ".join(row) + " |")

    tool_modes = [m for m in ("toolforge", "latm") if m in modes]
    lines += ["", "Tokens per item, and how often a tool was used:", "",
              "| task | " + " | ".join(f"{mode_label(m, meta)}: tokens" for m in modes)
              + "".join(f" | {mode_label(m, meta)}: used a tool | correct with a tool" for m in tool_modes) + " |",
              "|---" * (len(modes) + 2 * len(tool_modes) + 1) + "|"]
    for t in tasks:
        row = [t.replace("_", " ")]
        for m in modes:
            c = summary["cells"].get(f"{m}|{t}")
            if not c:
                row.append("–")
            elif m == "latm":
                big = c["tokens_mean"] - c["user_tokens_mean"]
                row.append(f"{c['tokens_mean']:,.0f} ({c['user_tokens_mean']:,.0f} small + {big:,.0f} big)")
            else:
                row.append(f"{c['tokens_mean']:,.0f}")
        for m in tool_modes:
            c = summary["cells"].get(f"{m}|{t}")
            row.append(f"{c['tool_frac']:.0%}" if c else "–")
            row.append(f"{c['tool_correct']}/{c['tool_items']}" if c and c.get("tool_items") else "–")
        lines.append("| " + " | ".join(row) + " |")

    if summary.get("make"):
        lines += ["", f"One-off tool making for `latm` (big model, {DEMOS} held-out demos, once per task and seed):",
                  "", "| task | seeds with a verified tool | mean tokens to build | tools built during use "
                      "| tools repaired after failing in use |", "|---|---|---|---|---|"]
        for t in tasks:
            mk = summary["make"].get(t)
            c = summary["cells"].get(f"latm|{t}")
            if mk:
                built, fixed = (c["built_during_use"], c.get("repaired_during_use", 0)) if c else ("–", "–")
                lines.append(f"| {t.replace('_', ' ')} | {mk['built']}/{mk['seeds']} | {mk['tokens_mean']:,.0f} | "
                             f"{built} | {fixed} |")
    return "\n".join(lines) + "\n"


def plot(summary: dict[str, Any], meta: dict[str, Any], path: Path) -> bool:
    """Grouped bars for the maker/user split: small alone, small + big's tools, big alone."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    series = [("direct-user", "small model alone"), ("latm", "small model + big model's tools"),
              ("direct", "big model alone")]
    series = [(m, label) for m, label in series if m in summary["overall"]]
    tasks = [t for t in TASKS if f"latm|{t}" in summary["cells"]]
    if len(series) < 2 or not tasks:
        return False
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light surface)
    surface, ink, muted = "#fcfcfb", "#1f1f1e", "#6b6a64"
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    width = 0.8 / len(series)
    names = [t.replace("_five_objects", "").replace("_", " ") for t in tasks] + ["macro average"]
    for k, (mode, label) in enumerate(series):
        vals = [summary["cells"][f"{mode}|{t}"]["acc_mean"] * 100 for t in tasks]
        sds = [summary["cells"][f"{mode}|{t}"]["acc_sd"] * 100 for t in tasks]
        vals.append(summary["overall"][mode]["acc_mean"] * 100)
        sds.append(summary["overall"][mode]["acc_sd"] * 100)
        xs = [i + (k - (len(series) - 1) / 2) * width for i in range(len(names))]
        ax.bar(xs, vals, width * 0.92, color=colors[k], label=label, edgecolor=surface, linewidth=2, zorder=2)
        ax.errorbar(xs, vals, yerr=sds, fmt="none", ecolor=muted, elinewidth=1, capsize=2, zorder=3)
        for x, v, sd in zip(xs, vals, sds):
            ax.text(x, min(v + sd, 100) + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=ink)
    ax.set_xticks(range(len(names)), names, fontsize=9, color=ink)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=muted)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfcec8")
    ax.tick_params(length=0)
    ax.axvline(len(tasks) - 0.5, color="#cfcec8", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_title(f"BIG-Bench Hard accuracy, mean ± SD over {len(meta.get('seeds', []))} seeds · small = "
                 f"{_short(meta.get('user_model'))}, big = {_short(meta.get('model'))}", fontsize=9.5,
                 color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.08),
              labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--modes", nargs="+", choices=MODES,
                    help="default: direct toolforge, plus direct-user latm when TOOLFORGE_USER_MODEL is set")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=15, help="items per task per seed")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--report", action="store_true", help="rebuild the report from saved progress")
    ap.add_argument("--protocol", type=int, choices=[1, 2], default=PROTOCOL,
                    help="latm protocol: 1 = parsed-parameter tools (first run), 2 = raw-text tools checked "
                         "against the demos (default)")
    ap.add_argument("--tag", default=None, help="output suffix: results/bbh_<tag>.md (default: 'latm' when a "
                                                "user model is set, none otherwise)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    settings = Settings()
    model = f"{settings.provider}:{settings.model}"
    user_model = (f"{settings.user_provider or settings.provider}:{settings.user_model}"
                  if settings.user_model else None)
    modes = args.modes or (list(MODES) if user_model else ["direct", "toolforge"])
    if USER_MODES & set(modes) and not user_model:
        ap.error("modes direct-user and latm need TOOLFORGE_USER_MODEL (the small tool-using model)")
    tag = args.tag if args.tag is not None else (("latm" if args.protocol == 1 else f"latm_v{args.protocol}")
                                                 if user_model else "")
    key = hashlib.sha256(json.dumps([model, settings.embedder, args.n]).encode()).hexdigest()[:10]
    checkpoint = RESULTS / ".bbh_progress" / key
    user_checkpoint = (checkpoint / ("user_" + hashlib.sha256(user_model.encode()).hexdigest()[:8])
                       if user_model else None)
    # direct-user rows do not depend on the latm protocol, so every protocol shares them; latm rows do
    latm_checkpoint = (user_checkpoint / f"p{args.protocol}" if user_checkpoint and args.protocol != 1
                       else user_checkpoint)
    meta = {"model": model, "user_model": user_model, "protocol": args.protocol, "embedder": settings.embedder,
            "n": args.n,
            "seeds": args.seeds, "modes": modes, "date": time.strftime("%Y-%m-%d")}

    if not args.report:
        from toolforge.agent import Toolforge
        from toolforge.llm import make_llm, make_user_llm

        def make_agent(db: str, role: str):
            s = Settings()
            s.db_path = db
            s.plan_with_library = role == "latm-use"
            s.min_needs = 1 if role == "latm-make" else 0  # the make phase exists to produce a tool
            if role == "latm-make" and args.protocol >= 2:
                s.max_repairs = max(s.max_repairs, 5)  # one-off build of a parser: allow a longer repair loop
            if role == "latm-use" and args.protocol >= 2:
                s.answer_from_task_tool = True
            if role != "latm-use":
                s.user_model = None  # the big model does everything
            return Toolforge(s)

        def make_role_llm(role: str):
            return make_user_llm(settings) if role == "user" else make_llm(settings)

        print(f"BBH: {len(args.tasks)} tasks x {args.n} items x {len(args.seeds)} seeds x {modes} on {model}"
              + (f", user model {user_model}" if user_model else ""))
        try:
            run_all(args.tasks, args.seeds, args.n, modes, checkpoint, make_role_llm, make_agent,
                    delay=args.delay, user_checkpoint=user_checkpoint, protocol=args.protocol,
                    latm_checkpoint=latm_checkpoint)
        except LLMConfigError as e:
            print(f"\nstopped: {e}\n  Every finished item is saved; run the same command again to resume.",
                  file=sys.stderr)
            return 2
        except NetworkDown as e:
            print(f"\nstopped: the network looks down ({e}).\n  Every finished item is saved; run the same "
                  "command again once you are back online.", file=sys.stderr)
            return 3
        except KeyboardInterrupt:
            print("\nstopped. Every finished item is saved; run the same command again to resume.", file=sys.stderr)
            return 130
    saved, makes = load_saved(checkpoint, user_checkpoint, latm_checkpoint)
    wanted_seeds = {f"s{x}" for x in args.seeds}
    results = {k: v for k, v in saved.items()
               if (parts := k.split("__"))[0] in modes and parts[1] in args.tasks and parts[2] in wanted_seeds}
    makes = {k: v for k, v in makes.items() if k.split("__")[1] in args.tasks and k.split("__")[2] in wanted_seeds}
    summary = summarize(results, makes)
    RESULTS.mkdir(exist_ok=True)
    stem = "bbh" + (f"_{tag}" if tag else "")
    if args.report and (RESULTS / f"{stem}.json").exists():  # a rebuilt report keeps the date of the run
        try:
            meta["date"] = json.loads((RESULTS / f"{stem}.json").read_text(encoding="utf-8"))["meta"]["date"]
        except (KeyError, ValueError):
            pass
    (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": results,
                                                      "make": makes}, indent=1), encoding="utf-8")
    md = render(summary, meta)
    (RESULTS / f"{stem}.md").write_text(md, encoding="utf-8")
    if "latm" in summary["overall"]:
        plot(summary, meta, RESULTS / f"{stem}.png")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
