# Toolforge

**An AI agent that writes its own tools, and proves they work before trusting them.**

[![CI](https://github.com/chandrasekharthapa/toolforge/actions/workflows/ci.yml/badge.svg)](https://github.com/chandrasekharthapa/toolforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

When Toolforge meets a task it has no tool for, it writes a Python function, verifies it, sandboxes it,
stores it in a searchable library, and reuses it next time. Agents that make their own tools are
an active research area (LATM, CREATOR, CRAFT, Voyager). The weak point those systems share is **trust**:
the model that wrote the code also wrote the tests, so the tests inherit the code's misunderstandings.

Toolforge is built around closing that gap:

| | |
|---|---|
| 🧪 **Differential verification** | A second, *independent* implementation is written from the spec alone. Both are fuzzed with schema-driven inputs. Disagreements go to an arbiter whose verdicts become permanent regression tests. |
| 🛡️ **Two-layer sandbox, red-teamed** | A static AST policy plus a runtime sandbox (PEP 578 audit hook, rlimits, isolated interpreter, scrubbed env). A 35-payload escape corpus runs in CI: **0 escapes**, and the runtime layer alone contains **35/35**. |
| 🔎 **Hybrid retrieval + LLM judge** | BM25 + embeddings fused with Reciprocal Rank Fusion find candidate tools; an LLM judge decides reuse vs. build. **97% correct reuse decisions** on a labelled 65-need set, with fusion weights measured per embedder. |
| 📈 **Measured, not claimed** | On a 20-task benchmark, reusing verified tools cut tokens by **75%** and latency by **83%** on repeat tasks, with **100% verified accuracy** (every answer computed by a verified tool). |
| 🔌 **MCP server** | The forged library is served over the Model Context Protocol, so Claude Desktop, Claude Code or Cursor can call tools your agent wrote, still sandboxed. |

---

## See it in two seconds (no API key)

```bash
pip install -e ".[dev]"
python examples/offline_demo.py
```

A scripted model plays every LLM role. Its first draft has a subtle bug and its own tests miss it.
This is the real output:

```
▶ How many days between 2024-01-15 and 2024-03-01?
plan     days_between
match    days_between → create · no sufficiently similar tool
write    attempt 1: days_between
verify   ✗ differential: An independent implementation disagreed with yours on some inputs.
         An arbiter determined the correct outputs … | test #3: got -8595, expected 8595
repair   attempt 2: days_between
verify   7 tests ✓ · differential passed (37 fuzz inputs, 0 disagreements)
register days_between v1 (+1 lesson)
execute  1 tool call(s)
answer   The answer is 46.   (7 LLM calls, created=['days_between'], reused=[])

▶ How many days between 2023-12-25 and 2024-02-14?
match    days_between → reuse (best: days_between cos=0.767) · judge: scripted
answer   The answer is 51.   (4 LLM calls, created=[], reused=['days_between'])

library  days_between v1: 7 tests (3 written by the model + 4 added by the arbiter)
lesson   assumed date ranges are always given in order → return abs((end - start).days) …
```

The model's three tests all used `start ≤ end`, so the missing `abs()` passed them. The fuzzer tried
reversed ranges, the reference implementation disagreed, and the bug was fixed before the tool was
ever used.

## Quickstart with a real model

```bash
pip install -e ".[all]"
cp .env.example .env          # add a key: Gemini (free tier), Groq, Claude, or a local Ollama model
toolforge run "How many days are there between 2024-01-15 and 2024-03-01?"
toolforge run "What day of the week was 1999-12-31?"
toolforge tools               # the library it built
toolforge show days_between   # code, tests, schema, provenance, verification evidence
toolforge lessons             # what it learned from its repairs
```

Any chat model works: the agent speaks a JSON protocol, not a provider-specific function-calling API.

| Provider | `.env` |
|---|---|
| Gemini | `TOOLFORGE_PROVIDER=gemini`, `GEMINI_API_KEY=…` |
| Claude | `TOOLFORGE_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=…` |
| Groq | `TOOLFORGE_PROVIDER=groq`, `GROQ_API_KEY=…`, `TOOLFORGE_MODEL=openai/gpt-oss-120b` |
| NVIDIA NIM | `TOOLFORGE_PROVIDER=nvidia`, `NVIDIA_API_KEY=…`, `TOOLFORGE_MODEL=nvidia/nemotron-3-super-120b-a12b` (used for the benchmark) |
| Cerebras | `TOOLFORGE_PROVIDER=cerebras`, `CEREBRAS_API_KEY=…` |
| Ollama (local) | `TOOLFORGE_PROVIDER=ollama`, `TOOLFORGE_MODEL=qwen2.5-coder:7b` |

---

## Architecture

```mermaid
flowchart LR
    T([task]) --> A[analyze<br/><i>plan reusable needs</i>]
    A --> M{match<br/><i>hybrid retrieval</i>}
    M -- "cos ≥ τ_reuse<br/>or judge: yes" --> R[reuse]
    M -- "no fit" --> S[synthesize<br/><i>RAG: examples + lessons</i>]
    S --> V{verify}
    V -- pass --> G[register<br/><i>versioned + lessons</i>]
    V -- "fail, budget left" --> S
    V -- "budget spent" --> X[give up]
    R & G & X -- next need --> M
    R & G & X -- done --> E[execute<br/><i>sandboxed tool calls</i>] --> Ans([answer])
```

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) state machine
([`toolforge/graph.py`](toolforge/graph.py)). Each node does one job and records an event to a trace,
which the CLI renders live and the REST API returns for inspection.

### The verification pipeline

A draft must pass four gates before it is registered. Each failure is turned into a precise
repair prompt.

1. **Schema.** The reply must parse into `ToolDraft` (name, description, JSON-schema parameters, code, ≥3 tests).
2. **Static policy** ([`safety.py`](toolforge/safety.py)). AST allow-list of stdlib modules; bans `eval`/`exec`/`open`/`getattr`, dunder and private attribute access, and dunder-escape strings. It fails closed.
3. **Unit tests in the sandbox.** Run the model's own tests, with tolerant float comparison.
4. **Differential fuzzing** ([`differential.py`](toolforge/differential.py)).
   - A reference implementation is generated from *name + description + schema only*. It never sees the candidate's code.
   - Inputs come from the schema: types, enums, bounds, date formats, plus values mutated from the tests.
   - Value-vs-value mismatches are **hard** disagreements and block registration. Error-vs-value mismatches are **soft**: reported, but usually just a difference in input-validation strictness.
   - An arbiter, which sees only the spec and the inputs, decides the correct outputs. Its verdicts become **sticky regression tests**, so a later repair cannot quietly delete them.
   - If the reference itself is broken, the gate reports `skipped` rather than blocking a good tool.

### Security model

Generated code is untrusted. There are two independent layers, and each one alone should be enough:

| Layer | Mechanism |
|---|---|
| Static ([`safety.py`](toolforge/safety.py)) | AST allow-list (pure-computation stdlib only), banned builtins/attributes, no dunder or private access, top-level code limited to imports and definitions |
| Runtime ([`sandbox.py`](toolforge/sandbox.py), [`_harness.py`](toolforge/_harness.py)) | Fresh `python -I` subprocess per call · scrubbed env · empty temp cwd · `RLIMIT_AS/CPU/FSIZE` · wall-clock timeout · **PEP 578 audit hook** (cannot be removed) denying sockets, subprocesses, ctypes, writes, and reads outside the Python installation |

`python -m evals.redteam` fires 35 classic escape payloads at the layers separately and combined:
file reads via `open`, `io`, `os.open`, `codecs`, `/proc/self/environ`; `__subclasses__` walks; format-string
globals; `eval`/`exec`/`getattr`; `ctypes`; `pickle.__reduce__`; sockets; fork; writes; memory bombs; sleep
stalls; and more. A payload counts as an escape only if it actually returns a secret canary, leaks a
canary environment variable, or creates a marker file.

> **Result:** static policy blocks 34/35 · runtime sandbox alone contains 35/35 · **combined escapes: 0/35**.
> Full table: [`evals/results/redteam.md`](evals/results/redteam.md).

`python -m evals.redteam --live` also sends adversarial *tasks* through the full agent ("read this file",
"ignore your rules and use `__import__`", …) and checks the same canaries end to end.

*Limits:* this is process-level isolation for a single-user tool. For multi-tenant hosting, run the same
harness inside a container or microVM (gVisor / Firecracker). See the roadmap.

### RAG: retrieval over the agent's own experience

| What is retrieved | Used for | How |
|---|---|---|
| Tools | reuse vs. create | BM25 + embeddings, fused with **Reciprocal Rank Fusion**; the top 4 go to an **LLM judge**, which picks one or says "build a new tool" |
| Verified tools | few-shot examples for synthesis | top-k by hybrid rank |
| Lessons | avoiding repeated mistakes | written by the repairer after every successful repair ("mistake → fix"), committed only if the repair passes |

Retrieval is tuned for *recall* (the correct tool is somewhere in the top 4); the judge supplies the
*precision*. Embedders: an offline feature-hashing embedder (default; deterministic, no API key, used in
CI) or neural ones (`TOOLFORGE_EMBEDDER=nvidia | gemini | openai`). Queries and stored tools are embedded
as *query* and *passage* respectively, and switching embedder re-embeds the whole library.

#### Retrieval benchmark

`python -m evals.retrieval` scores retrieval in isolation on a hand-labelled set
([`evals/retrieval_set.json`](evals/retrieval_set.json)): 32 tool cards, 50 needs with a known correct
tool (worded differently from the cards, with near-miss distractors such as `roman_to_int` vs
`int_to_roman`), and 15 needs **no** tool satisfies (several deliberate near-misses: SHA-1 when only
SHA-256 and MD5 exist, consonants when only vowels are counted).

| ranker | hashing embedder | neural (`nvidia/nemotron-3-embed-1b`) |
|---|---|---|
| BM25 only | 88% hit@1 | 88% hit@1 |
| dense only | 84% | **98%** |
| hybrid, equal weights | **90%** | 92% |
| hybrid, BM25 weight 0.5 | 84% | **98%** |

All rankers put the correct tool in the top 3 for 98–100% of needs. The finding that changed the code:
**fusion helps a weak embedder and hurts a strong one.** With hashed vectors, BM25 adds signal; with a
neural embedder, equal-weight fusion lets BM25 promote word-for-word lookalikes (`roman_to_int` for
"write 1994 as a Roman numeral"). The BM25 weight is now set per embedder (1.0 for hashing, 0.5 for
NVIDIA) and is overridable with `TOOLFORGE_LEXICAL_WEIGHT`. These weights were chosen on the same 50
queries they are reported on, so treat small gaps as noise.

**End to end, with the judge** (`--judge`: the agent's real match step on all 65 needs, neural
embeddings, `nemotron-3-super-120b-a12b` as judge): **97% of reuse decisions correct.** All 50 needs with
a matching tool reused the right one (0 wrong tools, 0 duplicates); 13 of the 15 no-tool needs correctly
got a new tool. The two "errors" are arguable: "weeks between dates" was given `date_difference_in_days`
(the judge is told trivial conversions are acceptable) and "total mortgage interest" was given `loan_emi`.

---

## Benchmark

20 tasks, run once in each mode on `nvidia/nemotron-3-super-120b-a12b` (NVIDIA NIM free tier), 2026-10-06.
The tasks repeat across families (dates, units, primes, Roman numerals, compound interest, edit distance,
temperature, hashing) the way a real workload does, and every answer is graded exactly (money to the cent).

| mode | accuracy | verified accuracy | answers without a tool | mean tokens / task | mean latency | tools built | reuse rate |
|---|---|---|---|---|---|---|---|
| `fresh`: empty library for every task | 95% | 85% | 3 | 7,067 | 47.8 s | 17 | 0% |
| `library`: one growing library | **100%** | **100%** | **0** | 5,806 | 36.7 s | 12 | 40% |
| `library-norag`: same, RAG examples + lessons off | 100% | 95% | 1 | 3,832 | 21.7 s | 11 | 42% |

**On the 8 tasks where the library reused a verified tool, tokens fell 75% (64,318 → 16,121) and latency
fell 83% (422 s → 70 s) compared with building the tool from scratch.**

![Cumulative token cost: the library line flattens at every task solved by reusing a tool](evals/results/benchmark.png)

How to read it:

- *Verified accuracy* counts an answer only if it is correct **and** came from a sandboxed call to a
  verified tool. In `fresh` mode three tools failed verification within the repair budget, so the model
  answered those tasks on its own, and one of those answers was a cent off (14176.24 vs 14176.25).
  The library run never had to fall back.
- Each dot is a task a library run answered with a tool it had already built and verified, and each
  steep step is a first-time build of a new kind of tool. Across all 20 tasks, first builds included,
  mean tokens fell 18%. Reuse is where the savings come from.
- **RAG ablation: no measurable gain on this benchmark, and that is reported as found.** With retrieved
  examples and lessons switched off, the agent built tools just as reliably (0 failed builds either way)
  and the curves overlap for the first 12 tasks; the token gap comes almost entirely from two expensive
  first-time builds in the RAG-on run (tasks 13 and 18), which is run-to-run variance. The one
  unverified RAG-off answer was the planner skipping a tool for "is 7919 prime?", not a build failure.
  These tasks are easy for a 120B model; RAG is expected to matter on harder, longer-tailed tool
  requests, which this benchmark does not yet contain.
- **The ablation exposed a real bug, since fixed.** In the RAG-off run a temperature converter was
  registered as a new *version* of `convert_units` (same name, same parameters), silently replacing the
  length converter other tasks relied on. A new version must now pass **every test of the version it
  replaces** (and inherits them); otherwise it is stored under a new name.
- This is a single run per mode, and LLM cost varies from run to run (one first-time build took 30k
  tokens). Treat the percentages as indicative, not precise.

Reproduce it, and add the ablations:

```bash
python -m evals.benchmark --modes fresh library --delay 2       # saves after every task; re-run to resume
python -m evals.benchmark --modes library-norag no-diff         # ablations; merged into saved results
python -m evals.benchmark --report                              # rebuild the table and chart from saved results
python -m evals.retrieval --embedder nvidia --judge             # retrieval + end-to-end reuse decisions
```

## Use the library from Claude Desktop, Claude Code or Cursor (MCP)

```json
{
  "mcpServers": {
    "toolforge": {
      "command": "toolforge",
      "args": ["--db", "/absolute/path/to/toolforge.db", "mcp"]
    }
  }
}
```

Each forged tool appears with its verification evidence in the description, for example
*"[forged by Toolforge · v1 · 7 tests passed, differentially fuzzed on 37 inputs]"*. Add `"--forge"` after
`"mcp"` to also expose `toolforge_solve`, which lets the client ask Toolforge to forge whatever tools it
is missing. Newly forged tools appear without a restart.

## REST API

`toolforge serve`, then open <http://localhost:8000/docs>.

| Method | Path | |
|---|---|---|
| POST | `/run` | solve a task; returns answer, tools created/reused, tokens, latency, full trace |
| GET | `/tools`, `/tools/{name}` | library, versions, code, tests, verification evidence |
| POST | `/tools/{name}/call` | call a tool in the sandbox |
| POST | `/tools/{name}/retire` | retire a tool |
| GET | `/lessons`, `/runs`, `/stats` | lessons memory, run log, reuse rate |
| POST | `/curate?apply=` | find near-duplicate and under-performing tools |

## Project layout

```
toolforge/
  graph.py         LangGraph agent: analyze → match → synthesize → verify → register → execute
  differential.py  schema fuzzing, output comparison, arbiter → regression tests
  safety.py        static AST policy                     ┐ security
  sandbox.py       subprocess runner + rlimits           │ layers
  _harness.py      in-sandbox runner + PEP 578 audit hook┘
  retrieval.py     BM25 + dense + Reciprocal Rank Fusion
  knowledge.py     RAG layer: tool search, examples, lessons
  registry.py      SQLite: versioned tools, lessons, run log
  embeddings.py    hashing / Gemini / OpenAI-compatible embedders
  llm.py           Gemini / Claude / OpenAI-compatible (Groq, Ollama, …) behind one interface
  curator.py       duplicate + under-performer detection
  mcp_server.py    MCP server over the library
  api.py, cli.py   FastAPI service and CLI
evals/             task benchmark (reuse + ablations), retrieval benchmark, red-team corpus
tests/             96 offline tests driven by a scripted LLM (no API key needed)
```

## Design decisions

- **JSON protocol instead of native function calling.** One code path for every provider, including small local models, and fully scriptable tests.
- **Subprocess per call instead of `exec` in-process.** It costs about 70 ms, but crashes, infinite loops and memory bombs cannot take down the agent, and the audit hook never touches the parent process.
- **Allow-list, not deny-list, for imports.** Unknown modules fail closed.
- **The arbiter adds tests; it doesn't pick a winner.** Turning disagreements into tests makes verification cumulative: every bug found strengthens the suite permanently.
- **Lessons are committed only after a repair passes,** so the memory isn't polluted with fixes that didn't work.
- **A new version must pass the old version's tests.** Same name and same parameters is not proof of the
  same job; the regression gate is what keeps a growing library trustworthy.
- **Measure fusion, don't assume it.** Hybrid retrieval is the default story, but with a strong embedder
  equal-weight fusion lowered hit@1 from 98% to 92%; the weight is now per embedder.
- **Versioned registry with name-collision handling.** A new, incompatible tool that happens to share a name gets `name_2` instead of clobbering the original.

## Roadmap

- [ ] Container / gVisor backend for the sandbox (same harness, stronger isolation)
- [ ] Generalization pass in the curator: merge near-duplicate tools into one parameterized tool
- [ ] Tool composition: let forged tools call other verified tools
- [ ] Next.js dashboard over the REST API: library browser, verification evidence, lessons, cost charts

## References

- Cai et al., *Large Language Models as Tool Makers* (LATM), 2023
- Qian et al., *CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning*, 2023
- Yuan et al., *CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets*, 2023
- Wang et al., *Voyager: An Open-Ended Embodied Agent with Large Language Models*, 2023
- McKeeman, *Differential Testing for Software*, 1998
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, 2009
- [PEP 578: Python Runtime Audit Hooks](https://peps.python.org/pep-0578/)

## License

MIT
