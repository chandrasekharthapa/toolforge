## Retrieval benchmark: 32 tools, 50 needs with a correct tool, 15 with none
Embedder: `nvidia:nvidia/nemotron-3-embed-1b`

| ranker | hit@1 | hit@3 | MRR |
|---|---|---|---|
| BM25 (lexical) | 88% | 100% | 0.930 |
| dense (embedding cosine) | 98% | 100% | 0.990 |
| **hybrid: BM25 + dense, RRF** | 92% | 100% | 0.960 |

**Reuse gate, shipped thresholds** (auto-reuse ≥ 0.8, ask the judge ≥ 0.25, otherwise create)

| | auto-reused | sent to judge | created |
|---|---|---|---|
| needs with a correct tool (50) | 0 right, 0 wrong | 50 (correct tool shown to it: 50) | 0 (duplicate) |
| needs with no tool (15) | **0** (wrong) | 13 | 2 |

**Reuse gate, thresholds suggested by this set** (auto-reuse ≥ 0.71, ask the judge ≥ 0.35, otherwise create)

| | auto-reused | sent to judge | created |
|---|---|---|---|
| needs with a correct tool (50) | 5 right, 0 wrong | 44 (correct tool shown to it: 44) | 1 (duplicate) |
| needs with no tool (15) | **0** (wrong) | 6 | 9 |

Suggested: `TOOLFORGE_REUSE_THRESHOLD=0.71`, `TOOLFORGE_CONSIDER_THRESHOLD=0.35`. Anything sent to the judge is still checked by an LLM before reuse; only auto-reuse skips that check.

**End to end, with the LLM judge** (`nvidia:nvidia/nemotron-3-super-120b-a12b`, the agent's real match step, shipped thresholds): decision accuracy **97%**

| | reused the right tool | reused a wrong tool | built a new tool |
|---|---|---|---|
| needs with a correct tool (50) | **50** | 0 | 0 (duplicate) |
| needs with no tool (15) | — | **2** | 13 |
