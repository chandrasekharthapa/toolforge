## Retrieval benchmark: 32 tools, 50 needs with a correct tool, 15 with none
Embedder: `hashing:1024`

| ranker | hit@1 | hit@3 | MRR |
|---|---|---|---|
| BM25 (lexical) | 88% | 100% | 0.930 |
| dense (embedding cosine) | 84% | 98% | 0.911 |
| **hybrid: BM25 + dense, RRF** (BM25 weight 1.0) | 90% | 98% | 0.943 |

Fusion sweep: BM25 weight in RRF (dense weight 1.0). Chosen on this same set, so treat a small difference as noise.

| BM25 weight | hit@1 | hit@3 | MRR |
|---|---|---|---|
| 1.0 | 90% | 98% | 0.943 |
| 0.5 | 84% | 98% | 0.913 |
| 0.25 | 84% | 98% | 0.912 |
| 0.1 | 84% | 98% | 0.911 |

**Reuse gate, shipped thresholds** (auto-reuse ≥ 0.86, ask the judge ≥ 0.22, otherwise create)

| | auto-reused | sent to judge | created |
|---|---|---|---|
| needs with a correct tool (50) | 0 right, 0 wrong | 48 (correct tool shown to it: 45) | 2 (duplicate) |
| needs with no tool (15) | **0** (wrong) | 13 | 2 |

**Reuse gate, thresholds suggested by this set** (auto-reuse ≥ 0.63, ask the judge ≥ 0.15, otherwise create)

| | auto-reused | sent to judge | created |
|---|---|---|---|
| needs with a correct tool (50) | 0 right, 0 wrong | 50 (correct tool shown to it: 49) | 0 (duplicate) |
| needs with no tool (15) | **0** (wrong) | 14 | 1 |

Suggested: `TOOLFORGE_REUSE_THRESHOLD=0.63`, `TOOLFORGE_CONSIDER_THRESHOLD=0.15`. Anything sent to the judge is still checked by an LLM before reuse; only auto-reuse skips that check.
