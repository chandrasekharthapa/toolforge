## Benchmark: 20 tasks · `nvidia:nvidia/nemotron-3-super-120b-a12b` · 2026-10-06

| mode | accuracy | verified accuracy | answers without a tool | mean tokens / task | mean latency (s) | tools built | reuse rate |
|---|---|---|---|---|---|---|---|
| fresh | 0.95 | 0.85 | 3 | 7067 | 47.77 | 17 | 0.0 |
| library | 1.0 | 1.0 | 0 | 5806 | 36.72 | 12 | 0.4 |
| library-norag | 1.0 | 0.95 | 1 | 3832 | 21.66 | 11 | 0.421 |

*Verified accuracy* counts an answer only if it is correct **and** was computed by a sandboxed call to a verified tool, not by the model on its own.

**On the 8 tasks where the library reused a verified tool, tokens fell 75% (64,318 → 16,121) and latency fell 83% (422.2s → 70.4s) versus building from scratch.**

**Ablation, RAG (retrieved examples + lessons) off:** verified accuracy 95% vs 100% with it on; failed tool builds 0 vs 0; mean tokens 3,832 vs 5,806.

Across all 20 tasks (including first-time builds of new tools) mean tokens per task fell 18%. Single run; LLM cost varies run to run.

![cumulative tokens](benchmark.png)
