## Distilling the reuse judge: `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M parameters)

Test: 65 hand-labelled needs over 32 tools that never appear in training; same candidate lists for all.

| judge | decision accuracy (human labels) | agrees with teacher | cost per decision | latency |
|---|---|---|---|---|
| teacher LLM judge | 98% | — | 873 tokens | 2,495 ms |
| cross-encoder, zero-shot | 78% | 80% | 0 tokens | 6.76 ms (cuda) |
| **cross-encoder, fine-tuned** | **80%** | 82% | **0 tokens** | **6.76 ms** (cuda), 63.67 ms CPU |

Fine-tuned: 44/50 needs with a tool reused the right one, 8/15 needs without one correctly built a new tool. Threshold 0.1231 chosen on validation tools (unseen in training). Trained 15.9 s, best epoch 1.
