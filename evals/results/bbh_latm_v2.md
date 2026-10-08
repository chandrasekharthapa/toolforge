## BIG-Bench Hard (LATM tasks): big = `nemotron-3-super-120b-a12b`, small = `llama-3.2-11b-vision-instruct` · 3 items per task per seed · seeds [3] · 2026-10-08

Accuracy, mean ± sample SD across seeds (each seed is a different random sample of items).

| task | small + big's tools |
|---|---|
| dyck languages | 33% ± 0% |
| logical deduction five objects | 67% ± 0% |
| **macro average** | **50% ± 0%** (1 seeds) |

Tokens per item, and how often a tool was used:

| task | small + big's tools: tokens | small + big's tools: used a tool | correct with a tool |
|---|---|---|---|
| dyck languages | 4,093 (4,093 small + 0 big) | 100% | 1/3 |
| logical deduction five objects | 15,425 (6,559 small + 8,866 big) | 100% | 2/3 |

One-off tool making for `latm` (big model, 3 held-out demos, once per task and seed):

| task | seeds with a verified tool | mean tokens to build | tools built during use | tools repaired after failing in use |
|---|---|---|---|---|
| dyck languages | 1/1 | 13,700 | 0 | 0 |
| logical deduction five objects | 1/1 | 70,044 | 0 | 1 |
