## BIG-Bench Hard (LATM tasks): big = `nemotron-3-super-120b-a12b`, small = `llama-3.2-11b-vision-instruct` · 15 items per task per seed · seeds [3] · 2026-10-08

Accuracy, mean ± sample SD across seeds (each seed is a different random sample of items).

| task | small, direct | small + big's tools |
|---|---|---|
| word sorting | 27% ± 0% | 100% ± 0% |
| dyck languages | 0% ± 0% | 100% ± 0% |
| logical deduction five objects | 67% ± 0% | 53% ± 0% |
| tracking shuffled objects five objects | 47% ± 0% | 100% ± 0% |
| **macro average** | **35% ± 0%** (1 seeds) | **88% ± 0%** (1 seeds) |

Tokens per item, and how often a tool was used:

| task | small, direct: tokens | small + big's tools: tokens | small + big's tools: used a tool | correct with a tool |
|---|---|---|---|---|
| word sorting | 149 | 1,868 (1,868 small + 0 big) | 100% | 15/15 |
| dyck languages | 291 | 6,161 (4,826 small + 1,336 big) | 100% | 14/14 |
| logical deduction five objects | 464 | 7,335 (7,094 small + 241 big) | 100% | 8/15 |
| tracking shuffled objects five objects | 482 | 2,408 (1,982 small + 426 big) | 100% | 15/15 |

One-off tool making for `latm` (big model, 3 held-out demos, once per task and seed):

| task | seeds with a verified tool | mean tokens to build | tools built during use | tools repaired after failing in use |
|---|---|---|---|---|
| word sorting | 1/1 | 11,704 | 0 | 0 |
| dyck languages | 1/1 | 14,230 | 0 | 1 |
| logical deduction five objects | 0/1 | 81,098 | 1 | 0 |
| tracking shuffled objects five objects | 1/1 | 47,693 | 0 | 0 |
