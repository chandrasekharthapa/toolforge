## BIG-Bench Hard (LATM tasks): big = `nemotron-3-super-120b-a12b`, small = `llama-3.2-11b-vision-instruct` · 15 items per task per seed · seeds [4, 5, 6] · 2026-10-10

Accuracy, mean ± sample SD across seeds (each seed is a different random sample of items).

| task | big, direct | small, direct | small + big's tools |
|---|---|---|---|
| word sorting | 98% ± 4% | 51% ± 20% | 98% ± 4% |
| dyck languages | 76% ± 4% | 0% ± 0% | 95% ± 4% |
| logical deduction five objects | 100% ± 0% | 53% ± 18% | 29% ± 4% |
| tracking shuffled objects five objects | 100% ± 0% | 56% ± 4% | 76% ± 27% |
| **macro average** | **93% ± 2%** (3 seeds) | **40% ± 10%** (3 seeds) | **74% ± 6%** (3 seeds) |

Tokens per item, and how often a tool was used:

| task | big, direct: tokens | small, direct: tokens | small + big's tools: tokens | small + big's tools: used a tool | correct with a tool |
|---|---|---|---|---|---|
| word sorting | 674 | 132 | 1,682 (1,682 small + 0 big) | 98% | 44/44 |
| dyck languages | 5,826 | 369 | 5,905 (5,905 small + 0 big) | 98% | 41/42 |
| logical deduction five objects | 1,118 | 482 | 8,678 (7,387 small + 1,292 big) | 98% | 13/44 |
| tracking shuffled objects five objects | 667 | 478 | 12,019 (6,544 small + 5,475 big) | 100% | 34/45 |

One-off tool making for `latm` (big model, 3 held-out demos, once per task and seed):

| task | seeds with a verified tool | mean tokens to build | tools built during use | tools repaired after failing in use |
|---|---|---|---|---|
| word sorting | 3/3 | 9,203 | 0 | 0 |
| dyck languages | 3/3 | 12,919 | 0 | 0 |
| logical deduction five objects | 0/3 | 91,278 | 2 | 0 |
| tracking shuffled objects five objects | 3/3 | 115,876 | 4 | 4 |
