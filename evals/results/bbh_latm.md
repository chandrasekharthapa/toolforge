## BIG-Bench Hard (LATM tasks): big = `nemotron-3-super-120b-a12b`, small = `llama-3.2-11b-vision-instruct` · 15 items per task per seed · seeds [0, 1, 2] · 2026-10-08

Accuracy, mean ± sample SD across seeds (each seed is a different random sample of items).

| task | big, direct | big + Toolforge | small, direct | small + big's tools |
|---|---|---|---|---|
| word sorting | 98% ± 4% | 98% ± 4% | 38% ± 20% | 87% ± 7% |
| dyck languages | 84% ± 4% | 84% ± 8% | 0% ± 0% | 11% ± 4% |
| logical deduction five objects | 100% ± 0% | 100% ± 0% | 53% ± 18% | 56% ± 21% |
| tracking shuffled objects five objects | 100% ± 0% | 100% ± 0% | 53% ± 18% | 64% ± 8% |
| **macro average** | **96% ± 1%** (3 seeds) | **96% ± 3%** (3 seeds) | **36% ± 11%** (3 seeds) | **54% ± 4%** (3 seeds) |

Tokens per item, and how often a tool was used:

| task | big, direct: tokens | big + Toolforge: tokens | small, direct: tokens | small + big's tools: tokens | big + Toolforge: used a tool | correct with a tool | small + big's tools: used a tool | correct with a tool |
|---|---|---|---|---|---|---|---|---|
| word sorting | 898 | 2,086 | 144 | 1,663 (1,600 small + 63 big) | 38% | 17/17 | 98% | 39/44 |
| dyck languages | 3,586 | 6,990 | 299 | 5,096 (4,808 small + 288 big) | 40% | 16/18 | 100% | 5/45 |
| logical deduction five objects | 921 | 2,657 | 514 | 18,184 (7,706 small + 10,477 big) | 7% | 3/3 | 98% | 24/44 |
| tracking shuffled objects five objects | 729 | 1,585 | 486 | 9,423 (5,002 small + 4,421 big) | 2% | 1/1 | 98% | 29/44 |

One-off tool making for `latm` (big model, 3 held-out demos, once per task and seed):

| task | seeds with a verified tool | mean tokens to build | tools built during use | tools repaired after failing in use |
|---|---|---|---|---|
| word sorting | 2/3 | 9,978 | 1 | 0 |
| dyck languages | 2/3 | 16,741 | 1 | 1 |
| logical deduction five objects | 1/3 | 37,152 | 5 | 12 |
| tracking shuffled objects five objects | 3/3 | 8,593 | 0 | 6 |
