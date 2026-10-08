## BIG-Bench Hard (LATM tasks): `nvidia:nvidia/nemotron-3-super-120b-a12b` · 15 items per task per seed · seeds [0, 1, 2] · 2026-10-06

Accuracy, mean ± sample SD across seeds (each seed is a different random sample of items).

| task | direct | Toolforge |
|---|---|---|
| word sorting | 98% ± 4% | 98% ± 4% |
| dyck languages | 82% ± 4% | 84% ± 8% |
| logical deduction five objects | 100% ± 0% | 100% ± 0% |
| tracking shuffled objects five objects | 100% ± 0% | 100% ± 0% |
| **macro average** | **95% ± 2%** (3 seeds) | **96% ± 3%** (3 seeds) |

Tokens per item, and how often a tool was used:

| task | direct: tokens | Toolforge: tokens | Toolforge: used a tool | correct with a tool |
|---|---|---|---|---|
| word sorting | 898 | 2,086 | 38% | 17/17 |
| dyck languages | 2,675 | 6,990 | 40% | 16/18 |
| logical deduction five objects | 921 | 2,657 | 7% | 3/3 |
| tracking shuffled objects five objects | 729 | 1,585 | 2% | 1/1 |
