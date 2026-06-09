# Cheapest manifold atom — first-projection width sweep

rank=3, steps=3000, lr=0.002, 85/15 train/test.

| concept | N | best test FVU (W) | rank-3 PCA | cheapest W ≤+10% | params@cheapest |
|---|---|---|---|---|---|
| temperature |  | 0.033 (W=32) | 0.123 | **W=32** | 268K |
| years |  | 0.828 (W=256) | 0.810 | **W=16** | 136K |
| days |  | 0.424 (W=256) | 0.560 | **W=128** | 1074K |
| colors |  | 0.188 (W=64) | 0.282 | **W=64** | 534K |
