# Aggregate: best single-atom capture achievable across the budget sweep

Per manifold, the lowest best-atom FVU over all budgets, the rank of that
atom, and the budget where it occurs. 'rank @ best' vs d_i/k_i shows whether
the cleanest single-atom capture lands near the intrinsic dim (de-shatter)
or the embedding dim.

| manifold | d_i | k_i | best FVU | rank @ best | budget @ best |
|---|---|---|---|---|---|
| circle | 1 | 2 | 0.005 | 1 | R6 |
| sphere | 2 | 3 | 0.004 | 3 | R16 |
| torus | 2 | 4 | 0.005 | 4 | R16 |
| mobius | 2 | 3 | 0.005 | 2 | R16 |
| swiss_roll | 2 | 3 | 0.005 | 2 | R9 |
| helix | 1 | 3 | 0.007 | 1 | R6 |
| flat_disk | 2 | 2 | 0.017 | 2 | R7 |
| segment | 1 | 1 | 0.565 | 3 | R9 |