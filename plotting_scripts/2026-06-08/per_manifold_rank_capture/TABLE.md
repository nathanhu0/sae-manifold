# Best atom per manifold — trained mixture SAE (R=6)

For one trained mixture checkpoint: the single best atom capturing each
manifold (min variance-unexplained of m_i, probed on L0=4 mixtures).

| manifold | d_i | k_i | best-atom rank | FVU (var unexpl.) |
|---|---|---|---|---|
| circle | 1 | 2 | 1 | 0.006 |
| sphere | 2 | 3 | 2 | 0.008 |
| torus | 2 | 4 | 1 | 0.106 |
| mobius | 2 | 3 | 1 | 0.597 |
| swiss_roll | 2 | 3 | 2 | 0.066 |
| helix | 1 | 3 | 1 | 0.007 |
| flat_disk | 2 | 2 | 2 | 0.035 |
| segment | 1 | 1 | 2 | 0.836 |