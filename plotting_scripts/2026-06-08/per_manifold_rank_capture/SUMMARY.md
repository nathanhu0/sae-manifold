# Per-manifold rank-capture (best trained atom per rank, on L0=4 mixtures)

Capture-FVU of the single best already-trained atom of each rank, probing
each manifold within real mixtures at training sparsity (L0=4). The atom is
chosen across all budget checkpoints (R3..R16). rank=k_i is seamless capture.

| type | d_i | k_i | rank 1 | rank 2 | rank 3 | rank 4 |
|---|---|---|---|---|---|---|
| circle | 1 | 2 | 0.006 | 0.005* | 0.114 | 0.075 |
| sphere | 2 | 3 | 0.371 | 0.007 | 0.004* | 0.142 |
| torus | 2 | 4 | 0.106 | 0.040 | 0.145 | 0.005* |
| mobius | 2 | 3 | 0.581 | 0.005 | 0.005* | 0.849 |
| swiss_roll | 2 | 3 | 0.029 | 0.005 | 0.386* | 0.943 |
| helix | 1 | 3 | 0.004 | 0.470 | 0.463* | 0.544 |
| flat_disk | 2 | 2 | 0.262 | 0.006* | 0.049 | 0.988 |
| segment | 1 | 1 | 0.491* | 0.704 | 0.505 | 0.632 |

(* = rank equals embedding dim k_i, the seamless-capture rank)