"""CORRECTED flat-vs-structured comparison (audit fix).

The Milestone-1 'flat@k vs structured' was unfair: a structured S^n latent uses
n+1 coordinates, but flat@k used only k=intrinsic_dim. The fair control is flat
at the SAME coordinate count = the manifold's minimal seam-free EMBEDDING dim
(circle 2, sphere 3, torus 4). At equal width the topology constraint adds no
fit benefit (ties on circle/sphere, loses on torus)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
df = pd.read_parquet("/juice2/u/nathu/sae-manifold/results/manifold_ae/sweep.parquet")
df = df[df.regime == "embedded_noisy"]

EMBED = {"circle": 2, "sphere": 3, "torus": 4}      # minimal seam-free dim
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(EMBED)); w = 0.26
bars = {"flat@intrinsic": [], "flat@embedding": [], "structured": []}
errs = {"flat@intrinsic": [], "flat@embedding": [], "structured": []}
for n, emb in EMBED.items():
    ki = df[df.name == n].intrinsic_dim.iloc[0]
    fi = df[(df.name == n) & (df.latent_mode == "flat") & (df.latent_dim == ki)].fvu_clean
    fe = df[(df.name == n) & (df.latent_mode == "flat") & (df.latent_dim == emb)].fvu_clean
    st = df[(df.name == n) & (df.latent_mode.isin(["structured", "matched"]))].fvu_clean
    for key, vals in [("flat@intrinsic", fi), ("flat@embedding", fe), ("structured", st)]:
        bars[key].append(vals.mean()); errs[key].append(vals.std())
colors = {"flat@intrinsic": "#bbbbbb", "flat@embedding": "#d6604d", "structured": "#4393c3"}
for j, key in enumerate(["flat@intrinsic", "flat@embedding", "structured"]):
    ax.bar(x + (j - 1) * w, bars[key], w, yerr=errs[key], capsize=3,
           label=key, color=colors[key])
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(embed dim {e})" for n, e in EMBED.items()])
ax.set_ylabel("FVU vs clean manifold (log), 3 seeds")
ax.set_title("Corrected comparison: at equal latent width, flat ties structured\n"
             "(circle/sphere) or beats it (torus). Topology constraint adds no fit benefit.")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig10_fair_comparison.png", dpi=130, bbox_inches="tight")
print("wrote fig10")
