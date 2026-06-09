"""Arch sweep figures: depth/width effects + the minimal-dim flat-vs-structured
wall. All runs are at latent_dim = k* (exact/minimal intrinsic dim)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
df = pd.read_parquet("/juice2/u/nathu/sae-manifold/results/manifold_ae/sweep_arch.parquet")
MAN = ["swiss_roll", "circle", "sphere", "sphere10", "torus", "torus5"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: FVU vs depth (width=256, min over lr)
sub = df[df.width == 256].groupby(["name", "latent_mode", "depth"]).fvu_clean.min().reset_index()
for n in MAN:
    for mode, ls in [("flat", "-"), ("structured", "--")]:
        s = sub[(sub.name == n) & (sub.latent_mode == mode)].sort_values("depth")
        if len(s):
            axes[0].plot(s.depth, s.fvu_clean, ls, marker="o",
                         label=f"{n}·{mode}" if mode == "flat" else None,
                         color=f"C{MAN.index(n)}")
axes[0].set_yscale("log"); axes[0].set_xlabel("depth (hidden layers)")
axes[0].set_ylabel("FVU vs clean (log)")
axes[0].set_title("Depth: 3-4 is the sweet spot; depth 6 destabilizes\n(width 256, best lr; solid=flat dashed=structured)")
axes[0].legend(fontsize=7, ncol=2)

# Panel B: FVU vs width (depth=4, min over lr)
sub = df[df.depth == 4].groupby(["name", "latent_mode", "width"]).fvu_clean.min().reset_index()
for n in MAN:
    for mode, ls in [("flat", "-"), ("structured", "--")]:
        s = sub[(sub.name == n) & (sub.latent_mode == mode)].sort_values("width")
        if len(s):
            axes[1].plot(s.width, s.fvu_clean, ls, marker="o", color=f"C{MAN.index(n)}",
                         label=f"{n}·{mode}" if mode == "flat" else None)
axes[1].set_yscale("log"); axes[1].set_xlabel("width"); axes[1].set_xscale("log", base=2)
axes[1].set_ylabel("FVU vs clean (log)")
axes[1].set_title("Width: knee at 128-256; 512 helps only hard cases")
axes[1].legend(fontsize=7, ncol=2)

# Panel C: best achievable FVU at minimal dim, flat vs structured
best = df.groupby(["name", "latent_mode"]).fvu_clean.min().reset_index()
x = np.arange(len(MAN)); w = 0.38
for j, mode in enumerate(["flat", "structured"]):
    vals = [best[(best.name == n) & (best.latent_mode == mode)].fvu_clean.min() for n in MAN]
    axes[2].bar(x + (j - 0.5) * w, vals, w, label=mode, color=["#d6604d", "#4393c3"][j])
axes[2].set_yscale("log"); axes[2].set_xticks(x); axes[2].set_xticklabels(MAN, rotation=40, ha="right")
axes[2].set_ylabel("best FVU over ALL lr/width/depth (log)")
axes[2].set_title("Minimal-dim wall: even best-tuned flat can't fit the tori\n(torus flat floors at 0.157; structured 0.002)")
axes[2].legend()

fig.suptitle("Architecture sweep (latent_dim = k*, the minimal/intrinsic dim): how deep/expressive, and where flat hits a topological wall", y=1.04)
fig.tight_layout()
fig.savefig(OUT / "fig9_arch.png", dpi=130, bbox_inches="tight")
print("wrote fig9")
