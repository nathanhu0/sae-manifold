"""Render per-manifold reconstruction from per_manifold.npz (pure numpy/matplotlib, local).

8 rows (types) x 3 cols: target | best single atom | full model, in V_i coords, colored by
the manifold's intrinsic angle. Single-atom FVU and full-model FVU in the column titles.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent
npz = np.load(D / "per_manifold.npz")
meta = json.load(open(D / "per_manifold_meta.json"))
TYPES = list(meta)

fig, axes = plt.subplots(len(TYPES), 3, figsize=(10, 3.0 * len(TYPES)))
COLS = ["target (ground truth)", "best single atom", "full model (union)"]

for r, name in enumerate(TYPES):
    tgt = npz[f"{name}__target"]
    th = npz[f"{name}__theta"][:, 0]
    series = [tgt, npz[f"{name}__single"], npz[f"{name}__full"]]
    ki = meta[name]["ki"]
    # shared axis limits from the target so distortion is visible
    if ki >= 2:
        x0, x1 = tgt[:, 0], tgt[:, 1]
    else:
        x0, x1 = tgt[:, 0], np.zeros_like(tgt[:, 0])
    xlim = (x0.min() - .1, x0.max() + .1)
    ylim = (x1.min() - .2, x1.max() + .2) if ki >= 2 else (-1, 1)

    for c, arr in enumerate(series):
        ax = axes[r, c]
        if ki >= 2:
            xs, ys = arr[:, 0], arr[:, 1]
        else:
            xs, ys = arr[:, 0], np.zeros_like(arr[:, 0])
        ax.scatter(xs, ys, c=th, cmap="hsv", s=6, alpha=0.7, linewidths=0)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"{name}\nd{meta[name]['di']} k{ki}", fontsize=9)
        if r == 0:
            ax.set_title(COLS[c], fontsize=10)
    axes[r, 1].set_title(f"best single atom (rank {meta[name]['best_rank']})  "
                         f"FVU={meta[name]['single_fvu']:.3f}", fontsize=9)
    axes[r, 2].set_title(f"full model (union)  FVU={meta[name]['full_fvu']:.3f}", fontsize=9)

fig.suptitle("Per-manifold reconstruction: target vs best single atom vs full model\n"
             "(STE hard gate, lam=0.02 lam_preact=1e-3; first 2 V_i coords, colored by angle)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(D / "per_manifold_recon.png", dpi=140)
print("saved", D / "per_manifold_recon.png")
