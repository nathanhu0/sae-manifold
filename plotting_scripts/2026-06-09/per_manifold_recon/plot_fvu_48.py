"""8x6 strip plot of FVU across all 48 instances (local, pure matplotlib).

One row per type; within a row the 6 variants are dots placed at their FVU (log x).
Best-single-atom FVU (circles) and full-model FVU (triangles) shown separately; the
0.05 cutoff is the vertical line. Per-row "k/6 captured" counts for each metric. Types
ordered by single-atom dedication (most-captured at top).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent
data = json.load(open(D / "fvu_48.json"))
types = list(dict.fromkeys(r["type"] for r in data))
# order types by # captured by a single atom (desc), then by median single FVU
types.sort(key=lambda t: (-sum(r["single"] < 0.05 for r in data if r["type"] == t),
                          np.median([r["single"] for r in data if r["type"] == t])))

fig, ax = plt.subplots(figsize=(11, 6.5))
for i, t in enumerate(types):
    y = len(types) - 1 - i
    sg = sorted(r["single"] for r in data if r["type"] == t)
    fl = sorted(r["full"] for r in data if r["type"] == t)
    ki = next(r["ki"] for r in data if r["type"] == t)
    ax.scatter(np.clip(sg, 1e-4, 5), [y + 0.15] * len(sg), c="C0", s=55, zorder=3,
               label="best single atom" if i == 0 else None, edgecolor="white", linewidth=0.5)
    ax.scatter(np.clip(fl, 1e-4, 5), [y - 0.15] * len(fl), c="C1", marker="^", s=55, zorder=3,
               label="full model (union)" if i == 0 else None, edgecolor="white", linewidth=0.5)
    ax.text(6.5, y + 0.15, f"{sum(s < 0.05 for s in sg)}/6", va="center", fontsize=8, color="C0")
    ax.text(6.5, y - 0.15, f"{sum(f < 0.05 for f in fl)}/6", va="center", fontsize=8, color="C1")
    ax.axhline(y - 0.5, color="0.9", lw=0.6, zorder=0)

ax.axvline(0.05, color="red", ls="--", lw=1.3, label="cutoff = 0.05", zorder=2)
ax.set_xscale("log")
ax.set_xlim(8e-4, 9)
ax.set_yticks(range(len(types)))
ax.set_yticklabels([f"{t}  (k{next(r['ki'] for r in data if r['type'] == t)})"
                    for t in reversed(types)], fontsize=9)
ax.set_xlabel("fraction of variance unexplained  (FVU, log)")
ax.set_title("Per-instance FVU across 8 types x 6 variants (STE hard gate, lam=0.02 pre=1e-3)\n"
             "circles=best single atom, triangles=full model; right column = # of 6 captured (<0.05)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig(D / "fvu_48.png", dpi=145)
print("saved", D / "fvu_48.png")
