"""Figures + summary for the toy manifold-AE sweep.

Reads results/manifold_ae/sweep.parquet and arrays/*.npz, writes PNGs + a
markdown summary alongside this script.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
REPO = Path("/juice2/u/nathu/sae-manifold")
RESULTS = REPO / "results" / "manifold_ae"
ARRAYS = RESULTS / "arrays"

EASY = ["helix", "swiss_roll", "s_curve", "wavy_sheet"]
HARD = ["circle", "sphere", "torus"]
ORDER = EASY + HARD

df = pd.read_parquet(RESULTS / "sweep.parquet")
# canonical comparison rows: flat at true intrinsic dim, and matched
flat_k = df[(df.latent_mode == "flat") & (df.latent_dim == df.intrinsic_dim)].copy()
flat_k["mode"] = "flat@k"
matched = df[df.latent_mode == "matched"].copy()
matched["mode"] = "matched"
canon = pd.concat([flat_k, matched])


def _agg(frame, value, regime):
    sub = frame[frame.regime == regime]
    g = sub.groupby(["name", "mode"])[value].agg(["mean", "std"]).reset_index()
    return g


# --- Fig 1: FVU_clean, flat@k vs matched, per regime ----------------------- #
regimes = ["native", "embedded", "embedded_noisy"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, regime in zip(axes, regimes):
    g = _agg(canon, "fvu_clean", regime)
    x = np.arange(len(ORDER))
    w = 0.38
    for j, mode in enumerate(["flat@k", "matched"]):
        means = [g[(g.name == n) & (g["mode"] == mode)]["mean"].values for n in ORDER]
        stds = [g[(g.name == n) & (g["mode"] == mode)]["std"].values for n in ORDER]
        means = [float(m[0]) if len(m) else np.nan for m in means]
        stds = [float(s[0]) if len(s) else 0.0 for s in stds]
        ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=2,
               label=mode, color=["#d6604d", "#4393c3"][j])
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=40, ha="right")
    ax.axvline(len(EASY) - 0.5, ls="--", c="gray", lw=1)
    ax.set_title(f"regime: {regime}")
    ax.set_yscale("log")
    if regime == "native":
        ax.set_ylabel("FVU vs clean manifold (log)")
    ax.legend()
fig.suptitle("Reconstruction error: flat R^k baseline vs topology-matched latent "
             "(left of dashed = contractible, right = closed)", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_fvu_flat_vs_matched.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --- Fig 2: AE vs PCA ratio (embedded_noisy) ------------------------------- #
fig, ax = plt.subplots(figsize=(9, 4.5))
g = _agg(canon, "ae_vs_pca_ratio", "embedded_noisy")
x = np.arange(len(ORDER))
w = 0.38
for j, mode in enumerate(["flat@k", "matched"]):
    means = [g[(g.name == n) & (g["mode"] == mode)]["mean"].values for n in ORDER]
    means = [float(m[0]) if len(m) else np.nan for m in means]
    ax.bar(x + (j - 0.5) * w, means, w, label=mode, color=["#d6604d", "#4393c3"][j])
ax.axhline(1.0, ls=":", c="k", label="parity with PCA")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(ORDER, rotation=40, ha="right")
ax.axvline(len(EASY) - 0.5, ls="--", c="gray", lw=1)
ax.set_ylabel("FVU(PCA) / FVU(AE) on clean manifold")
ax.set_title("How many times better than the linear baseline (embedded+noise)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_ae_vs_pca.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --- Fig 3: k-sensitivity (flat mode, embedded_noisy) ---------------------- #
fig, ax = plt.subplots(figsize=(9, 4.5))
sub = df[(df.latent_mode == "flat") & (df.regime == "embedded_noisy")]
for n in ORDER:
    s = sub[sub.name == n].groupby("latent_dim")["fvu_clean"].mean()
    ax.plot(s.index, s.values, "o-", label=n)
    ktrue = df[df.name == n]["intrinsic_dim"].iloc[0]
    ax.scatter([ktrue], [s.get(ktrue, np.nan)], s=140, facecolors="none",
               edgecolors="k", zorder=5)
ax.set_yscale("log")
ax.set_xlabel("flat latent dim (circled = true intrinsic dim)")
ax.set_ylabel("FVU vs clean (log)")
ax.set_title("k-sensitivity of the flat baseline")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_k_sensitivity.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --- Fig 4: chart faithfulness (latent reduced to 2D, colored by theta) ---- #
def reduce2d(z):
    if z.shape[1] == 1:
        return np.concatenate([z, np.zeros_like(z)], axis=1)
    if z.shape[1] == 2:
        return z
    zc = z - z.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    return zc @ vt[:2].T

avail = [n for n in HARD if (ARRAYS / f"{n}_flat.npz").exists()]
if avail:
    fig, axes = plt.subplots(2, len(avail), figsize=(4 * len(avail), 8))
    axes = np.atleast_2d(axes)
    for col, n in enumerate(avail):
        for row, mode in enumerate(["flat", "matched"]):
            ax = axes[row, col]
            f = ARRAYS / f"{n}_{mode}.npz"
            if not f.exists():
                ax.set_visible(False)
                continue
            d = np.load(f, allow_pickle=True)
            z2 = reduce2d(d["z"])
            c = d["theta"][:, 0]
            sc = ax.scatter(z2[:, 0], z2[:, 1], c=c, cmap="hsv", s=4)
            ax.set_title(f"{n} — {mode} latent")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Chart faithfulness: latent code colored by true intrinsic "
                 "coordinate (flat folds/seams; matched is clean)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_faithfulness.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# --- Summary markdown ------------------------------------------------------ #
lines = ["# Toy manifold-AE sweep — summary\n",
         f"{len(df)} runs. Metrics averaged over seeds.\n",
         "\n## FVU vs clean manifold (embedded+noise), flat@k vs matched\n",
         "| manifold | tier | flat@k | matched | AE/PCA flat | AE/PCA matched |",
         "|---|---|---|---|---|---|"]
en = canon[canon.regime == "embedded_noisy"]
for n in ORDER:
    tier = "easy" if n in EASY else "closed"
    def cell(mode, col):
        v = en[(en.name == n) & (en["mode"] == mode)][col]
        return f"{v.mean():.4f}" if len(v) else "—"
    lines.append(f"| {n} | {tier} | {cell('flat@k','fvu_clean')} | "
                 f"{cell('matched','fvu_clean')} | {cell('flat@k','ae_vs_pca_ratio')} | "
                 f"{cell('matched','ae_vs_pca_ratio')} |")
lines += ["\n## Membership AUROC & denoising (embedded+noise)\n",
          "| manifold | mode | AUROC | projection_ratio | participation_ratio |",
          "|---|---|---|---|---|"]
for n in ORDER:
    for mode in ["flat@k", "matched"]:
        r = en[(en.name == n) & (en["mode"] == mode)]
        if len(r):
            lines.append(f"| {n} | {mode} | {r.membership_auroc.mean():.3f} | "
                         f"{r.projection_ratio.mean():.3f} | "
                         f"{r.latent_participation_ratio.mean():.2f} |")
(OUT_DIR / "summary.md").write_text("\n".join(lines))
print("wrote figures + summary.md to", OUT_DIR)
