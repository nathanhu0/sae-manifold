"""Milestone-2 figures: latent-dim behavior and high-dim scaling."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent
R = Path("/juice2/u/nathu/sae-manifold/results/manifold_ae")

# --- fig7: latent-dim behavior -------------------------------------------- #
ld = pd.read_parquet(R / "sweep_latentdim.parquet")
g = ld.groupby(["name", "intrinsic_dim", "latent_dim"]).fvu_clean.mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
for name in ["swiss_roll", "sphere", "torus", "sphere10", "flat10"]:
    s = g[g.name == name].sort_values("latent_dim")
    k = int(s.intrinsic_dim.iloc[0])
    line, = ax.plot(s.latent_dim, s.fvu_clean, "o-", label=f"{name} (k*={k})")
    ax.axvline(k, color=line.get_color(), ls=":", alpha=0.4)
ax.set_yscale("log")
ax.set_xlabel("flat latent dim k  (dotted = true intrinsic dim k*)")
ax.set_ylabel("FVU vs clean (log)")
ax.set_title("Latent-dim behavior: under-complete fails hard, over-complete is safe & often better")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig7_latent_dim.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote fig7")

# --- fig8: high-dim scaling ----------------------------------------------- #
hd = pd.read_parquet(R / "sweep_highdim.parquet")
hd["family"] = hd.name.str.replace(r"\d+", "", regex=True)
g = hd.groupby(["family", "intrinsic_dim", "latent_mode"]).agg(
    fvu=("fvu_clean", "mean"), aepca=("ae_vs_pca_ratio", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"flat": "#1b9e77", "sphere": "#d95f02", "torus": "#7570b3"}
for fam in ["flat", "sphere", "torus"]:
    for mode, ls in [("flat", "-"), ("structured", "--")]:
        s = g[(g.family == fam) & (g.latent_mode == mode)].sort_values("intrinsic_dim")
        axes[0].plot(s.intrinsic_dim, s.fvu, ls, marker="o", color=colors[fam],
                     label=f"{fam} · {mode}")
        axes[1].plot(s.intrinsic_dim, s.aepca, ls, marker="o", color=colors[fam],
                     label=f"{fam} · {mode}")
axes[0].set_yscale("log"); axes[0].set_xlabel("intrinsic dim k*"); axes[0].set_ylabel("FVU vs clean (log)")
axes[0].axhline(1.0, color="k", ls=":", lw=1, alpha=0.6); axes[0].set_title("Fit quality vs intrinsic dim")
axes[1].set_yscale("log"); axes[1].set_xlabel("intrinsic dim k*"); axes[1].set_ylabel("FVU(PCA)/FVU(AE)")
axes[1].axhline(1.0, color="k", ls=":", lw=1, alpha=0.6); axes[1].set_title("Advantage over linear PCA")
axes[0].legend(fontsize=8, ncol=2)
fig.suptitle("High-dim scaling: spheres & flat patches scale; tori (products of circles) are the wall for flat", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "fig8_highdim_scaling.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote fig8")
