"""How cheap can ONE manifold atom get? Sweep the first-projection width W.

A single nonlinear manifold atom (funnel  d -> W -> W/2 -> W/4 -> coord, decoder
mirror) is fit to each cached concept manifold's real Llama activations. We sweep
ONLY W (the first linear drop), the knob that owns ~98% of an atom's params.

Held out: test split (guards the small-N memorization artifact) + a rank-R linear
PCA baseline (the "SAE-style linear subspace at equal rank"). The gap between the
curved atom and PCA at the same rank IS the de-shattering win, quantified.

Run: PYTHONPATH=. python plotting_scripts/2026-06-08/manifold_atom_width_sweep/sweep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_manifold_data
from manifold_ae.atom import Atom

OUT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

RANK = 3
WIDTHS = [256, 128, 64, 32, 16]
STEPS = 3000
LR = 2e-3
# concept -> (label key to color by, matplotlib cmap, is the label cyclic?)
CONCEPTS = {
    "temperature": ("fahrenheit", "coolwarm", False),  # line (linear control)
    "years":       ("year",       "viridis",  False),  # helix
    "days":        ("day_idx",    "hsv",      True),    # day x time cylinder
    "colors":      ("hue",        "hsv",      True),    # hue loop
}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fvu(xhat, x):
    return (((xhat - x) ** 2).sum() / (x ** 2).sum()).item()


def fit(xtr, xte, W):
    atom = Atom(W=W).to(DEV)
    opt = torch.optim.Adam(atom.parameters(), lr=LR)
    for _ in range(STEPS):
        xhat, _ = atom(xtr)
        loss = ((xhat - xtr) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    atom.eval()
    with torch.no_grad():
        xhtr, _ = atom(xtr)
        xhte, _ = atom(xte)
    return atom, fvu(xhtr, xtr), fvu(xhte, xte), n_params(atom)


def main():
    results = []                 # (concept, W, params, train_fvu, test_fvu)
    latents = {}                 # (concept, W) -> (z_all [N,RANK], labels [N])
    pca_fvu = {}                 # concept -> rank-RANK linear PCA test fvu

    for cname, (lkey, _cmap, _cyc) in CONCEPTS.items():
        data = load_manifold_data(cname, filter_outliers=True, n_std=3.0)
        X = data["activations"].float().numpy()
        labs = np.array([l[lkey] for l in data["labels"]], dtype=float)
        N = len(X)
        perm = np.random.default_rng(0).permutation(N)
        ntr = int(0.85 * N)
        tri, tei = perm[:ntr], perm[ntr:]

        mu = X[tri].mean(0)
        Xc = X - mu
        sig = float(np.sqrt((Xc[tri] ** 2).mean()))
        Xn = Xc / sig
        xtr = torch.tensor(Xn[tri], device=DEV)
        xte = torch.tensor(Xn[tei], device=DEV)

        # rank-RANK linear PCA baseline (SAE-style linear subspace, same rank)
        _, _, Vt = np.linalg.svd(Xn[tri], full_matrices=False)
        Vr = Vt[:RANK]
        proj_te = (Xn[tei] @ Vr.T) @ Vr
        pca_fvu[cname] = float(((Xn[tei] - proj_te) ** 2).sum() / (Xn[tei] ** 2).sum())

        for W in WIDTHS:
            atom, ftr, fte, npar = fit(xtr, xte, W)
            results.append((cname, W, npar, ftr, fte))
            with torch.no_grad():
                zall = atom.encode(torch.tensor(Xn, device=DEV)).cpu().numpy()
            latents[(cname, W)] = (zall, labs)
            print(f"{cname:12s} W={W:3d}  params={npar/1e3:6.0f}K  "
                  f"train_fvu={ftr:.3f}  test_fvu={fte:.3f}  "
                  f"(pca{RANK}={pca_fvu[cname]:.3f}, N={N})", flush=True)

    # ---- CSV ----
    with open(OUT / "width_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "W", "params", "train_fvu", "test_fvu", f"pca{RANK}_test_fvu"])
        for c, W, p, ftr, fte in results:
            w.writerow([c, W, p, f"{ftr:.4f}", f"{fte:.4f}", f"{pca_fvu[c]:.4f}"])

    # ---- FVU vs W ----
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONCEPTS)))
    for col, cname in zip(colors, CONCEPTS):
        rows = [r for r in results if r[0] == cname]
        Ws = [r[1] for r in rows]
        te = [r[4] for r in rows]
        tr = [r[3] for r in rows]
        ax.plot(Ws, te, "-o", color=col, label=f"{cname} (test)")
        ax.plot(Ws, tr, "--", color=col, alpha=0.35)
        ax.axhline(pca_fvu[cname], color=col, ls=":", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([f"{W}\n{Atom(W=W).enc[0].weight.numel()*2/1e6:.1f}M" for W in WIDTHS])
    ax.set_xlabel("first-projection width W  (≈ wing params/atom)")
    ax.set_ylabel("test FVU  (solid) | train (dashed) | rank-3 PCA (dotted)")
    ax.set_title("Cheapest manifold atom: test FVU vs first-projection width")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fvu_vs_width.png", dpi=140)
    plt.close(fig)

    # ---- latent scatters: rows=concepts, cols=W subset ----
    Wviz = [256, 64, 16]
    fig, axes = plt.subplots(len(CONCEPTS), len(Wviz),
                             figsize=(3.2 * len(Wviz), 3.0 * len(CONCEPTS)))
    for i, (cname, (lkey, cmap, _cyc)) in enumerate(CONCEPTS.items()):
        for j, W in enumerate(Wviz):
            ax = axes[i, j]
            z, labs = latents[(cname, W)]
            zc = z - z.mean(0)
            _, _, Vt = np.linalg.svd(zc, full_matrices=False)
            z2 = zc @ Vt[:2].T
            sc = ax.scatter(z2[:, 0], z2[:, 1], c=labs, cmap=cmap, s=14)
            fte = [r[4] for r in results if r[0] == cname and r[1] == W][0]
            ax.set_title(f"{cname} W={W}  fvu={fte:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if j == len(Wviz) - 1:
                plt.colorbar(sc, ax=ax, fraction=0.046, label=lkey)
    fig.suptitle("rank-3 atom latent (PCA→2D), colored by known label", y=1.002)
    fig.tight_layout()
    fig.savefig(OUT / "latents.png", dpi=140)
    plt.close(fig)

    # ---- SUMMARY.md: cheapest W within 10% rel of best test FVU per concept ----
    lines = ["# Cheapest manifold atom — first-projection width sweep\n",
             f"rank={RANK}, steps={STEPS}, lr={LR}, 85/15 train/test.\n",
             "| concept | N | best test FVU (W) | rank-3 PCA | cheapest W ≤+10% | params@cheapest |",
             "|---|---|---|---|---|---|"]
    for cname in CONCEPTS:
        rows = sorted([r for r in results if r[0] == cname], key=lambda r: r[1])
        best = min(rows, key=lambda r: r[4])
        thr = best[4] * 1.10
        ok = [r for r in rows if r[4] <= thr]
        cheap = min(ok, key=lambda r: r[1])
        N = "—"
        lines.append(f"| {cname} | {cheap and ''} | {best[4]:.3f} (W={best[1]}) | "
                     f"{pca_fvu[cname]:.3f} | **W={cheap[1]}** | {cheap[2]/1e3:.0f}K |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n[done] wrote width_sweep.csv, fvu_vs_width.png, latents.png, SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
