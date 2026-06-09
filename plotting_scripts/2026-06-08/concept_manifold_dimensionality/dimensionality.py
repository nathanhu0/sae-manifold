"""Actual vs functional dimensionality of each cached concept manifold.

Three notions, side by side:
  functional  = generative DOF by construction (from data.py builders)
  linear PR   = PCA participation ratio (Σλ)²/Σλ² — linear dim the cloud fills;
                a curved k-manifold inflates this above k (curvature surcharge)
  intrinsic   = TwoNN estimate (Facco et al.) — nonlinear, curvature-robust dim

Run: PYTHONPATH=. python plotting_scripts/2026-06-08/manifold_atom_width_sweep/dimensionality.py
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

from data import load_manifold_data, get_all_manifold_names

OUT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# functional DOF + topology, from the dataset builders (data.py)
FUNCTIONAL = {
    "age":         (1, "line"),
    "temperature": (1, "line (celsius redundant)"),
    "formality":   (1, "line (diverse sentences -> high ambient)"),
    "sent_length": (1, "line (diverse sentences -> high ambient)"),
    "years":       (1, "helix (1-D curve; decade=f(year))"),
    "days":        (2, "cylinder S^1 x interval (day cyclic x time)"),
    "geography":   (2, "sphere S^2 (lat,long on Earth)"),
    "colors":      (3, "S^1 x I x I (hue cyclic, lightness, saturation)"),
}


def participation_ratio(eigs):
    eigs = eigs[eigs > 0]
    return float((eigs.sum() ** 2) / (eigs ** 2).sum())


def n_for_var(eigs, frac):
    c = np.cumsum(eigs) / eigs.sum()
    return int(np.searchsorted(c, frac) + 1)


def twonn(X, max_n=2500, fit_frac=0.9, seed=0):
    """TwoNN intrinsic-dimension estimate."""
    if len(X) > max_n:
        idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X = X[idx]
    Xt = torch.tensor(X, device=DEV)
    D = torch.cdist(Xt, Xt)
    D.fill_diagonal_(float("inf"))
    r1, _ = D.min(dim=1)
    D.scatter_(1, D.argmin(dim=1, keepdim=True), float("inf"))
    r2, _ = D.min(dim=1)
    mu = (r2 / r1).cpu().numpy()
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    mu = np.sort(mu)
    Femp = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    x = np.log(mu)
    y = -np.log(1.0 - Femp)
    keep = int(fit_frac * len(x))
    return float(np.sum(x[:keep] * y[:keep]) / np.sum(x[:keep] ** 2))


def main():
    rows = []
    for name in get_all_manifold_names():
        data = load_manifold_data(name, filter_outliers=True, n_std=3.0)
        if data is None:
            continue
        X = data["activations"].float().numpy()
        N, d = X.shape
        Xc = X - X.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)
        eigs = (s ** 2) / (N - 1)
        pr = participation_ratio(eigs)
        n90 = n_for_var(eigs, 0.90)
        n95 = n_for_var(eigs, 0.95)
        idim = twonn(X)
        fdof, topo = FUNCTIONAL.get(name, ("?", "?"))
        rows.append((name, N, fdof, idim, pr, n90, n95, topo))
        print(f"{name:12s} N={N:5d}  functional={fdof}  TwoNN={idim:4.1f}  "
              f"PR={pr:6.1f}  PCs@90%={n90:4d}  PCs@95%={n95:4d}   {topo}", flush=True)

    with open(OUT / "dimensionality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "N", "functional_dof", "twonn_intrinsic",
                    "linear_PR", "PCs_90pct", "PCs_95pct", "topology"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", r[5], r[6], r[7]])

    # bar chart: functional vs TwoNN vs linear PR (log y)
    names = [r[0] for r in rows]
    fdof = [r[2] for r in rows]
    idim = [r[3] for r in rows]
    pr = [r[4] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.27, fdof, 0.27, label="functional DOF (by construction)")
    ax.bar(x, idim, 0.27, label="intrinsic (TwoNN, curvature-robust)")
    ax.bar(x + 0.27, pr, 0.27, label="linear PR (PCA)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("dimensionality (log)")
    ax.set_title("Concept manifolds: functional vs intrinsic vs linear dimensionality")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dimensionality.png", dpi=140)
    print("\n[done] wrote dimensionality.csv, dimensionality.png", flush=True)


if __name__ == "__main__":
    main()
