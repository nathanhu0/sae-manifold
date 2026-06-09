"""Intrinsic rank vs smooth-embedding rank: the sphere seam test.

S^2 is a CLOSED 2-manifold: it cannot embed in R^2 without a seam (you can't
flatten a globe). So a manifold atom with a rank-2 latent must TEAR the sphere
somewhere; a rank-3 latent can hold it seamlessly. We test this on:
  * synthetic S^2 isometrically embedded in R^4096 + noise (clean, known topology)
  * real `geography` activations (cities on Earth's S^2; the real instance)

Headline diagnostic is NOT mean FVU (a rank-2 atom can still fit most of the
sphere) but the SEAM MAP: per-point reconstruction error on the globe. Prediction:
rank-2 concentrates error along a meridian (the tear); rank-3 is uniform.

Run: PYTHONPATH=. python plotting_scripts/2026-06-08/manifold_atom_width_sweep/sphere_rank_test.py
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

from manifold_ae.atom import Atom
from data import load_manifold_data

OUT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

W_FIXED = 128
RANKS = [2, 3, 4]
STEPS = 5000
LR = 2e-3


def make_sphere(n=3000, ambient=4096, noise=0.05, seed=0):
    """Uniform S^2 in a random 3-D subspace of R^ambient + Gaussian noise.

    `noise` is the noise NORM relative to the unit-norm signal (per-coordinate
    std is noise/sqrt(ambient), else the sqrt(d) blow-up swamps the sphere)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)          # unit sphere
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, 3)))      # orthonormal embed
    X = (v @ Q.T).astype(np.float32)                        # (n, ambient), unit-norm rows
    X += (noise / np.sqrt(ambient)) * rng.normal(size=X.shape).astype(np.float32)
    lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return X, lat, lon


def normalize_split(X, split=0.85, seed=0):
    N = len(X)
    perm = np.random.default_rng(seed).permutation(N)
    ntr = int(split * N)
    tri, tei = perm[:ntr], perm[ntr:]
    mu = X[tri].mean(0)
    sig = float(np.sqrt(((X[tri] - mu) ** 2).mean()))
    Xn = (X - mu) / sig
    return Xn, tri, tei


def fit_rank(Xn, tri, tei, rank, W=W_FIXED):
    xtr = torch.tensor(Xn[tri], device=DEV)
    xte = torch.tensor(Xn[tei], device=DEV)
    atom = Atom(W=W, rank=rank).to(DEV)
    opt = torch.optim.Adam(atom.parameters(), lr=LR)
    for _ in range(STEPS):
        xhat, _ = atom(xtr)
        loss = ((xhat - xtr) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    atom.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xn, device=DEV)
        xhat, z = atom(Xt)
        err = ((xhat - Xt) ** 2).mean(dim=1).cpu().numpy()   # per-point error (all)
        tr = (((atom(xtr)[0] - xtr) ** 2).sum() / (xtr ** 2).sum()).item()
        te = (((atom(xte)[0] - xte) ** 2).sum() / (xte ** 2).sum()).item()
    return tr, te, err, z.cpu().numpy()


def seam_score(err, lon, nbins=24):
    """Error concentration along longitude: max-bin / mean-bin (high => seam)."""
    bins = np.digitize(lon, np.linspace(-180, 180, nbins + 1)[1:-1])
    means = np.array([err[bins == b].mean() if (bins == b).any() else np.nan
                      for b in range(nbins)])
    return float(np.nanmax(means) / np.nanmean(means))


def run(name, X, lat, lon, results, err_maps, latents):
    Xn, tri, tei = normalize_split(X)
    for rank in RANKS:
        tr, te, err, z = fit_rank(Xn, tri, tei, rank)
        ss = seam_score(err, lon)
        results.append((name, rank, tr, te, ss))
        err_maps[(name, rank)] = err
        latents[(name, rank)] = z
        print(f"{name:10s} rank={rank}  train_fvu={tr:.3f}  test_fvu={te:.3f}  "
              f"seam_score={ss:.2f}", flush=True)


def main():
    results, err_maps, latents = [], {}, {}

    Xs, lat_s, lon_s = make_sphere()
    run("sphere", Xs, lat_s, lon_s, results, err_maps, latents)

    data = load_manifold_data("geography", filter_outliers=True, n_std=3.0)
    Xg = data["activations"].float().numpy()
    lat_g = np.array([l["latitude"] for l in data["labels"]], dtype=float)
    lon_g = np.array([l["longitude"] for l in data["labels"]], dtype=float)
    run("geography", Xg, lat_g, lon_g, results, err_maps, latents)

    with open(OUT / "sphere_rank.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["manifold", "rank", "train_fvu", "test_fvu", "seam_score"])
        for r in results:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.3f}"])

    # ---- seam maps: rows=manifold, cols=rank; error on the globe ----
    coords = {"sphere": (lon_s, lat_s), "geography": (lon_g, lat_g)}
    for name in ("sphere", "geography"):
        lon, lat = coords[name]
        fig, axes = plt.subplots(1, len(RANKS), figsize=(4.2 * len(RANKS), 3.4))
        for ax, rank in zip(axes, RANKS):
            err = err_maps[(name, rank)]
            vmax = np.percentile(err, 98)
            sc = ax.scatter(lon, lat, c=err, cmap="inferno", s=8, vmin=0, vmax=vmax)
            te = [r[3] for r in results if r[0] == name and r[1] == rank][0]
            ss = [r[4] for r in results if r[0] == name and r[1] == rank][0]
            ax.set_title(f"rank={rank}  fvu={te:.2f}  seam={ss:.1f}", fontsize=10)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            plt.colorbar(sc, ax=ax, fraction=0.046, label="recon error")
        fig.suptitle(f"{name}: per-point reconstruction error on the globe "
                     f"(seam = error concentrated at a meridian)", y=1.03)
        fig.tight_layout()
        fig.savefig(OUT / f"seam_{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # ---- rank-2 sphere latent: the unrolled, torn sphere ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, lbl in [(axes[0], lat_s, "latitude"), (axes[1], lon_s, "longitude")]:
        z = latents[("sphere", 2)]
        sc = ax.scatter(z[:, 0], z[:, 1], c=col, cmap="hsv", s=10)
        ax.set_title(f"rank-2 sphere latent, colored by {lbl}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "sphere_rank2_latent.png", dpi=140)
    plt.close(fig)

    print("\n[done] wrote sphere_rank.csv, seam_sphere.png, seam_geography.png, "
          "sphere_rank2_latent.png", flush=True)


if __name__ == "__main__":
    main()
