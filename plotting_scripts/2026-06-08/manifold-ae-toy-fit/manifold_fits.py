"""Per-manifold 4-column view, with one consistent per-point coloring so you can
trace a point through the whole chart:

  col 1  original shape (ambient), colored by intrinsic coordinate
  col 2  latent code z (encoder output), same colors
  col 3  decoder reconstruction dec(enc(x)) (ambient), same colors
  col 4  overlay: original (grey) + reconstruction (colored)

Reconstruction uses the actual data points (always in-distribution), so there is
no grid-extrapolation artifact. Trains small models on native data so we can plot
directly. Run:  CUDA_VISIBLE_DEVICES=0 python manifold_fits.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, "/juice2/u/nathu/sae-manifold")
from manifold_ae.toy_manifolds import make_manifold
from manifold_ae.train import TrainConfig, train_one

OUT_DIR = Path(__file__).parent


def fit(name, mode, n=2500, epochs=600, width=256, depth=3):
    data = make_manifold(name, n=n, regime="native", noise_std=0.0, seed=0)
    cfg = TrainConfig(epochs=epochs, latent_mode=mode, width=width, depth=depth, seed=0)
    model = train_one(data, cfg).model
    dev = next(model.parameters()).device
    with torch.no_grad():
        X = torch.as_tensor(data.X, device=dev)
        z = model.encode(X).cpu().numpy()
        recon = model.decode(model.encode(X)).cpu().numpy()
    return data, z, recon


def _n(x):
    return (x - x.min()) / (np.ptp(x) + 1e-9)


def color_of(name, theta):
    """Intuitive per-point coloring from the true intrinsic coordinate."""
    if name == "circle":
        h = (theta[:, 0] % (2 * np.pi)) / (2 * np.pi)
        return hsv_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], 1))
    if name == "helix":
        return cm.viridis(_n(theta[:, 0]))[:, :3]
    if name == "sphere":                       # globe coloring: hue=azimuth, val=polar
        h, v = _n(theta[:, 0]), 0.35 + 0.65 * _n(theta[:, 1])
        return hsv_to_rgb(np.stack([h, np.ones_like(h), v], 1))
    if name == "torus":                        # hue=major angle, val=minor angle
        h = (theta[:, 0] % (2 * np.pi)) / (2 * np.pi)
        v = 0.4 + 0.6 * ((theta[:, 1] % (2 * np.pi)) / (2 * np.pi))
        return hsv_to_rgb(np.stack([h, np.ones_like(h), v], 1))
    return np.stack([_n(theta[:, 0]), _n(theta[:, 1]),
                     0.5 * np.ones(len(theta))], 1)   # 2D RGB chart


def ambient_proj(D):
    return "3d" if D == 3 else None


def latent_proj(k):
    return "3d" if k == 3 else None


def draw_ambient(ax, X, c, s=7, alpha=1.0):
    if X.shape[1] == 2:
        ax.scatter(X[:, 0], X[:, 1], c=c, s=s, alpha=alpha)
        ax.set_aspect("equal")
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, s=s, alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([])
    if X.shape[1] == 3:
        ax.set_zticks([])


def draw_latent(ax, z, c):
    k = z.shape[1]
    if k == 1:
        ax.scatter(z[:, 0], np.zeros_like(z[:, 0]), c=c, s=7)
        ax.set_yticks([])
    elif k == 2:
        ax.scatter(z[:, 0], z[:, 1], c=c, s=7); ax.set_aspect("equal")
    elif k == 3:
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, s=7); ax.set_zticks([])
    elif k == 4:                                # torus structured: (cos a,sin a,cos b,sin b)
        a = np.arctan2(z[:, 1], z[:, 0]); b = np.arctan2(z[:, 3], z[:, 2])
        ax.scatter(a, b, c=c, s=7)
    else:
        zc = z - z.mean(0); _, _, vt = np.linalg.svd(zc, full_matrices=False)
        p = zc @ vt[:2].T; ax.scatter(p[:, 0], p[:, 1], c=c, s=7)
    ax.set_xticks([]); ax.set_yticks([])


def build_figure(rows, out, title):
    """rows: list of (name, mode). One 4-col row each."""
    nr = len(rows)
    fig = plt.figure(figsize=(15, 3.6 * nr))
    col_titles = ["original shape", "latent code", "reconstruction",
                  "overlay (grey=true)"]
    for r, (name, mode) in enumerate(rows):
        data, z, recon = fit(name, mode)
        c = color_of(name, data.theta)
        D, k = data.ambient_dim, z.shape[1]
        projs = [ambient_proj(D), latent_proj(k), ambient_proj(D), ambient_proj(D)]
        for col in range(4):
            ax = fig.add_subplot(nr, 4, r * 4 + col + 1, projection=projs[col])
            if col == 0:
                draw_ambient(ax, data.X, c)
            elif col == 1:
                draw_latent(ax, z, c)
            elif col == 2:
                draw_ambient(ax, recon, c)
            else:
                draw_ambient(ax, data.X, np.full((len(data.X), 3), 0.8), alpha=0.4)
                draw_ambient(ax, recon, c, alpha=0.9)
            if r == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{name}\n({mode})", fontsize=10)
    fig.suptitle(title, y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


build_figure([("helix", "flat"), ("swiss_roll", "flat"),
              ("s_curve", "flat"), ("wavy_sheet", "flat")],
             OUT_DIR / "fig5_fits_contractible.png",
             "Contractible manifolds (flat latent): original / latent / reconstruction / overlay")

build_figure([("circle", "flat"), ("circle", "structured"),
              ("sphere", "flat"), ("sphere", "structured"),
              ("torus", "flat"), ("torus", "structured")],
             OUT_DIR / "fig6_fits_closed.png",
             "Closed manifolds, flat vs structured latent: original / latent / reconstruction / overlay")
