"""Aggregate-best panels with structure-revealing 3-D views.

Each manifold rendered at ITS best-capture budget (lowest best-atom FVU across the
sweep), with [latent | reconstruction | target] where the ambient views actually
show the geometry:
  * k_i = 1  -> (theta, coord)
  * k_i = 2  -> 2-D scatter
  * k_i = 3  -> 3-D scatter (top-3 PCA)            sphere looks spherical, etc.
  * torus    -> 3-D donut via recovered angles     (t = atan2(c1,c0), p = atan2(c3,c2))
latent is 2-D (rank<=2) or 3-D (rank>=3). Colored by the true intrinsic coord.

Run (CPU, read-only): source .venv/bin/activate && PYTHONPATH=. \
  python plotting_scripts/2026-06-08/per_manifold_rank_capture/aggregate_best_panels.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_ae.manifold_zoo import ManifoldZoo, TYPES
from manifold_ae.manifold_sae import ManifoldSAE

OUT_DIR = Path(__file__).parent
SWEEP = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-08_budget_sweep")
BUDGETS = [3, 4, 5, 6, 7, 8, 9, 12, 16]
VARIANT = 2
L0 = 4
BATCH_MIX = 10000
NMAX_PLOT = 700
DEV = "cpu"
CYCLIC = {"circle", "sphere", "torus", "mobius"}


def load(b):
    ck = torch.load(SWEEP / f"manifold_R{b}" / "ckpt.pt", map_location=DEV)
    m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], R_target=ck["R"],
                    enc_dims=tuple(ck["enc_dims"])).to(DEV)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["scale"]


def best_atom(model, scale, x_i, m_i):
    inp = torch.tensor(x_i / scale, device=DEV)
    tgt = torch.tensor(m_i / scale, device=DEV)
    with torch.no_grad():
        z, g = model.encode(inp - model.b_dec)
        decA = model.decode_all(z)
        c = g.unsqueeze(-1) * decA
        err = ((c - tgt.unsqueeze(1)) ** 2).sum(-1)
        den = (tgt ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
        fvu_a = (err / den).mean(0).cpu().numpy()
        a = int(np.argmin(fvu_a)); r = int(model.ranks[a]); fvu = float(fvu_a[a])
        z_a = z[:, a, :r].cpu().numpy()
        c_a = (g[:, a].unsqueeze(-1) * decA[:, a]).cpu().numpy()
    return a, r, fvu, z_a, c_a


def amb_view(pts_k, ki, theta, name, basis_mu):
    """-> (proj, coords-tuple). pts_k is (n, k_i) in V coordinates."""
    if ki == 1:
        return "2d", (theta, pts_k[:, 0])
    if ki == 2:
        return "2d", (pts_k[:, 0], pts_k[:, 1])
    if name == "torus":
        t = np.arctan2(pts_k[:, 1], pts_k[:, 0]); p = np.arctan2(pts_k[:, 3], pts_k[:, 2])
        return "3d", ((2 + np.cos(p)) * np.cos(t), (2 + np.cos(p)) * np.sin(t), np.sin(p))
    mu, B = basis_mu
    P = (pts_k - mu) @ B.T
    return "3d", (P[:, 0], P[:, 1], P[:, 2])


def latent_view(z, rank):
    if rank == 1:
        return "2d", (z[:, 0],)            # x = theta supplied by caller
    if rank == 2:
        return "2d", (z[:, 0], z[:, 1])
    return "3d", (z[:, 0], z[:, 1], z[:, 2])


def draw(fig, nrows, row, col, proj, coords, theta, cmap, title, xlabel=None):
    ax = fig.add_subplot(nrows, 3, row * 3 + col + 1, projection=("3d" if proj == "3d" else None))
    ax.scatter(*coords, c=theta, cmap=cmap, s=5)
    ax.set_title(title, fontsize=9)
    if proj == "2d" and xlabel:
        ax.set_xlabel(xlabel)
    if proj == "2d" and len(coords) == 2:
        ax.set_aspect("equal", "datalim")
    if proj == "3d":
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    return ax


def main():
    rng = np.random.default_rng(0)
    zoo = ManifoldZoo(d=256, seed=0)
    by_name = {inst.name: inst for inst in zoo.instances}
    x, masks, truth, theta = zoo.sample(BATCH_MIX, L0, rng,
                                        return_truth=True, return_theta=True)
    models = {b: load(b) for b in BUDGETS}

    nrows = len(TYPES)
    fig = plt.figure(figsize=(12.5, 3.2 * nrows))
    for ti, (name, di, ki, *_) in enumerate(TYPES):
        inst = by_name[f"{name}_{VARIANT}"]
        idx = zoo.instances.index(inst)
        cmap = "twilight" if name in CYCLIC else "viridis"
        rows = np.where(masks[:, idx])[0]
        x_i, m_i, theta_c = x[rows], truth[rows, idx], theta[rows, idx, 0]

        # pick the budget with the cleanest single-atom capture
        cand = [(b, *best_atom(*models[b], x_i, m_i)) for b in BUDGETS]
        b, a, r, fvu, z_a, c_a = min(cand, key=lambda t: t[3])
        rec_k = (c_a * models[b][1]) @ inst.V.T
        tgt_k = m_i @ inst.V.T

        sel = np.random.default_rng(0).permutation(len(rows))[:NMAX_PLOT]
        th = theta_c[sel]
        # shared top-3 PCA basis from target (for k_i==3 ambient views)
        if ki == 3:
            mu = tgt_k[sel].mean(0); _, _, Vt = np.linalg.svd(tgt_k[sel] - mu, full_matrices=False)
            basis_mu = (mu, Vt[:3])
        else:
            basis_mu = (None, None)

        lp, lc = latent_view(z_a[sel], r)
        if r == 1:
            lc = (th, lc[0])
        draw(fig, nrows, ti, 0, lp, lc, th, cmap,
             f"{name}: best atom rank={r}  (d_i={di}, k_i={ki}, R={b})\nlatent",
             xlabel="theta_i" if r == 1 else None)
        rp, rc = amb_view(rec_k[sel], ki, th, name, basis_mu)
        draw(fig, nrows, ti, 1, rp, rc, th, cmap, f"reconstruction  FVU={fvu:.3f}",
             xlabel="theta_i" if ki == 1 else None)
        tp, tc = amb_view(tgt_k[sel], ki, th, name, basis_mu)
        draw(fig, nrows, ti, 2, tp, tc, th, cmap, "target m_i",
             xlabel="theta_i" if ki == 1 else None)
        print(f"[{name}] best R={b} rank={r} FVU={fvu:.3f}  view={tp}", flush=True)

    fig.suptitle("Best single-atom capture per manifold (each at its best budget) — "
                 "3-D views; torus as recovered-angle donut", y=1.002, fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "aggregate_best_panels.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
