"""Best-atom-per-manifold across the FULL budget sweep, + aggregate table.

For EVERY trained mixture checkpoint (R3..R16) and one representative instance per
manifold type, find THE single best atom capturing that manifold (lowest
variance-unexplained of its ground-truth contribution m_i, over all atoms, any
rank), probed on the training mixture distribution (L0=4).

Outputs (alongside this script):
  * SWEEP_FVU.md   — manifold x budget matrix of best-atom FVU
  * SWEEP_RANK.md  — manifold x budget matrix of best-atom rank
  * AGGREGATE.md   — per manifold: d_i, k_i, best achievable FVU + its rank + budget
  * best_atom_panels_R{b}.png — [latent | reconstruction | target] per budget

Run (CPU, read-only): source .venv/bin/activate && PYTHONPATH=. \
  python plotting_scripts/2026-06-08/per_manifold_rank_capture/best_atom_sweep.py
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
NMAX_PLOT = 900
DEV = "cpu"
CYCLIC = {"circle", "sphere", "torus", "mobius"}


def pca2d(target):
    mu = target.mean(0)
    _, _, Vt = np.linalg.svd(target - mu, full_matrices=False)
    return mu, Vt[:2]


def scatter_latent(ax, z, theta, rank, cmap):
    if rank == 1:
        ax.scatter(theta, z[:, 0], c=theta, cmap=cmap, s=5)
        ax.set_xlabel("theta_i"); ax.set_ylabel("z")
    else:
        ax.scatter(z[:, 0], z[:, 1], c=theta, cmap=cmap, s=5)
        ax.set_xlabel("z0"); ax.set_ylabel("z1"); ax.set_aspect("equal", "datalim")


def scatter_space(ax, pts, theta, ki, cmap, mu, basis):
    if ki == 1:
        ax.scatter(theta, pts[:, 0], c=theta, cmap=cmap, s=5)
        ax.set_xlabel("theta_i"); ax.set_ylabel("coord")
    else:
        P = (pts - mu) @ basis.T
        ax.scatter(P[:, 0], P[:, 1], c=theta, cmap=cmap, s=5)
        ax.set_xlabel("pc0" + ("" if ki <= 2 else f" (of {ki})")); ax.set_ylabel("pc1")
        ax.set_aspect("equal", "datalim")


def best_atom_for_budget(model, scale, x, masks, truth, theta, zoo, by_name, render_path):
    """-> dict name -> (rank, fvu, atom). Also renders the [latent|recon|target] panels."""
    ranks = model.ranks.cpu().numpy()
    out = {}
    fig, axes = plt.subplots(len(TYPES), 3, figsize=(12, 3.0 * len(TYPES)), squeeze=False)
    for ti, (name, di, ki, *_) in enumerate(TYPES):
        inst = by_name[f"{name}_{VARIANT}"]
        idx = zoo.instances.index(inst)
        cmap = "twilight" if name in CYCLIC else "viridis"
        rows = np.where(masks[:, idx])[0]
        m_i = truth[rows, idx]; theta_c = theta[rows, idx, 0]
        inp = torch.tensor(x[rows] / scale, device=DEV)
        tgt = torch.tensor(m_i / scale, device=DEV)
        with torch.no_grad():
            z, g = model.encode(inp - model.b_dec)
            decA = model.decode_all(z)
            c = g.unsqueeze(-1) * decA
            err = ((c - tgt.unsqueeze(1)) ** 2).sum(-1)
            den = (tgt ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
            fvu_a = (err / den).mean(0).cpu().numpy()
            a = int(np.argmin(fvu_a)); r_a = int(ranks[a]); fvu = float(fvu_a[a])
            z_a = z[:, a, :r_a].cpu().numpy()
            c_a = (g[:, a].unsqueeze(-1) * decA[:, a]).cpu().numpy()
        out[name] = (r_a, fvu, a)
        rec_k = (c_a * scale) @ inst.V.T
        tgt_k = m_i @ inst.V.T
        sel = np.random.default_rng(0).permutation(len(rows))[:NMAX_PLOT]
        mu, basis = (pca2d(tgt_k[sel]) if ki > 1 else (None, None))
        axL, axM, axR = axes[ti]
        scatter_latent(axL, z_a[sel], theta_c[sel], r_a, cmap)
        scatter_space(axM, rec_k[sel], theta_c[sel], ki, cmap, mu, basis)
        scatter_space(axR, tgt_k[sel], theta_c[sel], ki, cmap, mu, basis)
        axL.set_title(f"{name}: best atom rank={r_a} (d_i={di}, k_i={ki})\nlatent", fontsize=9)
        axM.set_title(f"reconstruction  FVU={fvu:.3f}", fontsize=9)
        axR.set_title("target m_i", fontsize=9)
    fig.suptitle(f"Best atom per manifold — trained mixture SAE R={model.R_target} "
                 f"(probed on L0={L0} mixtures)", y=1.002, fontsize=12)
    fig.tight_layout(); fig.savefig(render_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    rng = np.random.default_rng(0)
    zoo = ManifoldZoo(d=256, seed=0)
    by_name = {inst.name: inst for inst in zoo.instances}
    x, masks, truth, theta = zoo.sample(BATCH_MIX, L0, rng,
                                        return_truth=True, return_theta=True)

    results = {}   # budget -> {name: (rank, fvu, atom)}
    for b in BUDGETS:
        p = SWEEP / f"manifold_R{b}" / "ckpt.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location=DEV)
        m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], R_target=ck["R"],
                        enc_dims=tuple(ck["enc_dims"])).to(DEV)
        m.load_state_dict(ck["state_dict"]); m.eval()
        results[b] = best_atom_for_budget(m, ck["scale"], x, masks, truth, theta,
                                          zoo, by_name, OUT_DIR / f"best_atom_panels_R{b}.png")
        print(f"[R={b:>2}] " + "  ".join(
            f"{n[:4]}:r{results[b][n][0]}/{results[b][n][1]:.2f}" for n, *_ in TYPES),
            flush=True)

    bs = [b for b in BUDGETS if b in results]
    # SWEEP_FVU + SWEEP_RANK matrices
    hdr = "| manifold | d_i | k_i | " + " | ".join(f"R{b}" for b in bs) + " |"
    sep = "|---|---|---|" + "---|" * len(bs)
    fvu_lines = ["# Best-atom FVU (variance unexplained) — manifold x budget\n", hdr, sep]
    rank_lines = ["# Best-atom rank — manifold x budget (compare to d_i / k_i)\n", hdr, sep]
    agg = []
    for name, di, ki, *_ in TYPES:
        fvu_row, rank_row = [], []
        best = (9.0, None, None)   # (fvu, rank, budget)
        for b in bs:
            r_a, fvu, _ = results[b][name]
            fvu_row.append(f"{fvu:.3f}"); rank_row.append(str(r_a))
            if fvu < best[0]:
                best = (fvu, r_a, b)
        fvu_lines.append(f"| {name} | {di} | {ki} | " + " | ".join(fvu_row) + " |")
        rank_lines.append(f"| {name} | {di} | {ki} | " + " | ".join(rank_row) + " |")
        agg.append((name, di, ki, best[0], best[1], best[2]))

    agg_lines = ["# Aggregate: best single-atom capture achievable across the budget sweep\n",
                 "Per manifold, the lowest best-atom FVU over all budgets, the rank of that",
                 "atom, and the budget where it occurs. 'rank @ best' vs d_i/k_i shows whether",
                 "the cleanest single-atom capture lands near the intrinsic dim (de-shatter)",
                 "or the embedding dim.\n",
                 "| manifold | d_i | k_i | best FVU | rank @ best | budget @ best |",
                 "|---|---|---|---|---|---|"]
    for name, di, ki, fvu, r, b in agg:
        agg_lines.append(f"| {name} | {di} | {ki} | {fvu:.3f} | {r} | R{b} |")

    (OUT_DIR / "SWEEP_FVU.md").write_text("\n".join(fvu_lines))
    (OUT_DIR / "SWEEP_RANK.md").write_text("\n".join(rank_lines))
    (OUT_DIR / "AGGREGATE.md").write_text("\n".join(agg_lines))
    print("\n" + "\n".join(fvu_lines))
    print("\n" + "\n".join(rank_lines))
    print("\n" + "\n".join(agg_lines))
    print(f"\n[done] SWEEP_FVU.md, SWEEP_RANK.md, AGGREGATE.md + {len(bs)} panel PNGs -> {OUT_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
