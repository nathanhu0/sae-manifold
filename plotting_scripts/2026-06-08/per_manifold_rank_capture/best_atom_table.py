"""Best-atom-per-manifold table + panels, for ONE trained mixture SAE.

For a single trained mixture checkpoint, and one representative instance per
manifold type:
  * find THE single best atom capturing that manifold (lowest variance-unexplained
    of its ground-truth contribution m_i, over ALL atoms, any rank), probing on the
    training mixture distribution (L0=4) using known truth m_i;
  * report that atom's RANK and FVU (variance unexplained), next to the manifold's
    intrinsic d_i and embedding k_i;
  * render per type: [ latent z(theta_i) | reconstruction | target ].

Outputs TABLE.md + best_atom_panels.png alongside this script.

Run (CPU, read-only): source .venv/bin/activate && PYTHONPATH=. \
  python plotting_scripts/2026-06-08/per_manifold_rank_capture/best_atom_table.py
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
CKPT = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-08_budget_sweep/manifold_R6/ckpt.pt")
VARIANT = 2
L0 = 4
BATCH_MIX = 12000
NMAX_PLOT = 900                      # subsample rows for legible scatter
DEV = "cpu"
CYCLIC = {"circle", "sphere", "torus", "mobius"}


def load_model():
    ck = torch.load(CKPT, map_location=DEV)
    m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], R_target=ck["R"],
                    enc_dims=tuple(ck["enc_dims"])).to(DEV)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["scale"], ck["R"]


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


def main():
    rng = np.random.default_rng(0)
    zoo = ManifoldZoo(d=256, seed=0)
    by_name = {inst.name: inst for inst in zoo.instances}
    model, scale, R = load_model()
    ranks = model.ranks.cpu().numpy()
    print(f"[model] R={R}, {model.n_atoms} atoms, scale={scale:.3f}", flush=True)

    x, masks, truth, theta = zoo.sample(BATCH_MIX, L0, rng,
                                        return_truth=True, return_theta=True)
    rows_out = []
    fig, axes = plt.subplots(len(TYPES), 3, figsize=(12, 3.0 * len(TYPES)), squeeze=False)

    for ti, (name, di, ki, *_) in enumerate(TYPES):
        inst = by_name[f"{name}_{VARIANT}"]
        idx = zoo.instances.index(inst)
        cmap = "twilight" if name in CYCLIC else "viridis"
        rows = np.where(masks[:, idx])[0]
        x_i = x[rows]; m_i = truth[rows, idx]; theta_c = theta[rows, idx, 0]

        inp = torch.tensor(x_i / scale, device=DEV)
        tgt = torch.tensor(m_i / scale, device=DEV)
        with torch.no_grad():
            z, g = model.encode(inp - model.b_dec)
            c = g.unsqueeze(-1) * model.decode_all(z)            # (n,N,d)
            err = ((c - tgt.unsqueeze(1)) ** 2).sum(-1)
            den = (tgt ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
            fvu_a = (err / den).mean(0).cpu().numpy()            # (N,)
        a = int(np.argmin(fvu_a))                                # THE best atom
        r_a = int(ranks[a]); fvu = float(fvu_a[a])
        with torch.no_grad():
            z_a = z[:, a, :r_a].cpu().numpy()
            c_a = (g[:, a].unsqueeze(-1) * model.decode_all(z)[:, a]).cpu().numpy()
        rec_k = (c_a * scale) @ inst.V.T
        tgt_k = m_i @ inst.V.T
        rows_out.append((name, di, ki, r_a, fvu, a))

        # panels: subsample for legibility, shared PCA basis from target
        sel = np.random.default_rng(0).permutation(len(rows))[:NMAX_PLOT]
        mu, basis = (pca2d(tgt_k[sel]) if ki > 1 else (None, None))
        axL, axM, axR = axes[ti]
        scatter_latent(axL, z_a[sel], theta_c[sel], r_a, cmap)
        scatter_space(axM, rec_k[sel], theta_c[sel], ki, cmap, mu, basis)
        scatter_space(axR, tgt_k[sel], theta_c[sel], ki, cmap, mu, basis)
        axL.set_title(f"{name}: best atom rank={r_a}  (d_i={di}, k_i={ki})\nlatent",
                      fontsize=9)
        axM.set_title(f"reconstruction   FVU={fvu:.3f}", fontsize=9)
        axR.set_title("target  m_i", fontsize=9)
        print(f"[{name}] d_i={di} k_i={ki}  best atom #{a} rank={r_a}  FVU={fvu:.3f}",
              flush=True)

    fig.suptitle(f"Best atom per manifold  (trained mixture SAE, R={R}; "
                 f"probed on L0={L0} mixtures)", y=1.003, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_atom_panels.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    lines = [f"# Best atom per manifold — trained mixture SAE (R={R})\n",
             "For one trained mixture checkpoint: the single best atom capturing each",
             "manifold (min variance-unexplained of m_i, probed on L0=4 mixtures).\n",
             "| manifold | d_i | k_i | best-atom rank | FVU (var unexpl.) |",
             "|---|---|---|---|---|"]
    for name, di, ki, r_a, fvu, a in rows_out:
        lines.append(f"| {name} | {di} | {ki} | {r_a} | {fvu:.3f} |")
    (OUT_DIR / "TABLE.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    print(f"\n[done] TABLE.md + best_atom_panels.png -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
