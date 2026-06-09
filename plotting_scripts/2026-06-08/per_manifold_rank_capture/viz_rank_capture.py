"""Per-manifold rank-capture visualization (uses ALREADY-TRAINED atoms,
probed ON THE TRAINING MIXTURE DISTRIBUTION).

For each manifold type we ask, qualitatively: which trained atom best captures it,
and what does capture look like at deficit / matched / surplus rank?

Every plotted point comes from the SAME manifold mixture / sparsity the SAEs were
trained on: we draw one shared batch of real mixtures x = sum_{i in S} m_i (L0=4),
and for a target manifold i use the rows where i is active. The ground-truth
contribution m_i (truth) and intrinsic coord theta_i are known, so:
  * rank every atom by how well its in-mixture contribution c_a = g_a*dec_a(z_a)
    reconstructs m_i (per-atom capture FVU), and
  * for each rank r in {1,2,3,4} pick the globally-best atom of that rank ACROSS
    all saved budget checkpoints (R3..R16) -> cleanest already-trained atom at rank r.
Render, per type, rows = rank (deficit .. matched=k_i .. surplus), columns =
  [ latent z(theta_i) | reconstruction vs target m_i ], both in V_i coordinates.
The seam (closed manifold under-ranked at rank<k_i) shows as a discontinuity in the
latent's theta-coloring; rank=k_i should close it. Latent is a scatter (distractor
manifolds in the mixture add spread off the chart) — honest, in-distribution.

Run (CPU, read-only): source .venv/bin/activate && PYTHONPATH=. \
  python plotting_scripts/2026-06-08/per_manifold_rank_capture/viz_rank_capture.py
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
RANKS = [1, 2, 3, 4]                 # deficit..surplus (rank-8 decoy omitted)
VARIANT = 2                          # representative mid-scale variant per type
L0 = 4                               # training sparsity
BATCH_MIX = 12000                    # shared mixture batch (~1000 rows / manifold)
DEV = "cpu"
CYCLIC = {"circle", "sphere", "torus", "mobius"}   # primary theta is an angle


def load_models():
    models = []
    for b in BUDGETS:
        p = SWEEP / f"manifold_R{b}" / "ckpt.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location=DEV)
        m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], R_target=ck["R"],
                        enc_dims=tuple(ck["enc_dims"])).to(DEV)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append({"R": b, "model": m, "scale": ck["scale"]})
    print(f"[load] {len(models)} manifold checkpoints", flush=True)
    return models


def atom_response(model, x_scaled, atom, rank):
    """-> latent z_a (n, rank), in-mixture contribution c_a (n, d)."""
    with torch.no_grad():
        z, g = model.encode(x_scaled - model.b_dec)
        dec = model.decode_all(z)
        z_a = z[:, atom, :rank].cpu().numpy()
        c_a = (g[:, atom].unsqueeze(-1) * dec[:, atom]).cpu().numpy()
    return z_a, c_a


def pca2d(target):
    mu = target.mean(0)
    _, _, Vt = np.linalg.svd(target - mu, full_matrices=False)
    return mu, Vt[:2]


def scatter_latent(ax, z, theta, rank, cmap):
    if rank == 1:
        ax.scatter(theta, z[:, 0], c=theta, cmap=cmap, s=4)
        ax.set_xlabel("theta_i"); ax.set_ylabel("z")
    else:
        ax.scatter(z[:, 0], z[:, 1], c=theta, cmap=cmap, s=4)
        ax.set_xlabel("z0"); ax.set_ylabel("z1")
        ax.set_aspect("equal", "datalim")


def scatter_recon(ax, tgt, rec, theta, ki, cmap):
    if ki == 1:
        ax.scatter(theta, tgt[:, 0], c="0.8", s=6)
        ax.scatter(theta, rec[:, 0], c=theta, cmap=cmap, s=4)
        ax.set_xlabel("theta_i"); ax.set_ylabel("coord")
    else:
        mu, basis = pca2d(tgt)
        T = (tgt - mu) @ basis.T
        R = (rec - mu) @ basis.T
        ax.scatter(T[:, 0], T[:, 1], c="0.85", s=6)
        ax.scatter(R[:, 0], R[:, 1], c=theta, cmap=cmap, s=4)
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel("pc0" + ("" if ki <= 2 else f" (of {ki})")); ax.set_ylabel("pc1")


def main():
    rng = np.random.default_rng(0)
    zoo = ManifoldZoo(d=256, seed=0)
    by_name = {inst.name: inst for inst in zoo.instances}
    models = load_models()

    # ONE shared mixture batch at training sparsity -> every panel comes from it.
    x, masks, truth, theta = zoo.sample(BATCH_MIX, L0, rng,
                                        return_truth=True, return_theta=True)
    print(f"[mix] {BATCH_MIX} mixtures, L0={L0}; ~{masks.sum(0).mean():.0f} rows/manifold",
          flush=True)
    fvu_table = {}

    for name, di, ki, *_ in TYPES:
        inst = by_name[f"{name}_{VARIANT}"]
        idx = zoo.instances.index(inst)
        cmap = "twilight" if name in CYCLIC else "viridis"
        rows = np.where(masks[:, idx])[0]
        x_i = x[rows]                          # mixtures that contain manifold i
        m_i = truth[rows, idx]                 # ground-truth contribution (n, d)
        theta_c = theta[rows, idx, 0]          # true primary intrinsic coord
        tgt_k = m_i @ inst.V.T                  # target in V_i coords (n, k_i)

        # best trained atom of each rank, across checkpoints, by in-mixture FVU of m_i
        best = {r: (1e9, None, None) for r in RANKS}
        for md in models:
            inp = torch.tensor(x_i / md["scale"], device=DEV)
            tgt = torch.tensor(m_i / md["scale"], device=DEV)
            with torch.no_grad():
                z, g = md["model"].encode(inp - md["model"].b_dec)
                c = g.unsqueeze(-1) * md["model"].decode_all(z)
                err = ((c - tgt.unsqueeze(1)) ** 2).sum(-1)
                den = (tgt ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
                fvu_a = (err / den).mean(0).cpu().numpy()
            ranks = md["model"].ranks.cpu().numpy()
            for r in RANKS:
                ridx = np.where(ranks == r)[0]
                if len(ridx) == 0:
                    continue
                a = ridx[np.argmin(fvu_a[ridx])]
                if fvu_a[a] < best[r][0]:
                    best[r] = (float(fvu_a[a]), md, int(a))

        fig, axes = plt.subplots(len(RANKS), 2, figsize=(8.5, 3.0 * len(RANKS)),
                                 squeeze=False)
        fvu_table[name] = {}
        for ri, r in enumerate(RANKS):
            _, md, atom = best[r]
            axL, axR = axes[ri]
            if md is None:
                axL.set_visible(False); axR.set_visible(False); continue
            z_a, c_a = atom_response(md["model"], torch.tensor(x_i / md["scale"], device=DEV),
                                     atom, r)
            rec_k = (c_a * md["scale"]) @ inst.V.T
            leak = float(np.linalg.norm(c_a * md["scale"] - rec_k @ inst.V, axis=1).mean()
                         / (np.linalg.norm(c_a * md["scale"], axis=1).mean() + 1e-9))
            fvu_g = float(((m_i / md["scale"] - c_a) ** 2).sum(1).mean()
                          / ((m_i / md["scale"]) ** 2).sum(1).mean())
            fvu_table[name][r] = dict(fvu=fvu_g, leak=leak, src_R=md["R"])

            scatter_latent(axL, z_a, theta_c, r, cmap)
            scatter_recon(axR, tgt_k, rec_k, theta_c, ki, cmap)
            tag = " <= k_i (seamless)" if r == ki else (
                " (deficit)" if r < ki else " (surplus)")
            axL.set_title(f"rank {r}{tag}   latent  (src R={md['R']})", fontsize=9)
            axR.set_title(f"recon vs target   FVU={fvu_g:.3f}  leak={leak:.2f}", fontsize=9)
        fig.suptitle(f"{name}   d_i={di}  k_i={ki}   "
                     f"(rank={ki} = seamless capture; gray = target m_i; "
                     f"all points from L0={L0} mixtures)", y=1.005, fontsize=11)
        fig.tight_layout()
        out = OUT_DIR / f"capture_{name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[{name}] " + "  ".join(
            f"r{r}:FVU={fvu_table[name].get(r, {}).get('fvu', float('nan')):.3f}"
            for r in RANKS) + f"  -> {out.name}", flush=True)

    lines = ["# Per-manifold rank-capture (best trained atom per rank, on L0=4 mixtures)\n",
             "Capture-FVU of the single best already-trained atom of each rank, probing",
             "each manifold within real mixtures at training sparsity (L0=4). The atom is",
             "chosen across all budget checkpoints (R3..R16). rank=k_i is seamless capture.\n",
             "| type | d_i | k_i | " + " | ".join(f"rank {r}" for r in RANKS) + " |",
             "|---|---|---|" + "---|" * len(RANKS)]
    for name, di, ki, *_ in TYPES:
        cells = []
        for r in RANKS:
            e = fvu_table.get(name, {}).get(r)
            cells.append(f"{e['fvu']:.3f}" + ("*" if r == ki else "") if e else "-")
        lines.append(f"| {name} | {di} | {ki} | " + " | ".join(cells) + " |")
    lines.append("\n(* = rank equals embedding dim k_i, the seamless-capture rank)")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[done] wrote {len(TYPES)} figures + SUMMARY.md to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
