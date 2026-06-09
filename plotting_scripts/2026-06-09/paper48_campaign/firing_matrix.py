"""Atom firing-fraction matrix for the winner (K8/lr3e-4/lam0.003/1x): rows = 48 manifolds
(grouped by type), cols = 64 atoms. value = fraction of that manifold's ISOLATED samples on
which the atom fires. Reveals: dedicated atoms (one bright cell/row), spanning (>1 bright/row),
and ALWAYS-ON atoms (full bright columns -- the rank-0 bias atoms 29/51) + dead atoms (empty cols).
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from manifold_ae.manifold_sae import ManifoldSAE
from manifold_ae.manifold_zoo import ManifoldZoo

RUN = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-09_paper48_stage2_lr3e-4/k8_lam0.003_rev1x")
OUT = Path(__file__).parent
DEV = "cpu"
ck = torch.load(RUN / "ckpt.pt", map_location=DEV)
m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], enc_dims=tuple(ck["enc_dims"]),
                jump_eps=ck["jump_eps"], learn_rank=ck["learn_rank"]).to(DEV); m.load_state_dict(ck["state_dict"]); m.eval()
zoo = ManifoldZoo(d=256, seed=0, variants_per_type=6)
ranks = m.atom_ranks().round().long().tolist()

rng = np.random.default_rng(0)
F = np.zeros((zoo.n_inst, m.n_atoms))                    # firing fraction [manifold, atom]
for i, inst in enumerate(zoo.instances):
    _, amb = inst.sample_full(1500, rng)
    with torch.no_grad():
        _, _, _, active, _, _ = m.forward_jump(torch.tensor(amb, device=DEV) / ck["scale"])
    F[i] = active.float().mean(0).cpu().numpy()

col_mean = F.mean(0)                                     # how broadly each atom fires across manifolds
always_on = np.where(col_mean > 0.8)[0]
dead = np.where(F.max(0) < 0.05)[0]
print(f"always-on atoms (fire on >80% of manifolds): {[(int(a), 'r'+str(ranks[a])) for a in always_on]}")
print(f"dead atoms (fire on <5% of all): {len(dead)}")
# column order: owner manifold (argmax firing) among non-always-on/dead, then always-on, then dead
normal = [a for a in range(m.n_atoms) if a not in always_on and a not in dead]
normal.sort(key=lambda a: F[:, a].argmax())
order = normal + list(always_on) + list(dead)

fig, ax = plt.subplots(figsize=(15, 9))
im = ax.imshow(F[:, order], aspect="auto", cmap="magma", vmin=0, vmax=1)
ax.set_yticks(range(zoo.n_inst)); ax.set_yticklabels([inst.name for inst in zoo.instances], fontsize=5)
ax.set_xlabel(f"atom (reordered: dedicated/spanning | {len(always_on)} always-on | {len(dead)} dead)")
ax.set_ylabel("manifold instance (grouped by type)")
# mark the boundary between normal and always-on/dead
ax.axvline(len(normal) - 0.5, color="cyan", lw=1.5, ls="--")
ax.axvline(len(normal) + len(always_on) - 0.5, color="red", lw=1.5, ls="--")
ax.set_title(f"{RUN.name}: atom firing-fraction (rows=48 manifolds, cols=64 atoms). "
             f"cyan|=always-on bias atoms, red|=dead")
fig.colorbar(im, label="firing fraction"); fig.tight_layout()
fig.savefig(OUT / "firing_matrix.png", dpi=140); plt.close(fig)
print(f"wrote {OUT/'firing_matrix.png'}")
