"""Atom co-occurrence / co-activation structure for a trained manifold-SAE ckpt.

On real mixtures (independent p_active), two atoms that jointly code ONE manifold (a split)
fire iff that manifold is present -> correlation ~1; atoms serving DIFFERENT manifolds co-fire
only by chance -> correlation ~0. So the live-atom correlation matrix has block structure:
one block per manifold, block size = how many atoms it's split across. High OFF-block
correlation = cross-manifold entanglement (the real pathology).

We assign each live atom an OWNER manifold from isolated firing (the manifold it fires on most),
reorder by owner so blocks are legible, and draw manifold boundaries. Read-only on the ckpt.

Run: source .venv/bin/activate && PYTHONPATH=. python plotting_scripts/2026-06-09/atom_cooccurrence/cooccurrence.py [--ckpt <ckpt.pt>]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifold_ae.manifold_zoo import ManifoldZoo
from manifold_ae.manifold_sae import ManifoldSAE

OUT_DIR = Path(__file__).parent
DEFAULT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-09_opt_sweep/lr1e-3_cosine_preact3e-4_lamramp/ckpt.pt"


def build(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ManifoldSAE(d_model=256, rank_dist=c["pool"], enc_dims=tuple(c["enc_dims"]),
                    jump_eps=c.get("jump_eps", 2.0))
    m.load_state_dict(c["state_dict"], strict=False); m.eval()
    zoo = ManifoldZoo(d=256, seed=0, variants_per_type=c["variants_per_type"])
    return m, zoo, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT)
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    tag = Path(a.ckpt).parent.name
    m, zoo, c = build(a.ckpt)
    scale, p_active = c["scale"], c["p_active"]
    rng = np.random.default_rng(0)

    # mixture activations (n x N) -> live atoms + correlation
    x, _ = zoo.sample(a.n, None, rng, p_active=p_active)
    with torch.no_grad():
        _, _, _, active, _, _ = m.forward_jump(torch.tensor(x, dtype=torch.float32) / scale)
    A = active.cpu().numpy()                                   # (n, N) binary
    rate = A.mean(0)                                           # per-atom firing rate
    live = np.where((rate > 0.002) & (rate < 0.998))[0]        # drop dead + always-on (degenerate corr)
    Cfull = np.corrcoef(A[:, live].T)
    Cfull = np.nan_to_num(Cfull)

    # owner manifold per atom: the manifold it fires on most (isolated samples)
    own_frac = np.zeros((len(zoo.instances), m.n_atoms))
    for i, inst in enumerate(zoo.instances):
        _, amb = inst.sample_full(2000, rng)
        with torch.no_grad():
            _, _, _, act_i, _, _ = m.forward_jump(torch.tensor(amb, dtype=torch.float32) / scale)
        own_frac[i] = act_i.cpu().numpy().mean(0)
    owner = own_frac.argmax(0)                                 # (N,) best manifold per atom
    owner_max = own_frac.max(0)
    names = [inst.name for inst in zoo.instances]

    # order live atoms by owner manifold (group blocks), then by firing rate within
    order = sorted(range(len(live)), key=lambda k: (owner[live[k]], -rate[live[k]]))
    li = live[order]
    C = Cfull[np.ix_(order, order)]

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="activation correlation (phi)")
    # manifold block boundaries + center labels
    bounds, labels, start = [], [], 0
    for k in range(1, len(li) + 1):
        if k == len(li) or owner[li[k]] != owner[li[k - 1]]:
            bounds.append(k)
            labels.append((start + k - 1) / 2, )
            mid = (start + k - 1) / 2
            ax.text(-1.4, mid, names[owner[li[start]]], ha="right", va="center", fontsize=7)
            ax.text(mid, len(li) + 0.8, f"{k - start}", ha="center", va="top", fontsize=7, color="gray")
            start = k
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="k", lw=0.6, alpha=0.5); ax.axvline(b - 0.5, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Atom co-activation correlation on mixtures — {tag}\n"
                 f"(reordered by owner manifold; block size above = atoms/manifold; "
                 f"{len(live)} live of {m.n_atoms})", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / f"cooccurrence_{tag}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")

    # ---- text summary ----
    block = {}
    for k in range(len(li)):
        block.setdefault(int(owner[li[k]]), []).append(int(li[k]))
    print(f"\n[{tag}] live atoms {len(live)} / {m.n_atoms}  (dead/always-on: {m.n_atoms - len(live)})")
    print("per-manifold owned atoms (atom@firing-rate), within-block mean corr:")
    summary = {}
    for i, inst in enumerate(zoo.instances):
        atoms = block.get(i, [])
        if not atoms:
            print(f"   {inst.name:13s} -- no owned live atom"); summary[inst.name] = dict(n=0); continue
        idx = [list(li).index(at) for at in atoms]
        sub = C[np.ix_(idx, idx)]
        wc = float((sub.sum() - len(atoms)) / max(1, len(atoms) ** 2 - len(atoms)))   # off-diag mean
        astr = " ".join(f"{at}@{rate[at]*100:.0f}%" for at in atoms)
        print(f"   {inst.name:13s} {len(atoms)} atoms: {astr}   within-corr={wc:.2f}")
        summary[inst.name] = dict(n=len(atoms), atoms=atoms, within_corr=wc)
    # cross-manifold entanglement: live pairs with different owners but high corr
    print("\ncross-manifold pairs |corr|>0.4 (entanglement):")
    found = 0
    for a_ in range(len(li)):
        for b_ in range(a_ + 1, len(li)):
            if owner[li[a_]] != owner[li[b_]] and abs(C[a_, b_]) > 0.4:
                print(f"   {names[owner[li[a_]]]}:atom{li[a_]}  ~  {names[owner[li[b_]]]}:atom{li[b_]}  corr={C[a_, b_]:.2f}")
                found += 1
    if not found:
        print("   none (off-block correlations all <0.4 -> manifolds cleanly separated)")
    json.dump(summary, open(OUT_DIR / f"cooccurrence_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
