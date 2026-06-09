"""Bridge eval: does the C4-trained manifold-SAE represent known concept manifolds
with FEW atoms whose coordinates recover the concept's topology?

Loads the real 5M manifold-SAE checkpoint, encodes each cached concept manifold
(days/years/colors/temperature), and per concept reports:
  * reconstruction FVU on these (OOD-ish) concept activations
  * atoms/token + how concentrated firing is (a few atoms => de-shatter)
  * best topology recovery: for the top-firing atoms, does the atom's coordinate
    recover the known label? (circular corr for cyclic labels; Spearman for ordinal)

Run (SLURM jag): ebatch bridge_eval slconf40s \
  "PYTHONUNBUFFERED=1 PYTHONPATH=. python plotting_scripts/2026-06-08/bridge_eval_real_manifold/bridge_eval.py"
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_manifold_data
from manifold_ae.manifold_sae import ManifoldSAE

OUT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-08_real_manifold_5M/ckpt.pt"

# concept -> (label key, is the label cyclic, cmap, period if cyclic)
CONCEPTS = {
    "days":        ("day_idx",    True,  "hsv",     7),
    "colors":      ("hue",        True,  "hsv",     1.0),
    "years":       ("year",       False, "viridis", None),
    "temperature": ("fahrenheit", False, "coolwarm", None),
}


def circ_corr(th, ph):
    thm = np.angle(np.mean(np.exp(1j * th)))
    phm = np.angle(np.mean(np.exp(1j * ph)))
    s1, s2 = np.sin(th - thm), np.sin(ph - phm)
    d = np.sqrt(np.sum(s1 ** 2) * np.sum(s2 ** 2))
    return float(np.sum(s1 * s2) / d) if d > 0 else 0.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ManifoldSAE(d_model=4096, rank_dist=ck["rank_dist"],
                        R_target=ck["R_target"], enc_dims=tuple(ck["enc_dims"])).to(DEV)
    model.load_state_dict(ck["state_dict"]); model.eval()
    mean = ck["mean"].to(DEV); scale = ck["scale"]
    ranks = model.ranks.cpu().numpy()
    print(f"[ckpt] {model.n_atoms} atoms, R_target={model.R_target}, scale={scale:.3f}", flush=True)

    results = {}
    fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(4.2 * len(CONCEPTS), 3.8))
    for ax, (cname, (lkey, cyclic, cmap, period)) in zip(axes, CONCEPTS.items()):
        data = load_manifold_data(cname, filter_outliers=True, n_std=3.0)
        X = torch.tensor(data["activations"].float().numpy(), device=DEV)
        labs = np.array([l[lkey] for l in data["labels"]], dtype=float)
        x = (X - mean) / scale
        # batch the forward: decode_all is (B, N_atoms, d_model) and blows up if B is large
        zs, gs, xhs = [], [], []
        with torch.no_grad():
            for i in range(0, x.shape[0], 64):
                xh_b, z_b, g_b = model(x[i:i + 64])
                zs.append(z_b); gs.append(g_b); xhs.append(xh_b)
        z = torch.cat(zs); gate = torch.cat(gs); x_hat = torch.cat(xhs)
        fvu = (((x_hat - x) ** 2).sum() / (x ** 2).sum()).item()
        active = (gate > 0).cpu().numpy()        # (B,N)
        atoms_tok = active.sum(1).mean()
        fire = active.mean(0)                     # firing rate per atom
        top = np.argsort(fire)[::-1][:12]         # most-firing atoms
        top_frac = float(fire[top[0]])            # firing fraction of the single top atom

        # best topology recovery among top firing atoms
        zc = z.cpu().numpy()
        best = dict(corr=0.0, atom=-1, kind="")
        for a in top:
            r = int(ranks[a])
            za = zc[:, a, :r]
            if r >= 2:
                th = np.arctan2(za[:, 1], za[:, 0])
                if cyclic:
                    ph = labs * (2 * np.pi / period)
                    c = abs(circ_corr(th, ph))
                else:
                    c = abs(spearman(th, labs))   # angle vs ordinal (loose)
            else:
                c = abs(spearman(za[:, 0], labs))
            # also try a raw coord-vs-label Spearman (captures ordinal in any dim)
            c2 = max(abs(spearman(za[:, k], labs)) for k in range(r))
            cc = max(c, c2)
            if cc > best["corr"]:
                best = dict(corr=cc, atom=int(a), kind=f"rank{r}")

        results[cname] = dict(fvu=fvu, atoms_per_token=float(atoms_tok),
                              top_atom_fire_frac=top_frac, best_topology_corr=best["corr"],
                              best_atom=best["atom"], best_atom_kind=best["kind"], N=len(labs))
        print(f"{cname:12s} fvu={fvu:.3f} atoms/tok={atoms_tok:4.1f} "
              f"top_atom_fires={top_frac:.2f} best_topo_corr={best['corr']:.2f} "
              f"(atom {best['atom']}, {best['kind']})", flush=True)

        # plot the best atom's 2-D coordinate (PCA if rank>2), colored by label
        a = best["atom"]; r = int(ranks[a]) if a >= 0 else 1
        za = zc[:, a, :max(r, 1)]
        if za.shape[1] >= 2:
            zac = za - za.mean(0)
            _, _, Vt = np.linalg.svd(zac, full_matrices=False)
            p = zac @ Vt[:2].T
        else:
            p = np.stack([za[:, 0], np.zeros_like(za[:, 0])], 1)
        sc = ax.scatter(p[:, 0], p[:, 1], c=labs, cmap=cmap, s=14)
        ax.set_title(f"{cname}: atom {a} ({best['kind']})\ntopo corr={best['corr']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, label=lkey)
    fig.suptitle("Bridge eval: top concept atom's coordinate vs known label (5M manifold-SAE)", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "bridge_eval.png", dpi=140, bbox_inches="tight")
    json.dump(results, open(OUT / "bridge_metrics.json", "w"), indent=2)
    print("[done] wrote bridge_eval.png, bridge_metrics.json", flush=True)


if __name__ == "__main__":
    main()
