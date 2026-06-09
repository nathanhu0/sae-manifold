"""Hardened bridge eval: does a manifold-SAE atom genuinely CAPTURE a concept,
and does it beat the linear SAE?

For each cached concept, on the manifold-SAE (real 5M) and linear-SAE (real 5M):
  * candidate units = those firing on >= FIRE_THR of the concept's tokens
    AND selective (concept firing rate > background C4 firing rate)
  * topology recovery: best candidate's coordinate vs known label
    (circular corr for cyclic, |Spearman| for ordinal), with:
      - null = label-shuffled recovery (chance level)
      - margin = best minus second-best candidate (decisive single-unit capture?)
  * 3-panel geometric test for the manifold's best atom:
      true manifold (PCA of acts) | atom latent z | atom contribution gate*dec(z)
      (contribution projected on the SAME PCA basis as the true acts)
  * linear baseline: best single feature's recovery (one linear feature cannot
    represent a cyclic loop -> exposes shattering)

Run (jag): ebatch bridge_hard slconf40s \
  "PYTHONUNBUFFERED=1 PYTHONPATH=. python plotting_scripts/2026-06-08/bridge_eval_hardened/bridge_hard.py"
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
from background import load_background
from manifold_ae.manifold_sae import ManifoldSAE
from saes import BatchTopKSAE

OUT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAN_CKPT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-08_real_manifold_5M/ckpt.pt"
LIN_CKPT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-08_real_linear_5M/ckpt.pt"
FIRE_THR = 0.90
BG_N = 4000
CONCEPTS = {  # label key, cyclic?, cmap, period
    "days":        ("day_idx",    True,  "hsv",      7),
    "colors":      ("hue",        True,  "hsv",      1.0),
    "years":       ("year",       False, "viridis",  None),
    "temperature": ("fahrenheit", False, "coolwarm", None),
}


def circ_corr(th, ph):
    thm = np.angle(np.mean(np.exp(1j * th))); phm = np.angle(np.mean(np.exp(1j * ph)))
    s1, s2 = np.sin(th - thm), np.sin(ph - phm)
    d = np.sqrt(np.sum(s1 ** 2) * np.sum(s2 ** 2))
    return float(np.sum(s1 * s2) / d) if d > 0 else 0.0


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return abs(float(np.corrcoef(ra, rb)[0, 1]))


def recover(coord, labs, cyclic, period):
    """Topology-recovery score of a unit's coordinate vs label."""
    if coord.ndim == 1:
        coord = coord[:, None]
    best = 0.0
    if cyclic and coord.shape[1] >= 2:
        th = np.arctan2(coord[:, 1], coord[:, 0])
        best = abs(circ_corr(th, labs * (2 * np.pi / period)))
    for k in range(coord.shape[1]):
        best = max(best, spearman(coord[:, k], labs))
    return best


@torch.no_grad()
def man_forward(model, x, bs=64):
    zs, gs = [], []
    for i in range(0, len(x), bs):
        _, z, g = model(x[i:i + bs]); zs.append(z); gs.append(g)
    return torch.cat(zs), torch.cat(gs)


@torch.no_grad()
def man_contrib(model, z, gate, a, bs=64):
    out = []
    for i in range(0, len(z), bs):
        dec = model.decode_all(z[i:i + bs])           # (b, N, d)
        out.append((gate[i:i + bs, a].unsqueeze(-1) * dec[:, a, :]))
    return torch.cat(out).cpu().numpy()


def pca2(M, basis=None):
    Mc = M - M.mean(0)
    if basis is None:
        _, _, Vt = np.linalg.svd(Mc, full_matrices=False); basis = Vt[:2]
    return Mc @ basis.T, basis


def main():
    mk = torch.load(MAN_CKPT, map_location="cpu", weights_only=False)
    M = ManifoldSAE(d_model=4096, rank_dist=mk["rank_dist"], R_target=mk["R_target"],
                    enc_dims=tuple(mk["enc_dims"])).to(DEV)
    M.load_state_dict(mk["state_dict"]); M.eval()
    Mmean, Mscale, ranks = mk["mean"].to(DEV), mk["scale"], M.ranks.cpu().numpy()

    lk = torch.load(LIN_CKPT, map_location="cpu", weights_only=False)
    L = BatchTopKSAE(d_in=4096, d_sae=lk["d_sae"], k=lk["k"]).to(DEV)
    L.load_state_dict(lk["state_dict"]); L.eval(); L.k = lk["k"]
    Lmean, Lscale = lk["mean"].to(DEV), lk["scale"]

    # background firing rates (selectivity baseline)
    acts = load_background(5_000_000)
    bg = torch.tensor(np.asarray(acts[np.random.default_rng(0).integers(0, len(acts), BG_N)],
                                 dtype=np.float32), device=DEV)
    _, gbg = man_forward(M, (bg - Mmean) / Mscale)
    man_bg_fire = (gbg > 0).float().mean(0).cpu().numpy()
    with torch.no_grad():
        cbg = torch.cat([L.encode(((bg - Lmean) / Lscale)[i:i + 512]) for i in range(0, BG_N, 512)])
    lin_bg_fire = (cbg > 0).float().mean(0).cpu().numpy()

    results = {}
    fig, axes = plt.subplots(len(CONCEPTS), 4, figsize=(15, 3.4 * len(CONCEPTS)))
    for row, (cname, (lkey, cyclic, cmap, period)) in enumerate(CONCEPTS.items()):
        data = load_manifold_data(cname, filter_outliers=True, n_std=3.0)
        Xnp = data["activations"].float().numpy()
        labs = np.array([l[lkey] for l in data["labels"]], dtype=float)
        X = torch.tensor(Xnp, device=DEV)

        # ---- manifold ----
        xM = (X - Mmean) / Mscale
        z, gate = man_forward(M, xM)
        fire = (gate > 0).float().mean(0).cpu().numpy()
        zc = z.cpu().numpy()
        cand = [a for a in range(M.n_atoms)
                if fire[a] >= FIRE_THR and fire[a] > man_bg_fire[a] + 0.05]
        scored = sorted(((recover(zc[:, a, :int(ranks[a])], labs, cyclic, period), a)
                         for a in cand), reverse=True)
        if scored:
            best_corr, best = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            null = np.mean([recover(zc[:, best, :int(ranks[best])],
                                    np.random.default_rng(s).permutation(labs), cyclic, period)
                            for s in range(5)])
        else:
            best, best_corr, second, null = -1, 0.0, 0.0, 0.0

        # ---- linear ----
        xL = (X - Lmean) / Lscale
        with torch.no_grad():
            codes = torch.cat([L.encode(xL[i:i + 512]) for i in range(0, len(xL), 512)])
        cnp = codes.cpu().numpy()
        lfire = (cnp > 0).mean(0)
        lcand = [f for f in range(L.d_sae) if lfire[f] >= FIRE_THR and lfire[f] > lin_bg_fire[f] + 0.05]
        lscored = sorted(((spearman(cnp[:, f], labs), f) for f in lcand), reverse=True)
        lin_corr, lin_feat = (lscored[0] if lscored else (0.0, -1))

        results[cname] = dict(
            manifold=dict(best_atom=int(best), rank=int(ranks[best]) if best >= 0 else 0,
                          fire=float(fire[best]) if best >= 0 else 0.0,
                          recovery=float(best_corr), second=float(second),
                          margin=float(best_corr - second), null=float(null),
                          n_candidates=len(cand)),
            linear=dict(best_feat=int(lin_feat), recovery=float(lin_corr), n_candidates=len(lcand)))
        print(f"{cname:12s} MAN atom={best} rank={int(ranks[best]) if best>=0 else 0} "
              f"fire={fire[best] if best>=0 else 0:.2f} recov={best_corr:.2f} "
              f"(2nd={second:.2f} margin={best_corr-second:.2f} null={null:.2f} ncand={len(cand)})  | "
              f"LIN feat={lin_feat} recov={lin_corr:.2f} ncand={len(lcand)}", flush=True)

        # ---- 3-panel + linear contribution ----
        A, basis = pca2(Xnp)                                   # true manifold
        axes[row, 0].scatter(A[:, 0], A[:, 1], c=labs, cmap=cmap, s=12)
        axes[row, 0].set_ylabel(cname, fontsize=11)
        axes[row, 0].set_title("true manifold (PCA of acts)" if row == 0 else "", fontsize=9)
        if best >= 0:
            za = zc[:, best, :int(ranks[best])]
            B = pca2(za)[0] if za.shape[1] >= 2 else np.stack([za[:, 0], np.zeros_like(za[:, 0])], 1)
            axes[row, 1].scatter(B[:, 0], B[:, 1], c=labs, cmap=cmap, s=12)
            contrib = man_contrib(M, z, gate, best)
            C = (contrib - Xnp.mean(0)) @ basis.T              # same basis as true
            axes[row, 2].scatter(C[:, 0], C[:, 1], c=labs, cmap=cmap, s=12)
        axes[row, 1].set_title(f"manifold latent z (atom {best}, r{int(ranks[best]) if best>=0 else 0}, "
                               f"recov {best_corr:.2f})", fontsize=9)
        axes[row, 2].set_title(f"atom contribution gate*dec(z)", fontsize=9)
        if lin_feat >= 0:
            lc = (cnp[:, lin_feat:lin_feat + 1] * L.decoder.weight[:, lin_feat].detach().cpu().numpy()[None, :])
            D = (lc - Xnp.mean(0)) @ basis.T
            axes[row, 3].scatter(D[:, 0], D[:, 1], c=labs, cmap=cmap, s=12)
        axes[row, 3].set_title(f"LINEAR best feat {lin_feat} contrib (recov {lin_corr:.2f})", fontsize=9)
        for c in range(4):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    fig.suptitle("Hardened bridge eval: true | manifold latent | manifold contribution | linear contribution",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "bridge_hard.png", dpi=140, bbox_inches="tight")
    json.dump(results, open(OUT / "bridge_hard_metrics.json", "w"), indent=2)
    print("[done] wrote bridge_hard.png, bridge_hard_metrics.json", flush=True)


if __name__ == "__main__":
    main()
