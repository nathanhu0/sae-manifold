"""Per-canonical-manifold fit visualization across the budget sweep.

Trains manifold-SAE and linear-SAE on the toy mixture at budgets {6,8,12,16}, then
for each canonical manifold feeds PURE single-manifold samples through them and
draws, in the original-data style:

  one figure per manifold; rows = budget; columns =
    [ original manifold | manifold-SAE latent z | manifold-SAE decoded contribution
      | linear-SAE reconstruction (how its features span it) ]
  all colored by the manifold's intrinsic parameter.

So you can SEE, as the manifold budget tightens, what atom captures each manifold,
what its latent coordinate looks like, what it decodes to, and how linear spans it.

Run (jag): ebatch viz_toy_fits slconf40s \
  "PYTHONUNBUFFERED=1 PYTHONPATH=. python plotting_scripts/2026-06-08/toy_fit_viz/viz_toy_fits.py"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_ae.manifold_zoo import ManifoldZoo
from manifold_ae.manifold_sae import ManifoldSAE
from saes import BatchTopKSAE

OUT = Path(__file__).parent
CKPT_DIR = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-08_toy_fit_models")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUDGETS = [6, 8, 12, 16]
POOL = {1: 128, 2: 64, 4: 32, 8: 16}
D, L0, STEPS, WARMUP, LR, BATCH = 256, 4, 20000, 3000, 1e-3, 2048
torch.manual_seed(0)

# representative instance per type + how to color by intrinsic param (from raw coords)
PARAM = {
    "circle":     lambda r: np.arctan2(r[:, 1], r[:, 0]),
    "sphere":     lambda r: r[:, 2],
    "torus":      lambda r: np.arctan2(r[:, 1], r[:, 0]),
    "mobius":     lambda r: np.arctan2(r[:, 1], r[:, 0]),
    "swiss_roll": lambda r: np.sqrt(r[:, 0] ** 2 + r[:, 2] ** 2),
    "helix":      lambda r: r[:, 2],
    "segment":    lambda r: r[:, 0],
}
SHOW = ["circle", "sphere", "helix", "swiss_roll", "torus", "segment"]


def train_manifold(zoo, scale, R):
    m = ManifoldSAE(d_model=D, rank_dist=POOL, R_target=R, enc_dims=(128, 64, 32)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR); rng = np.random.default_rng(1)
    lr_ratio = np.log(R / m.full_R)
    for step in range(STEPS):
        Rcur = m.full_R * np.exp(lr_ratio * min(1.0, step / WARMUP))
        x, _ = zoo.sample(BATCH, L0, rng)
        xt = torch.tensor(x, device=DEV) / scale
        xh, _, gate = m(xt, R=Rcur)
        loss = ((xh - xt) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); return m


def train_linear(zoo, scale, k):
    sae = BatchTopKSAE(d_in=D, d_sae=2048, k=2048).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=LR); rng = np.random.default_rng(1)
    lr_ratio = np.log(k / 2048)
    for step in range(STEPS):
        sae.k = int(round(2048 * np.exp(lr_ratio * min(1.0, step / WARMUP))))
        x, _ = zoo.sample(BATCH, L0, rng)
        xt = torch.tensor(x, device=DEV) / scale
        xh, _ = sae(xt); loss = ((xh - xt) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sae.eval(); sae.k = k; return sae


def get_manifold(zoo, scale, R):
    """Load saved manifold ckpt if present, else train + SAVE (never retrain to plot)."""
    p = CKPT_DIR / f"manifold_R{R}.pt"
    if p.exists():
        m = ManifoldSAE(d_model=D, rank_dist=POOL, R_target=R, enc_dims=(128, 64, 32)).to(DEV)
        m.load_state_dict(torch.load(p, map_location=DEV)["state_dict"]); m.eval()
        print(f"[load] manifold R={R}", flush=True); return m
    m = train_manifold(zoo, scale, R)
    torch.save({"state_dict": m.state_dict(), "R": R, "pool": POOL,
                "enc_dims": (128, 64, 32), "scale": scale}, p)
    print(f"[train+save] manifold R={R}", flush=True); return m


def get_linear(zoo, scale, k):
    p = CKPT_DIR / f"linear_k{k}.pt"
    if p.exists():
        sae = BatchTopKSAE(d_in=D, d_sae=2048, k=k).to(DEV)
        sae.load_state_dict(torch.load(p, map_location=DEV)["state_dict"]); sae.eval(); sae.k = k
        print(f"[load] linear k={k}", flush=True); return sae
    sae = train_linear(zoo, scale, k)
    torch.save({"state_dict": sae.state_dict(), "k": k, "d_sae": 2048, "scale": scale}, p)
    print(f"[train+save] linear k={k}", flush=True); return sae


@torch.no_grad()
def man_fit(m, xt, bs=256):
    zs, gs = [], []
    for i in range(0, len(xt), bs):
        _, z, g = m(xt[i:i + bs]); zs.append(z); gs.append(g)
    z, gate = torch.cat(zs), torch.cat(gs)
    a = int((gate > 0).float().mean(0).argmax())          # dominant firing atom
    r = int(m.ranks[a])
    contrib = []
    for i in range(0, len(z), bs):
        dec = m.decode_all(z[i:i + bs])
        contrib.append(gate[i:i + bs, a].unsqueeze(-1) * dec[:, a, :])
    return a, r, z[:, a, :r].cpu().numpy(), torch.cat(contrib).cpu().numpy(), \
        int((gate > 0).sum(1).float().mean())


@torch.no_grad()
def lin_fit(sae, xt, bs=256):
    rec, n = [], []
    for i in range(0, len(xt), bs):
        c = sae.encode(xt[i:i + bs]); rec.append(sae.decode(c)); n.append((c > 0).sum(1))
    return torch.cat(rec).cpu().numpy(), float(torch.cat(n).float().mean())


def pca2(M, basis=None):
    Mc = M - M.mean(0)
    if basis is None:
        _, _, Vt = np.linalg.svd(Mc, full_matrices=False); basis = Vt[:2]
    return Mc @ basis.T, basis


def main():
    zoo = ManifoldZoo(d=D, seed=0)
    x0, _ = zoo.sample(8192, L0, np.random.default_rng(7))
    scale = float(np.sqrt((x0 ** 2).mean()))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("[models] load-or-train+save at budgets", BUDGETS, flush=True)
    M = {R: get_manifold(zoo, scale, R) for R in BUDGETS}
    Ln = {k: get_linear(zoo, scale, k) for k in BUDGETS}
    inst_of = {t: next(i for i in zoo.instances if i.type == t) for t in SHOW}

    for t in SHOW:
        inst = inst_of[t]
        rng = np.random.default_rng(42)
        raw = inst._fn(1000, inst.params, rng)
        gt = ((raw - inst.mu) / inst.sigma).astype(np.float32)
        param = PARAM[t](raw)
        xamb = gt @ inst.V
        xt = torch.tensor(xamb, device=DEV) / scale
        cmap = "hsv" if t in ("circle", "torus", "mobius") else "viridis"
        A, basis = pca2(xamb)

        fig, ax = plt.subplots(len(BUDGETS), 4, figsize=(15, 3.4 * len(BUDGETS)))
        for row, B in enumerate(BUDGETS):
            a, r, lat, contrib, natoms = man_fit(M[B], xt)
            lrec, nfeat = lin_fit(Ln[B], xt)
            ax[row, 0].scatter(A[:, 0], A[:, 1], c=param, cmap=cmap, s=10)
            latp = pca2(lat)[0] if lat.shape[1] >= 2 else np.stack([lat[:, 0], 0 * lat[:, 0]], 1)
            ax[row, 1].scatter(latp[:, 0], latp[:, 1], c=param, cmap=cmap, s=10)
            C = (contrib - xamb.mean(0)) @ basis.T
            ax[row, 2].scatter(C[:, 0], C[:, 1], c=param, cmap=cmap, s=10)
            Lr = (lrec - xamb.mean(0)) @ basis.T
            ax[row, 3].scatter(Lr[:, 0], Lr[:, 1], c=param, cmap=cmap, s=10)
            ax[row, 0].set_ylabel(f"budget {B}", fontsize=11)
            ax[row, 1].set_title(f"manifold latent (atom {a}, rank {r}, {natoms} atoms/tok)" if row == 0 else
                                 f"atom {a} r{r}", fontsize=8)
            ax[row, 2].set_title("manifold decoded contribution" if row == 0 else "", fontsize=8)
            ax[row, 3].set_title(f"linear recon ({nfeat:.0f} feats)" if row == 0 else
                                 f"{nfeat:.0f} feats", fontsize=8)
            if row == 0:
                ax[row, 0].set_title(f"{t}: original (d={inst.di}, k={inst.ki})", fontsize=9)
            for c in range(4):
                ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
        fig.suptitle(f"{t}: original | manifold latent | manifold decoded | linear span  (rows = budget)",
                     y=1.005)
        fig.tight_layout(); fig.savefig(OUT / f"fit_{t}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[done] fit_{t}.png", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
