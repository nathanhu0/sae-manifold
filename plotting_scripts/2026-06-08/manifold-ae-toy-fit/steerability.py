"""How bad is 'pretend every manifold is open'? (corrected metrics)

Plain flat latent at embedding dim (no topology). Measure steerability with
FAIR probes that sample each latent's NATURAL domain, not the bounding box:
  - local : perturb an encoded point by 0.3*std, decode -> dist to manifold
            (realistic 'nudge an existing point' steering).
  - interp: linear interpolation between two encoded points, decode along it.
  - gen   : sample the latent's natural generative distribution and decode --
            for flat = a Gaussian fit to the encoded support; for structured =
            the true topological domain (S^1/S^2/T^k). Closed manifolds under a
            flat latent fill the off-manifold slack (gen large); structured
            stays on (gen ~0). gen_proj = gen after one dec(enc(.)) projection,
            showing the slack is largely fixable.
Distances are nearest-neighbour to a dense clean reference; manifold RMS scale 1.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/juice2/u/nathu/sae-manifold")
from manifold_ae.toy_manifolds import make_manifold
from manifold_ae.train import TrainConfig, train_one

OUT = Path(__file__).parent
EMB = {"helix": 1, "swiss_roll": 2, "s_curve": 2, "wavy_sheet": 2,
       "circle": 2, "sphere": 3, "torus": 3}
OPEN = {"helix", "swiss_roll", "s_curve", "wavy_sheet"}
CLOSED = ["circle", "sphere", "torus"]


def fit(name, mode, m):
    data = make_manifold(name, n=4000, regime="native", noise_std=0.0, seed=0)
    cfg = TrainConfig(epochs=800, latent_mode=mode, latent_dim=m, width=256,
                      depth=3, lr=3e-3, seed=0)
    model = train_one(data, cfg).model
    dev = next(model.parameters()).device
    with torch.no_grad():
        z = model.encode(torch.as_tensor(data.X, device=dev)).cpu().numpy()
    return data, model, z, dev


def dec(model, dev, Z):
    with torch.no_grad():
        return model.decode(torch.as_tensor(Z, dtype=torch.float32, device=dev)).cpu().numpy()


def enc(model, dev, X):
    with torch.no_grad():
        return model.encode(torch.as_tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()


def sample_natural(mode, z, signature, n, rng):
    """flat -> Gaussian over the encoded support; structured -> topological domain."""
    if mode == "flat":
        mu, cov = z.mean(0), np.cov(z, rowvar=False) + 1e-5 * np.eye(z.shape[1])
        return rng.multivariate_normal(mu, cov, size=n).astype(np.float32)
    blocks = []
    for kind, d in signature:
        if kind == "S" and d == 1:
            a = rng.uniform(0, 2 * np.pi, n)
            blocks.append(np.stack([np.cos(a), np.sin(a)], 1))
        elif kind == "S":
            v = rng.standard_normal((n, d + 1)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            blocks.append(v)
    return np.concatenate(blocks, 1).astype(np.float32)


def measure(name, mode, m):
    data, model, z, dev = fit(name, mode, m)
    ref = make_manifold(name, n=20000, regime="native", noise_std=0.0, seed=1).X
    tree = cKDTree(ref); nn = lambda P: float(tree.query(P)[0].mean())
    rng = np.random.default_rng(0)

    fvu = float(((dec(model, dev, z) - data.X) ** 2).sum()
                / ((data.X - data.X.mean(0)) ** 2).sum())
    out = {"fvu": fvu}

    if mode == "flat":
        Zl = z[rng.integers(0, len(z), 4000)] + 0.3 * z.std(0) * rng.standard_normal((4000, m))
        out["local"] = nn(dec(model, dev, Zl))
        i, j = rng.integers(0, len(z), 500), rng.integers(0, len(z), 500)
        out["interp"] = float(np.mean([nn(dec(model, dev, (1 - t) * z[i] + t * z[j]))
                                       for t in np.linspace(0, 1, 11)[1:-1]]))

    G = dec(model, dev, sample_natural(mode, z, data.signature, 4000, rng))
    out["gen"] = nn(G)
    if mode == "flat":
        out["gen_proj"] = nn(dec(model, dev, enc(model, dev, G)))   # one projection step
    return out, (data.X, G)


rows, viz = {}, {}
for name, m in EMB.items():
    rows[(name, "flat")], viz[(name, "flat")] = measure(name, "flat", m)
for name in CLOSED:
    rows[(name, "structured")], viz[(name, "structured")] = measure(name, "structured", EMB[name])

print(f"\n{'manifold':10s} {'type':>6s} {'mode':>11s} {'FVU':>6s} {'local':>6s} "
      f"{'interp':>6s} {'gen':>6s} {'gen_proj':>8s}")
for name in EMB:
    for mode in ["flat", "structured"]:
        if (name, mode) in rows:
            r = rows[(name, mode)]; t = "open" if name in OPEN else "CLOSED"
            print(f"{name:10s} {t:>6s} {mode:>11s} {r['fvu']:>6.3f} "
                  f"{r.get('local', float('nan')):>6.3f} {r.get('interp', float('nan')):>6.3f} "
                  f"{r['gen']:>6.3f} {r.get('gen_proj', float('nan')):>8.3f}")
json.dump({f"{k[0]}|{k[1]}": v for k, v in rows.items()}, open(OUT / "steerability.json", "w"), indent=2)

# figure: CLOSED manifolds, flat (fills slack) vs structured (traces) generative decodes
fig = plt.figure(figsize=(12, 8))
for col, name in enumerate(CLOSED):
    for row, mode in enumerate(["flat", "structured"]):
        truth, G = viz[(name, mode)]
        proj = None if truth.shape[1] == 2 else "3d"
        ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection=proj)
        s, g = truth[::5], G[::3]
        if truth.shape[1] == 2:
            ax.scatter(s[:, 0], s[:, 1], s=4, c="0.7"); ax.scatter(g[:, 0], g[:, 1], s=3, c="#c0392b", alpha=0.3)
            ax.set_aspect("equal")
        else:
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=4, c="0.7", alpha=0.5)
            ax.scatter(g[:, 0], g[:, 1], g[:, 2], s=3, c="#c0392b", alpha=0.25); ax.set_zticks([])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name} — {mode}\ngen dist={rows[(name,mode)]['gen']:.2f}", fontsize=10)
fig.suptitle("Generative completeness on CLOSED manifolds: sample the latent's natural domain, decode (red) over truth (grey).\n"
             "flat (top) fills the off-manifold slack; structured (bottom) stays on the manifold.", y=1.03)
fig.tight_layout()
fig.savefig(OUT / "fig11_steerability.png", dpi=130, bbox_inches="tight")
print("wrote fig11")
