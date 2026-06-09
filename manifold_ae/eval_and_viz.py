"""Inline eval + visualization for the manifold-SAE (binary JumpReLU gate), saved per run.

ONE forward pass -> into out_dir:
  * metrics.json : overall FVU, active-count distribution (Binomial under independent
    presence), dead atoms, and per-instance best-single-atom FVU + full-model (union)
    FVU + best-atom rank; captured counts (single & full, < thresh).
  * fvu_strip.png : per-instance single & full FVU vs the 0.05 cutoff (sorted).
  * triptych.png  : per manifold, CANONICAL (target in V_i coords) | LATENT (the best
    atom's encoder z, its learned chart) | DECODER (that atom's reconstruction), colored
    by the canonical angle so structure preservation is visible.

Built for the small zoo (few variants/type, independent p_active presence). The forward
runs on whatever device the model is on (i.e. inside the training job, on jag).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _cell(fig, n, ncols, r, c, arr, color, title):
    """One panel. 3D scatter when the data has >=3 dims, else 2D / 1D."""
    dim = arr.shape[1]
    ax = fig.add_subplot(n, ncols, r * ncols + c + 1, projection="3d" if dim >= 3 else None)
    if dim >= 3:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=color, cmap="hsv", s=6,
                   alpha=0.7, linewidths=0)
        ax.set_zticks([])
    else:
        ys = arr[:, 1] if dim >= 2 else np.zeros_like(arr[:, 0])
        ax.scatter(arr[:, 0], ys, c=color, cmap="hsv", s=8, alpha=0.75, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def _triptych(tri, path):
    """Per manifold (isolated samples): canonical -> best-atom chart -> best-atom recon ->
    FULL-model recon. Columns 3 vs 4 make splitting visible: if the manifold is split across
    atoms, the best single atom (col 3) covers only an arc while the full model (col 4) covers
    it -- and col 4's title lists the participating atom indices."""
    names = list(tri); n = len(names); ncols = 4
    fig = plt.figure(figsize=(13, 3.1 * n))
    for r, nm in enumerate(names):
        t = tri[nm]
        _cell(fig, n, ncols, r, 0, t["target"], t["theta"], f"{nm}  k{t['ki']} — canonical")
        _cell(fig, n, ncols, r, 1, t["latent"], t["theta"], f"best atom {t['best']} chart (r{t['rank']})")
        _cell(fig, n, ncols, r, 2, t["decoder"], t["theta"],
              f"best-atom recon — single FVU {t['single']:.3f}")
        _cell(fig, n, ncols, r, 3, t["full_recon"], t["theta"],
              f"full model ({t['n_used']} atoms {t['used_ids']}) — full FVU {t['full']:.3f}")
    fig.suptitle("Per-manifold (isolated): canonical -> best-atom chart -> best-atom recon -> "
                 "FULL-model recon  (3D where dim>=3; colored by canonical angle)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=120); plt.close(fig)


def _strip(rows, thresh, path):
    rows = sorted(rows, key=lambda m: m["single"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.5))
    ax.scatter(np.clip([m["single"] for m in rows], 1e-4, 5), y, c="C0", s=45,
               label="best single atom", zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter(np.clip([m["full"] for m in rows], 1e-4, 5), y, c="C1", marker="^", s=45,
               label="full model (union)", zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(thresh, color="red", ls="--", lw=1.3, label=f"cutoff {thresh}")
    ax.set_xscale("log"); ax.set_xlim(8e-4, 9)
    ax.set_yticks(y); ax.set_yticklabels([f"{m['name']} (k{m['ki']})" for m in rows], fontsize=8)
    ax.set_xlabel("fraction of variance unexplained (FVU, log)")
    ax.set_title("Per-instance FVU: best single atom vs full model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def compute_capture(model, zoo, scale, p_active=0.25, n=6000, n_iso=2000, thresh=0.05,
                    cap=400, want_tri=True, l0=None):
    """Compute metrics (no file writes). Returns (res, tri).

    Global deployment metrics on real MIXTURES (FVU, active-count, dead atoms) + per-manifold
    reconstruction on ISOLATED samples (only that manifold present), so BOTH best-single-atom
    and full-model FVU are free of co-active cross-talk. tri = per-manifold viz arrays (empty
    if not want_tri -- skip it for cheap periodic evals)."""
    dev = next(model.parameters()).device
    rng = np.random.default_rng(123)
    # mixture eval at the training presence mode: constant-L0 (paper) if l0 set, else Bernoulli.
    x, masks = zoo.sample(n, l0, rng, p_active=(None if l0 is not None else p_active))
    xt = torch.tensor(x, device=dev) / scale
    with torch.no_grad():
        xh, z, gpre, active, dec, l0 = model.forward_jump(xt)
        fvu_mix = (((xh - xt) ** 2).sum() / (xt ** 2).sum()).item()
        ever = (active.sum(0) > 0).cpu().numpy()
    act_count = masks.sum(1)
    gmask = model.rank_gate().detach()                     # (N, max_rank) {0,1} on-dim mask (learned or fixed)
    atom_rank_vec = gmask.sum(-1).cpu()                    # per-atom rank = on-dim count (NEVER max_rank)
    rows, tri = [], {}
    for idx, inst in enumerate(zoo.instances):
        th_i, amb = inst.sample_full(n_iso, rng)               # ONLY manifold idx
        x_iso = torch.tensor(amb, device=dev) / scale          # at the in-mixture per-manifold scale
        with torch.no_grad():
            xh_i, z_i, _, active_i, dec_i, _ = model.forward_jump(x_iso)
            contrib = active_i.unsqueeze(-1) * dec_i           # (n, N, d)
            den = (x_iso ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
            fa = (((contrib - x_iso.unsqueeze(1)) ** 2).sum(-1) / den).mean(0)   # per-atom FVU
            best = int(fa.argmin()); r = int(round(float(atom_rank_vec[best])))
            single = float(fa[best])
            full = float((((xh_i - x_iso) ** 2).sum(-1) / den.squeeze(-1)).mean())
            # which atoms actually carry this manifold: fraction of its points each fires on.
            # A split shows >=2 atoms each firing on a complementary chunk (e.g. ~50/50 arcs),
            # each with poor single FVU, but the union (full) reconstructs it.
            firing_frac = active_i.float().mean(0)             # (N,)
            part = (firing_frac > 0.05).nonzero(as_tuple=True)[0]
            part = part[firing_frac[part].argsort(descending=True)]
            atoms = [dict(atom=int(i), rank=int(round(float(atom_rank_vec[i]))), frac=float(firing_frac[i]),
                          fvu=float(fa[i])) for i in part[:8]]
            n_used = int(len(part))
        rows.append(dict(name=inst.name, type=inst.type, ki=int(inst.ki), di=int(inst.di),
                         best_rank=r, best_atom=best, single=single, full=full,
                         n_atoms_used=n_used, atoms=atoms))
        if want_tri:
            Vi = torch.tensor(inst.V, device=dev)
            ss = np.arange(n_iso) if n_iso <= cap else rng.choice(n_iso, cap, replace=False)
            on = gmask[best].nonzero(as_tuple=True)[0]         # the atom's ON dims (need not be contiguous)
            latent_dims = on if on.numel() else torch.zeros(1, dtype=torch.long, device=dev)
            tri[inst.name] = dict(
                target=(x_iso[ss] @ Vi.t()).cpu().numpy(),
                latent=z_i[ss][:, best][:, latent_dims].cpu().numpy(),
                decoder=(contrib[ss, best] @ Vi.t()).cpu().numpy(),
                full_recon=(xh_i[ss] @ Vi.t()).cpu().numpy(),
                theta=np.asarray(th_i)[ss, 0], rank=r, ki=int(inst.ki), best=best,
                n_used=n_used, used_ids=str([a["atom"] for a in atoms[:4]]),
                single=single, full=full)
    res = dict(fvu=fvu_mix, act_rank=float(l0.mean()), n_inst=len(rows),
               active_count_mean=float(act_count.mean()), active_count_std=float(act_count.std()),
               dead_atoms=int((~ever).sum()),
               captured_single=sum(m["single"] < thresh for m in rows),
               captured_full=sum(m["full"] < thresh for m in rows),
               eval_mode="isolated", per_instance=rows)
    return res, tri


def eval_and_viz(model, zoo, scale, out_dir, p_active=0.25, n=6000, n_iso=2000,
                 thresh=0.05, cap=400, extra=None, l0=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res, tri = compute_capture(model, zoo, scale, p_active, n, n_iso, thresh, cap, want_tri=True, l0=l0)
    res.update(extra or {})
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    _strip(res["per_instance"], thresh, out / "fvu_strip.png")
    _triptych(tri, out / "triptych.png")
    print(f"[eval] FVU(mix)={res['fvu']:.4f} act_rank={res['act_rank']:.1f} "
          f"active/sample={res['active_count_mean']:.1f}+-{res['active_count_std']:.1f} "
          f"dead={res['dead_atoms']}  captured(isolated) single={res['captured_single']}/{res['n_inst']} "
          f"full={res['captured_full']}/{res['n_inst']}", flush=True)
    # per-manifold: which atoms carry it (best single vs the full participating set) -> splitting
    for m in sorted(res["per_instance"], key=lambda r: -r["full"]):
        ids = " ".join(f"{a['atom']}(r{a['rank']},{a['frac']*100:.0f}%)" for a in m["atoms"])
        print(f"   {m['name']:13s} k{m['ki']}  single={m['single']:.3f}(atom {m['best_atom']}) "
              f"full={m['full']:.3f}  uses {m['n_atoms_used']} atoms: {ids}", flush=True)
    return res
