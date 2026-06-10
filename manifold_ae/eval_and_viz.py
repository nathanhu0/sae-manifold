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
    l0_presence = l0                                       # presence mode (int L0 or None); keep separate from
    rng_mix = np.random.default_rng(123)                   # forward's rank-L0 below. Mixture gets its OWN rng; each
    # mixture eval at the training presence mode: constant-L0 (paper) if l0 set, else Bernoulli.
    x, masks = zoo.sample(n, l0_presence, rng_mix, p_active=(None if l0_presence is not None else p_active))
    xt = torch.tensor(x, device=dev) / scale
    with torch.no_grad():
        xh, z, gpre, active, dec, l0_rank = model.forward_jump(xt)
        fvu_mix = (((xh - xt) ** 2).sum() / (xt ** 2).sum()).item()
        firing_frac_mix = active.float().mean(0).cpu().numpy()   # per-atom fraction of mixture samples it fires on
    act_count = masks.sum(1)
    gmask = model.rank_gate().detach()                     # (N, max_rank) {0,1} on-dim mask (learned or fixed)
    atom_rank_vec = gmask.sum(-1).cpu()                    # per-atom rank = on-dim count (NEVER max_rank)
    rows, tri = [], {}
    for idx, inst in enumerate(zoo.instances):
        rng_i = np.random.default_rng(10_000 + idx)            # per-instance rng: isolated FVUs + best_atom are
        th_i, amb = inst.sample_full(n_iso, rng_i)             # reproducible regardless of n / want_tri / order
        x_iso = torch.tensor(amb, device=dev) / scale          # at the in-mixture per-manifold scale
        with torch.no_grad():
            xh_i, z_i, _, active_i, dec_i, _ = model.forward_jump(x_iso)
            contrib = active_i.unsqueeze(-1) * dec_i           # (n, N, d)
            # AGGREGATE FVU = sum_n ||resid_n||^2 / sum_n ||x_n||^2. NOT the per-sample mean of ratios
            # mean_n[||r_n||^2/||x_n||^2]: that divides each sample by its OWN energy and EXPLODES on
            # origin-crossing manifolds (segment, ki=1, has samples with ||x||~0) -- a low-norm-sample
            # artifact, not reconstruction error (the per-sample version reported segment 0/6 while its
            # aggregate FVU is ~0.01; verified by claude_scripts/fvu_definition_stress_test.py). Data is
            # per-instance mean-centered, so aggregate-energy == variance-FVU here.
            tot = (x_iso ** 2).sum().clamp_min(1e-9)                              # scalar total energy
            fa = ((contrib - x_iso.unsqueeze(1)) ** 2).sum(dim=(0, 2)) / tot      # (N,) WHOLE-manifold aggregate FVU
            best = int(fa.argmin()); r = int(round(float(atom_rank_vec[best])))
            single = float(fa[best])                           # best atom over the WHOLE manifold (strict dedication)
            full = float(((xh_i - x_iso) ** 2).sum() / tot)    # union of every firing atom
            # which atoms actually carry this manifold: fraction of its points each fires on.
            # A split shows >=2 atoms each firing on a complementary chunk (e.g. ~50/50 arcs),
            # each with poor WHOLE-manifold FVU, but the union (full) reconstructs it.
            firing_frac = active_i.float().mean(0)             # (N,)
            # CONDITIONAL FVU: an atom's error ONLY on the samples it fires on. Separates a clean spatial TILING
            # (each atom near-perfect on its own chart -- single is high only because no ONE atom spans the whole
            # manifold, e.g. segment_0 cond ~0.001) from a genuinely BROKEN split (atoms bad even where they fire,
            # e.g. flat_disk_1 cond ~0.42). single<thresh asks "does one atom span the whole manifold"; tiled_capture
            # asks "does the model reconstruct it with clean local charts". Denominator is firing-set ENERGY (the
            # firing region is a SUB-arc, not mean-centred, so energy is the natural normaliser).
            def _cond_fvu(a):
                fire = active_i[:, a].bool()
                if int(fire.sum()) < 5:
                    return float("nan")
                num = ((contrib[fire, a] - x_iso[fire]) ** 2).sum()
                den = (x_iso[fire] ** 2).sum().clamp_min(1e-9)
                return float(num / den)
            cond_best = _cond_fvu(best)                        # best-atom quality on its OWN firing region
            ff_best = float(firing_frac[best])
            part = (firing_frac > 0.05).nonzero(as_tuple=True)[0]
            part = part[firing_frac[part].argsort(descending=True)]
            atoms = [dict(atom=int(i), rank=int(round(float(atom_rank_vec[i]))), frac=float(firing_frac[i]),
                          fvu=float(fa[i]), cond_fvu=_cond_fvu(int(i))) for i in part[:8]]
            n_used = int(len(part))
            # captured-as-clean-tiling: a single atom spans it, OR the union reconstructs it AND its dominant
            # atom is a clean local chart (rules out the redundant-overlap broken split, which has cond_best high).
            tiled = bool(single < thresh
                         or (full < thresh and not np.isnan(cond_best) and cond_best < thresh))
        rows.append(dict(name=inst.name, type=inst.type, ki=int(inst.ki), di=int(inst.di),
                         best_rank=r, best_atom=best, single=single, full=full, cond_fvu=cond_best,
                         best_firing_frac=ff_best, tiled_capture=tiled,
                         n_atoms_used=n_used, atoms=atoms))
        if want_tri:
            Vi = torch.tensor(inst.V, device=dev)
            ss = np.arange(n_iso) if n_iso <= cap else rng_i.choice(n_iso, cap, replace=False)
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
    # per-family CONTINUOUS aggregates (not just the discrete <thresh count): mean single/full FVU
    # (catches near-misses -- "captured but off by a little"), mean atoms/manifold (= per-manifold L0,
    # the spanning amount), mean dedicating-atom rank. Grouped by manifold type.
    fam = {}
    for r in rows:
        fam.setdefault(r["type"], []).append(r)
    per_family = {t: dict(
        n=len(rs), captured_single=sum(r["single"] < thresh for r in rs),
        captured_tiled=sum(r["tiled_capture"] for r in rs),
        mean_single_fvu=float(np.mean([r["single"] for r in rs])),
        mean_full_fvu=float(np.mean([r["full"] for r in rs])),
        mean_cond_fvu=float(np.nanmean([r["cond_fvu"] for r in rs])),
        atoms_per_manifold=float(np.mean([r["n_atoms_used"] for r in rs])),
        mean_rank=float(np.mean([r["best_rank"] for r in rs]))) for t, rs in fam.items()}
    res = dict(fvu=fvu_mix, act_rank=float(l0_rank.mean()), n_inst=len(rows),
               active_count_mean=float(act_count.mean()), active_count_std=float(act_count.std()),
               dead_atoms=int((firing_frac_mix < 1e-3).sum()),   # firing-fraction floor (strict ==0 was n/seed-unstable)
               captured_single=sum(m["single"] < thresh for m in rows),
               captured_full=sum(m["full"] < thresh for m in rows),
               captured_tiled=sum(m["tiled_capture"] for m in rows),
               mean_single_fvu=float(np.mean([m["single"] for m in rows])),
               mean_cond_fvu=float(np.nanmean([m["cond_fvu"] for m in rows])),
               eval_mode="isolated",
               eval_n=int(n), eval_n_iso=int(n_iso), thresh=float(thresh), eval_cap=int(cap),
               presence=(f"constant_l0={l0_presence}" if l0_presence is not None
                         else f"bernoulli_p_active={p_active}"),
               per_family=per_family, per_instance=rows)
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
          f"tiled={res['captured_tiled']}/{res['n_inst']} full={res['captured_full']}/{res['n_inst']}  "
          f"mean_single_FVU={res['mean_single_fvu']:.3f} mean_cond_FVU={res['mean_cond_fvu']:.3f}", flush=True)
    # per-family CONTINUOUS view: single (strict) + tiled (clean charts) caps, mean single/cond/full FVU, spanning, rank
    print("  [per-family] type        single tiled  mean_single  mean_cond  mean_full  atoms/mfld  rank", flush=True)
    for t in sorted(res["per_family"]):
        d = res["per_family"][t]
        print(f"   {t:13s} {d['captured_single']}/{d['n']}   {d['captured_tiled']}/{d['n']}   "
              f"{d['mean_single_fvu']:.3f}       {d['mean_cond_fvu']:.3f}      {d['mean_full_fvu']:.3f}      "
              f"{d['atoms_per_manifold']:.1f}        {d['mean_rank']:.1f}", flush=True)
    # per-manifold: which atoms carry it (best single vs the full participating set) -> splitting
    for m in sorted(res["per_instance"], key=lambda r: -r["full"]):
        ids = " ".join(f"{a['atom']}(r{a['rank']},{a['frac']*100:.0f}%)" for a in m["atoms"])
        print(f"   {m['name']:13s} k{m['ki']}  single={m['single']:.3f}(atom {m['best_atom']}) "
              f"full={m['full']:.3f}  uses {m['n_atoms_used']} atoms: {ids}", flush=True)
    return res
