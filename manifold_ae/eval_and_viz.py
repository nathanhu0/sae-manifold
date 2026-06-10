"""Inline eval + visualization for the manifold-SAE (binary JumpReLU gate), saved per run.

ONE forward pass -> into out_dir:
  * metrics.json : overall FVU, active-count distribution (Binomial under independent
    presence), dead atoms, and per-instance best-single-atom FVU + full-model (union)
    FVU + best-atom rank; captured counts (single & full, < thresh).
  * report.md : the whole eval as one artifact -- headline + per-family tables, then a
    section per manifold family embedding its figures.
  * fvu_strip.png : per-instance single & full FVU vs the 0.05 cutoff (sorted).
  * grid_overview.png : ONE 8-row overview -- per family a single canonical example, then per
    variant the best atom's latent chart + its single-atom recon side by side, colored by the
    canonical coordinate so structure preservation is visible.
  * tiling_<name>.png : per multi-atom manifold, the per-atom view -- owner-colored target,
    then each participating atom's latent + reconstruction in its own color.

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


def _cell(fig, n, ncols, r, c, arr, color, title, cmap="hsv", vmin=None, vmax=None):
    """One panel. 3D scatter when the data has >=3 dims, else 2D / 1D. Pass vmin/vmax to pin
    the color gradient to a GLOBAL coordinate range (panels showing subsets of a manifold must
    not re-stretch their own slice of theta to the full colormap)."""
    dim = arr.shape[1]
    ax = fig.add_subplot(n, ncols, r * ncols + c + 1, projection="3d" if dim >= 3 else None)
    if dim >= 3:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=color, cmap=cmap, s=6,
                   alpha=0.7, linewidths=0, vmin=vmin, vmax=vmax)
        ax.set_zticks([])
    else:
        ys = arr[:, 1] if dim >= 2 else np.zeros_like(arr[:, 0])
        ax.scatter(arr[:, 0], ys, c=color, cmap=cmap, s=8, alpha=0.75, linewidths=0,
                   vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return ax


_CYCLIC_THETA0 = {"circle", "torus", "mobius"}      # families whose coordinate 0 wraps around

def _theta_cmap(typ):
    """Colormap for the canonical-coordinate gradient. Cyclic hsv ONLY where coordinate 0
    actually wraps (the seam is real); otherwise a NON-cyclic rainbow (turbo), so the two ends
    of an open manifold don't both map to red — 'red here = red there' stays injective across
    every panel that uses the gradient."""
    return "hsv" if typ in _CYCLIC_THETA0 else "turbo"


def _atom_cmap(j):
    """Light->dark colormap of ONE base color (tab10 cycle): the hue identifies the atom,
    the intensity tracks the canonical coordinate -- so cross-referencing a latent panel with
    a reconstruction panel (or the owner panel) works by hue AND position along the manifold."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    base = to_rgb(plt.get_cmap("tab10")(j % 10))
    light = tuple(1 - 0.3 * (1 - c) for c in base)             # 70% toward white
    dark = tuple(0.55 * c for c in base)
    return LinearSegmentedColormap.from_list(f"atom{j}", [light, base, dark])


def _scatter_sel(ax, arr, sel, dim, **kw):
    """Scatter the selected rows of arr on an existing axes (2D/3D handled)."""
    if not sel.any():
        return
    if dim >= 3:
        ax.scatter(arr[sel, 0], arr[sel, 1], arr[sel, 2], s=6, linewidths=0, **kw)
    else:
        ys = arr[sel, 1] if dim >= 2 else np.zeros(int(sel.sum()))
        ax.scatter(arr[sel, 0], ys, s=8, linewidths=0, **kw)


def _apply_lims(ax, arr, dim):
    """Pin a panel to the data range of arr (pad 5%) so panels in the same coordinate
    frame compose visually."""
    lo, hi = arr.min(0), arr.max(0)
    pad = 0.05 * (hi - lo) + 1e-6
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    if dim >= 2:
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    if dim >= 3:
        ax.set_zlim3d(lo[2] - pad[2], hi[2] + pad[2])


def _tiling_fig(nm, t, path):
    """One-chart-at-a-time atlas view of a multi-atom manifold. Top row, all in the SAME
    canonical V_i coordinates and axis limits: TARGET | ATLAS RECON (each point = its best
    FIRING atom's output ALONE, never the sum), both colored by the canonical-coordinate
    gradient | the same atlas recon colored by OWNING atom (hue = atom, intensity = canonical
    coordinate; gray = target points no participating atom covers). Below: each participating
    atom's latent chart in the same canonical gradient, so latent <-> manifold-position
    correspondence reads directly across every panel."""
    av = t["atoms_viz"]
    ncols = 3
    n = 1 + (len(av) + ncols - 1) // ncols
    dim = t["target"].shape[1]
    fig = plt.figure(figsize=(11, 3.4 * n))
    owner, recon, cov = t["atlas_owner"], t["atlas_recon"], t["atlas_owner"] >= 0
    vmn, vmx = float(t["theta"].min()), float(t["theta"].max())   # GLOBAL gradient range
    cm = _theta_cmap(t["type"])
    ax = _cell(fig, n, ncols, 0, 0, t["target"], t["theta"],
               f"{nm} — {t['di']}D manifold in {t['ki']}D subspace — canonical",
               cmap=cm, vmin=vmn, vmax=vmx)
    _apply_lims(ax, t["target"], dim)
    ax = _cell(fig, n, ncols, 0, 1, recon[cov], t["theta"][cov],
               "atlas recon — best firing atom per point, one at a time",
               cmap=cm, vmin=vmn, vmax=vmx)
    _apply_lims(ax, t["target"], dim)
    ax = fig.add_subplot(n, ncols, 3, projection="3d" if dim >= 3 else None)
    _scatter_sel(ax, t["target"], ~cov, dim, c="0.85")
    for j in range(len(av)):
        sel = owner == j
        _scatter_sel(ax, recon, sel, dim, c=t["theta"][sel], cmap=_atom_cmap(j), alpha=0.85,
                     vmin=vmn, vmax=vmx)
    ax.set_xticks([]); ax.set_yticks([])
    if dim >= 3:
        ax.set_zticks([])
    _apply_lims(ax, t["target"], dim)
    ax.set_title(f"same recon by OWNING atom (gray = uncovered; overlap {t['frac_overlap']*100:.0f}%)",
                 fontsize=8)
    for j, a in enumerate(av):
        cond = "n/a" if np.isnan(a["cond_fvu"]) else f"{a['cond_fvu']:.3f}"
        ax = _cell(fig, n, ncols, 1 + j // ncols, j % ncols, a["latent"], a["theta"],
                   f"atom {a['atom']} (rank {a['rank']}) latent — fires on {a['frac']*100:.0f}%, "
                   f"cond FVU {cond}", cmap=cm, vmin=vmn, vmax=vmx)
        if a["latent"].shape[1] < 3:                  # frame the panel in the atom's hue (2D axes only)
            for spine in ax.spines.values():
                spine.set_edgecolor(_atom_cmap(j)(1.0)); spine.set_linewidth(2.5)
    verdict = "clean tiling" if t["tiled"] and t["single"] >= 0.05 else (
        "single-atom capture" if t["tiled"] else "redundant overlap (atoms sum, not tile)")
    fig.suptitle(f"{nm}: {t['n_used']} atoms — {verdict}  "
                 f"(single FVU {t['single']:.3f}, full FVU {t['full']:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120); plt.close(fig)


def _overview_grid(tri, out_dir):
    """ONE overview figure, 8 rows x (1 + 6x2) panels: per family ONE canonical example (the
    variants look the same up to the random ambient rotation V_i and parameter scale), then for
    EACH of the 6 variants its best-atom latent chart and single best-atom recon side by side
    (thin spacer column between variant pairs). Full-model recon is a separate criterion and is
    NOT in this figure. Returns {key: filename} for the report."""
    fams = {}
    for nm, t in tri.items():
        fams.setdefault(t["type"], []).append(nm)
    fam_names = sorted(fams)
    nvar = max(len(v) for v in fams.values())
    nrows = len(fam_names)
    widths = [1.15] + [0.16, 1, 1] * nvar                 # canonical | (spacer, latent, recon) x variant
    fig = plt.figure(figsize=(1.5 * (1 + 2 * nvar) + 1.6, 1.8 * nrows))
    gs = fig.add_gridspec(nrows, len(widths), width_ratios=widths, wspace=0.07, hspace=0.5)

    def cell(r, c, arr, color, title, cmap, vmn, vmx):
        dim = arr.shape[1]
        ax = fig.add_subplot(gs[r, c], projection="3d" if dim >= 3 else None)
        if dim >= 3:
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=color, cmap=cmap, s=4, alpha=0.7,
                       linewidths=0, vmin=vmn, vmax=vmx)
            ax.set_zticks([])
        else:
            ys = arr[:, 1] if dim >= 2 else np.zeros_like(arr[:, 0])
            ax.scatter(arr[:, 0], ys, c=color, cmap=cmap, s=5, alpha=0.75, linewidths=0,
                       vmin=vmn, vmax=vmx)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=6)

    for r, fam in enumerate(fam_names):
        names = sorted(fams[fam], key=lambda nm: -tri[nm]["single"])   # worst single FVU first
        t0 = tri[sorted(fams[fam])[0]]
        cm = _theta_cmap(t0["type"])
        cell(r, 0, t0["target"], t0["theta"], f"{fam} — {t0['di']}D in {t0['ki']}D",
             cm, float(t0["theta"].min()), float(t0["theta"].max()))
        for v, nm in enumerate(names):
            t = tri[nm]
            vi = nm.rsplit("_", 1)[1]                                  # true variant id, not display order
            vmn, vmx = float(t["theta"].min()), float(t["theta"].max())
            cell(r, 2 + 3 * v, t["latent"], t["theta"],
                 f"v{vi} latent (atom {t['best']} r{t['rank']})", cm, vmn, vmx)
            cell(r, 3 + 3 * v, t["decoder"], t["theta"],
                 f"v{vi} recon FVU {t['single']:.3f}", cm, vmn, vmx)
    # column-group labels + vertical separators (drawn in figure coords from the gridspec slots)
    import matplotlib.lines as mlines
    top = gs[0, 0].get_position(fig).y1
    bot = gs[nrows - 1, 0].get_position(fig).y0
    for v in range(nvar):
        sp = gs[0, 1 + 3 * v].get_position(fig)
        x = sp.x0 + sp.width / 2
        fig.add_artist(mlines.Line2D([x, x], [bot - 0.005, top + 0.045], color="0.65", lw=0.9,
                                     transform=fig.transFigure))
    p0 = gs[0, 0].get_position(fig)
    fig.text(p0.x0 + p0.width / 2, top + 0.03, "original manifold", ha="center", fontsize=10)
    for v in range(nvar):
        pl = gs[0, 2 + 3 * v].get_position(fig)
        pr = gs[0, 3 + 3 * v].get_position(fig)
        fig.text((pl.x0 + pr.x1) / 2, top + 0.03, f"latent {v + 1}    reconstruction {v + 1}",
                 ha="center", fontsize=10)
    fig.suptitle("Per family: one canonical example, then per variant (sorted worst single-FVU first) "
                 "the best-atom latent chart + its single-atom recon", fontsize=12, y=top + 0.075)
    fig.savefig(Path(out_dir) / "grid_overview.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    return {"overview": "grid_overview.png"}


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


def _inmixture_capture(model, zoo, scale, dev, l0, p_active, thresh, n=4000):
    """In-mixture (DEPLOYMENT) single-atom capture. Isolated `single` feeds one manifold at a time
    (mildly OOD); this samples REAL L0-mixtures with ground-truth per-manifold contributions and asks,
    for each instance, the best atom's AGGREGATE FVU reconstructing THAT manifold's truth over the
    samples where it is present. Deployment analog of isolated single (a bit more pessimistic via
    cross-talk). Returns (captured_count, mean_inmix_fvu, per_type_captured)."""
    rng = np.random.default_rng(321)
    x, masks, truth = zoo.sample(n, l0, rng, p_active=(None if l0 is not None else p_active),
                                 return_truth=True)                     # truth (n, n_inst, d), only-active rows set
    xt = torch.tensor(x, device=dev) / scale
    truth_t = torch.tensor(truth, device=dev) / scale
    masks_t = torch.tensor(masks, device=dev)
    with torch.no_grad():
        _, _, _, active, dec, _ = model.forward_jump(xt)                # active (n,N), dec (n,N,d)
    per_type = {}
    fvus = []
    per_inst = {}
    rank_vec = model.rank_gate().detach().sum(-1)                       # (N,) per-atom learned/fixed rank
    for i, inst in enumerate(zoo.instances):
        sel = masks_t[:, i].bool()
        if int(sel.sum()) < 5:
            continue
        ti = truth_t[sel, i]                                            # (n_sel, d) ground-truth contribution
        csel = active[sel].unsqueeze(-1) * dec[sel]                     # (n_sel, N, d) each atom's mixture output
        den = (ti ** 2).sum().clamp_min(1e-9)
        fa = ((csel - ti.unsqueeze(1)) ** 2).sum(dim=(0, 2)) / den      # (N,) per-atom in-mixture aggregate FVU
        best = int(fa.argmin()); s = float(fa[best])
        fvus.append(s)
        per_inst[inst.name] = dict(fvu=s, atom=best, rank=int(round(float(rank_vec[best]))))
        per_type.setdefault(inst.type, []).append(s)
    captured = sum(s < thresh for s in fvus)
    per_type_cap = {t: sum(s < thresh for s in ss) for t, ss in per_type.items()}
    return captured, (float(np.mean(fvus)) if fvus else float("nan")), per_type_cap, per_inst


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
        atoms_per_sample = float(active.sum(1).float().mean())   # MODEL atoms firing/sample (NOT the data's active_count)
    act_count = masks.sum(1)
    gmask = model.rank_gate().detach()                     # (N, max_rank) {0,1} on-dim mask (learned or fixed)
    atom_rank_vec = gmask.sum(-1).cpu()                    # per-atom rank = on-dim count (NEVER max_rank)
    overlap_tol = 0.05                                     # tiled = charts fire one-at-a-time: allow <=5% co-fire
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
            cond_best = _cond_fvu(best)                        # best-atom quality on its OWN firing region (diag)
            ff_best = float(firing_frac[best])
            # ONE-CHART-AT-A-TIME: a true spatial TILING covers the manifold with charts that fire DISJOINTLY,
            # so each sample is reconstructed by a SINGLE chart. >=2 atoms firing on the same sample = redundant
            # OVERLAP (the atoms ADD, not tile). frac_overlap = fraction of the manifold's samples with >=2 active.
            kcount = active_i.sum(1)                            # (n,) atoms firing per sample
            frac_overlap = float((kcount >= 2).float().mean())
            frac_uncovered = float((kcount == 0).float().mean())
            mean_cofire = float(kcount.float().mean())
            part = (firing_frac > 0.05).nonzero(as_tuple=True)[0]
            part = part[firing_frac[part].argsort(descending=True)]
            atoms = [dict(atom=int(i), rank=int(round(float(atom_rank_vec[i]))), frac=float(firing_frac[i]),
                          fvu=float(fa[i]), cond_fvu=_cond_fvu(int(i))) for i in part[:8]]
            n_used = int(len(part))
            # TILED CAPTURE = captured by a clean one-chart-at-a-time atlas: ONE atom spans the whole manifold
            # (single), OR the union reconstructs it (full<thresh) with charts that fire DISJOINTLY
            # (frac_overlap < tol). Disjoint + full<thresh => the union is a concatenation of single-chart
            # reconstructions, each good on its own region. (segment_0: 2 half-charts, overlap ~0 -> tiled;
            # flat_disk_1: 2 atoms both firing ~80% -> overlap high -> NOT tiled, they sum rather than tile.)
            tiled = bool(single < thresh
                         or (full < thresh and frac_overlap < overlap_tol))
            # charts NEEDED for the tiling: 1 if a single atom spans the whole manifold, else the number of
            # disjoint participating charts (n_used). Only meaningful when tiled; reported in the tiled aggregate.
            n_charts = 1 if single < thresh else n_used
        rows.append(dict(name=inst.name, type=inst.type, ki=int(inst.ki), di=int(inst.di),
                         best_rank=r, best_atom=best, single=single, full=full, cond_fvu=cond_best,
                         best_firing_frac=ff_best, frac_overlap=frac_overlap, frac_uncovered=frac_uncovered,
                         mean_cofire=mean_cofire, tiled_capture=tiled, n_charts=n_charts,
                         n_atoms_used=n_used, atoms=atoms))
        if want_tri:
            Vi = torch.tensor(inst.V, device=dev)
            ss = np.arange(n_iso) if n_iso <= cap else rng_i.choice(n_iso, cap, replace=False)
            on = gmask[best].nonzero(as_tuple=True)[0]         # the atom's ON dims (need not be contiguous)
            latent_dims = on if on.numel() else torch.zeros(1, dtype=torch.long, device=dev)
            # per PARTICIPATING atom (the tiling view): its latent chart on its OWN firing samples.
            atoms_viz = []
            for a in part[:6].tolist():
                on_a = gmask[a].nonzero(as_tuple=True)[0]
                dims_a = on_a if on_a.numel() else torch.zeros(1, dtype=torch.long, device=dev)
                idx = np.where(active_i[:, a].bool().cpu().numpy())[0]
                if len(idx) > cap:
                    idx = np.sort(rng_i.choice(idx, cap, replace=False))
                idx_t = torch.as_tensor(idx, device=dev)
                atoms_viz.append(dict(
                    atom=int(a), rank=int(round(float(atom_rank_vec[a]))),
                    frac=float(firing_frac[a]), cond_fvu=_cond_fvu(int(a)),
                    latent=z_i[idx_t][:, a][:, dims_a].cpu().numpy(),
                    theta=np.asarray(th_i)[idx, 0]))
            # one-chart-AT-A-TIME atlas recon on the ss subsample: each point rendered by its best
            # FIRING atom ALONE (single-atom outputs only, never the sum; pointwise-error tiebreak
            # under overlap). owner = index into atoms_viz, -1 where no participating atom fires.
            sub = torch.as_tensor(ss, device=dev)
            part_t = part[:6]
            if part_t.numel():
                err = ((contrib[sub][:, part_t] - x_iso[sub].unsqueeze(1)) ** 2).sum(-1)
                fire_sub = active_i[sub][:, part_t].bool()
                owner = err.masked_fill(~fire_sub, float("inf")).argmin(1)
                atlas_owner = torch.where(fire_sub.any(1), owner, torch.full_like(owner, -1))
                atlas_recon = (contrib[sub, part_t[owner]] @ Vi.t()).cpu().numpy()
                atlas_owner = atlas_owner.cpu().numpy()
            else:
                atlas_recon = np.zeros((len(ss), inst.ki), np.float32)
                atlas_owner = np.full(len(ss), -1)
            tri[inst.name] = dict(
                target=(x_iso[ss] @ Vi.t()).cpu().numpy(),
                latent=z_i[ss][:, best][:, latent_dims].cpu().numpy(),
                decoder=(contrib[ss, best] @ Vi.t()).cpu().numpy(),
                full_recon=(xh_i[ss] @ Vi.t()).cpu().numpy(),
                theta=np.asarray(th_i)[ss, 0], rank=r, ki=int(inst.ki), di=int(inst.di),
                type=inst.type, best=best, tiled=tiled, frac_overlap=frac_overlap,
                n_used=n_used, used_ids=str([a["atom"] for a in atoms[:4]]),
                single=single, full=full, atoms_viz=atoms_viz,
                atlas_recon=atlas_recon, atlas_owner=atlas_owner)
    # multi-threshold capture (the 0.05 cliff hides near-misses): single + tiled at thresh and 2*thresh,
    # recomputed from the stored per-instance fields. tiled at th = single<th OR (full<th AND disjoint).
    captures_by_thresh = {f"{th:.3f}": dict(
        single=sum(m["single"] < th for m in rows),
        tiled=sum(m["single"] < th or (m["full"] < th and m["frac_overlap"] < overlap_tol) for m in rows))
        for th in (thresh, 2 * thresh)}
    # in-mixture (deployment) single-atom capture vs ground-truth contributions -- only for the full eval
    # (want_tri), not the cheap periodic one. Mildly more pessimistic than isolated (cross-talk).
    if want_tri:
        inmix_captured, inmix_mean_fvu, inmix_per_type, inmix_per_inst = _inmixture_capture(
            model, zoo, scale, dev, l0_presence, p_active, thresh)
        for r in rows:                                  # per-manifold deployment FVU next to the isolated ones
            pi = inmix_per_inst.get(r["name"])
            r["inmix_fvu"] = pi["fvu"] if pi else float("nan")
            r["inmix_best_atom"] = pi["atom"] if pi else None
            r["inmix_best_rank"] = pi["rank"] if pi else None
    else:
        inmix_captured, inmix_mean_fvu, inmix_per_type = None, None, None
    # per-family CONTINUOUS aggregates (not just the discrete <thresh count): mean single/full FVU
    # (catches near-misses -- "captured but off by a little"), mean atoms/manifold (= per-manifold L0,
    # the spanning amount), mean dedicating-atom rank. Grouped by manifold type.
    fam = {}
    for r in rows:
        fam.setdefault(r["type"], []).append(r)
    def _charts_mean(rs):
        c = [r["n_charts"] for r in rs if r["tiled_capture"]]
        return float(np.mean(c)) if c else 0.0
    per_family = {t: dict(
        n=len(rs), captured_single=sum(r["single"] < thresh for r in rs),
        captured_at_di=sum(r["single"] < thresh and r["best_rank"] == r["di"] for r in rs),
        captured_tiled=sum(r["tiled_capture"] for r in rs),
        tiled_charts_mean=_charts_mean(rs),
        mean_single_fvu=float(np.mean([r["single"] for r in rs])),
        mean_full_fvu=float(np.mean([r["full"] for r in rs])),
        mean_cond_fvu=float(np.nanmean([r["cond_fvu"] for r in rs])),
        atoms_per_manifold=float(np.mean([r["n_atoms_used"] for r in rs])),
        mean_rank=float(np.mean([r["best_rank"] for r in rs]))) for t, rs in fam.items()}
    res = dict(fvu=fvu_mix, act_rank=float(l0_rank.mean()), n_inst=len(rows),
               atoms_per_sample_mean=atoms_per_sample,
               active_count_mean=float(act_count.mean()), active_count_std=float(act_count.std()),
               dead_atoms=int((firing_frac_mix < 1e-3).sum()),   # firing-fraction floor (strict ==0 was n/seed-unstable)
               captured_single=sum(m["single"] < thresh for m in rows),
               # WINDING criterion: captured AND the dedicating atom's effective rank equals the
               # manifold's INTRINSIC dim di (a wound chart, not a flat one at the embedding dim).
               # captured_at_ki = the flat/subspace-capture counterpart (rank == embedding dim).
               captured_at_di=sum(m["single"] < thresh and m["best_rank"] == m["di"] for m in rows),
               captured_at_ki=sum(m["single"] < thresh and m["best_rank"] == m["ki"] for m in rows),
               captured_tiled=sum(m["tiled_capture"] for m in rows),
               captured_full=sum(m["full"] < thresh for m in rows),   # union, kept as a secondary diagnostic
               tiled_charts_mean=(float(np.mean([m["n_charts"] for m in rows if m["tiled_capture"]]))
                                  if any(m["tiled_capture"] for m in rows) else 0.0),
               tiled_charts_hist={int(k): sum(m["n_charts"] == k for m in rows if m["tiled_capture"])
                                  for k in sorted({m["n_charts"] for m in rows if m["tiled_capture"]})},
               mean_single_fvu=float(np.mean([m["single"] for m in rows])),
               mean_cond_fvu=float(np.nanmean([m["cond_fvu"] for m in rows])),
               captures_by_thresh=captures_by_thresh,
               inmix_captured_single=inmix_captured, inmix_mean_fvu=inmix_mean_fvu,
               inmix_per_type=inmix_per_type,
               eval_mode="isolated",
               eval_n=int(n), eval_n_iso=int(n_iso), thresh=float(thresh), eval_cap=int(cap),
               presence=(f"constant_l0={l0_presence}" if l0_presence is not None
                         else f"bernoulli_p_active={p_active}"),
               per_family=per_family, per_instance=rows)
    return res, tri


def _rank_fvu_fig(res, path, thresh):
    """Per-run, cutoff-free capture summary: one panel per manifold FAMILY; x = atom rank,
    y = FVU (log); thresholds drawn as reference lines only. Each color is a different FVU
    lens, plotted at the rank of the atom it actually measures:
      blue  = single (best whole-manifold atom, at that atom's rank)
      green = one-chart-at-a-time (EACH participating atom's conditional FVU at ITS rank)
      orange= in-mixture single (best deployment atom, at its rank)
    full/union has no single-atom rank -- it is a separate criterion, not shown here."""
    fams = {}
    for m in res["per_instance"]:
        fams.setdefault(m["type"], []).append(m)
    names = sorted(fams)
    ncols = (len(names) + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.3 * ncols, 7.2), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    jit = lambda: rng.uniform(-0.16, 0.16)
    for ax, fam in zip(axes.flat, names):
        ms = fams[fam]
        for m in ms:
            ax.scatter(m["best_rank"] + jit(), max(m["single"], 1e-4), c="C0", s=30, linewidths=0)
            for a in m["atoms"]:
                if a["cond_fvu"] == a["cond_fvu"]:                     # skip NaN (rarely-firing atom)
                    ax.scatter(a["rank"] + jit(), max(a["cond_fvu"], 1e-4), c="C2", s=16,
                               alpha=0.65, linewidths=0)
            if m.get("inmix_best_rank") is not None:
                ax.scatter(m["inmix_best_rank"] + jit(), max(m["inmix_fvu"], 1e-4), c="C1",
                           s=30, marker="^", linewidths=0)
        for th, ls in [(thresh, "--"), (2 * thresh, ":")]:
            ax.axhline(th, color="red", ls=ls, lw=0.9, alpha=0.6)
        ax.set_yscale("log"); ax.set_ylim(8e-5, 12)
        ax.set_title(f"{fam} — {ms[0]['di']}D manifold in {ms[0]['ki']}D subspace", fontsize=8)
        ax.set_xticks(range(0, int(max(5, max(m['best_rank'] for m in ms) + 2))))
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("FVU (log)")
    for ax in axes[-1]:
        ax.set_xlabel("atom rank")
    handles = [plt.Line2D([], [], marker="o", ls="", color="C0", label="single (best atom)"),
               plt.Line2D([], [], marker="o", ls="", color="C2", label="per-chart conditional"),
               plt.Line2D([], [], marker="^", ls="", color="C1", label="in-mixture single")]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Capture without cutoffs: per-family atom rank vs FVU "
                 f"(red lines = {thresh:g} / {2*thresh:g} reference)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140); plt.close(fig)


def _report_md(res, tri_paths, tiling_paths, out, thresh):
    """report.md: the whole eval as ONE self-contained artifact -- headline metrics, the per-family
    capture table, then a section per manifold family: per-instance verdict table, the family
    glance grids, and the per-atom tiling view of every multi-atom manifold."""
    N = res["n_inst"]; loose = res["captures_by_thresh"][f"{2*thresh:.3f}"]
    L = [f"# Eval report — {out.name}\n",
         f"presence `{res['presence']}` · n={res['eval_n']} (mixture) / {res['eval_n_iso']} (isolated)"
         f" · capture threshold {thresh:g}\n",
         "## Headline\n", "| metric | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in [
        ("mixture FVU", f"{res['fvu']:.4f}"),
        ("active rank-dof / sample", f"{res['act_rank']:.1f}"),
        ("atoms firing / sample", f"{res['atoms_per_sample_mean']:.1f}"),
        ("dead atoms", res["dead_atoms"]),
        (f"captured single @{thresh:g}", f"{res['captured_single']}/{N}"),
        (f"captured at INTRINSIC dim (wound chart) @{thresh:g}",
         f"{res['captured_at_di']}/{N} (at embedding dim ki: {res['captured_at_ki']})"),
        (f"captured tiled @{thresh:g}", f"{res['captured_tiled']}/{N} "
         f"(charts ~{res['tiled_charts_mean']:.1f}, hist {res['tiled_charts_hist']})"),
        (f"captured full/union @{thresh:g}", f"{res['captured_full']}/{N}"),
        (f"in-mixture (deployment) single @{thresh:g}", f"{res['inmix_captured_single']}/{N}"),
        (f"single / tiled @{2*thresh:g}", f"{loose['single']}/{N} · {loose['tiled']}/{N}"),
        ("mean single FVU", f"{res['mean_single_fvu']:.3f}")]]
    L += ["\n## Capture by family\n",
          "| family | single | at-di | tiled | charts | mean single FVU | mean full FVU | atoms/manifold | rank |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(res["per_family"]):
        d = res["per_family"][t]
        L.append(f"| {t} | {d['captured_single']}/{d['n']} | {d['captured_at_di']}/{d['n']} | "
                 f"{d['captured_tiled']}/{d['n']} | "
                 f"{d['tiled_charts_mean']:.1f} | {d['mean_single_fvu']:.3f} | {d['mean_full_fvu']:.3f} | "
                 f"{d['atoms_per_manifold']:.1f} | {d['mean_rank']:.1f} |")
    L.append("\n![per-family rank vs FVU, cutoff-free](rank_vs_fvu.png)\n")
    L.append("\n## All manifolds at a glance\n")
    for key, fname in tri_paths.items():
        L.append(f"\n![{key} grid]({fname})\n")
    L.append("\n![per-instance FVU strip](fvu_strip.png)\n")
    fam_rows = {}
    for m in res["per_instance"]:
        fam_rows.setdefault(m["type"], []).append(m)
    for fam in sorted(fam_rows):
        ms = fam_rows[fam]
        L += [f"\n## {fam} — {ms[0]['di']}D manifold in {ms[0]['ki']}D subspace\n",
              "| instance | single FVU (best atom) | full FVU | conditional FVU | overlap | verdict | atoms used |",
              "|---|---|---|---|---|---|---|"]
        for m in ms:
            verdict = ("single" if m["single"] < thresh else
                       f"tiled, {m['n_charts']} charts" if m["tiled_capture"] else
                       "span/overlap (union fits, atoms sum)" if m["full"] < thresh else "missed")
            atoms = " ".join(f"{a['atom']}(r{a['rank']},{a['frac']*100:.0f}%)" for a in m["atoms"][:4])
            L.append(f"| {m['name']} | {m['single']:.3f} (atom {m['best_atom']}) | {m['full']:.3f} | "
                     f"{m['cond_fvu']:.3f} | {m['frac_overlap']*100:.0f}% | {verdict} | {atoms} |")
        for m in ms:
            if m["name"] in tiling_paths:
                L.append(f"\n### {m['name']} — per-atom tiling view\n\n"
                         f"![{m['name']} tiling]({tiling_paths[m['name']]})\n")
    (out / "report.md").write_text("\n".join(L) + "\n")


def render_figures(res, tri, out, thresh=0.05):
    """ALL figure + report rendering from (res, tri) -- no model, no GPU. tri comes either fresh
    from compute_capture or from the cached viz_arrays.npy, so figure iteration is a local CPU
    operation (eval CLI: --figures-only)."""
    out = Path(out)
    if "captured_at_di" not in res:      # retrofit metrics.json written before the winding metric
        th = res.get("thresh", thresh)
        res["captured_at_di"] = sum(m["single"] < th and m["best_rank"] == m["di"]
                                    for m in res["per_instance"])
        res["captured_at_ki"] = sum(m["single"] < th and m["best_rank"] == m["ki"]
                                    for m in res["per_instance"])
        for t, d in res["per_family"].items():
            d["captured_at_di"] = sum(m["single"] < th and m["best_rank"] == m["di"]
                                      for m in res["per_instance"] if m["type"] == t)
    _strip(res["per_instance"], thresh, out / "fvu_strip.png")
    _rank_fvu_fig(res, out / "rank_vs_fvu.png", thresh)
    tri_paths = _overview_grid(tri, out)
    tiling_paths = {}
    for nm, t in tri.items():
        if len(t["atoms_viz"]) >= 2:                # the per-atom view only says something for >=2 atoms
            _tiling_fig(nm, t, out / f"tiling_{nm}.png")
            tiling_paths[nm] = f"tiling_{nm}.png"
    _report_md(res, tri_paths, tiling_paths, out, thresh)


def eval_and_viz(model, zoo, scale, out_dir, p_active=0.25, n=6000, n_iso=2000,
                 thresh=0.05, cap=400, extra=None, l0=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res, tri = compute_capture(model, zoo, scale, p_active, n, n_iso, thresh, cap, want_tri=True, l0=l0)
    res.update(extra or {})
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    np.save(out / "viz_arrays.npy", tri, allow_pickle=True)   # cache the viz payload (a few MB):
    render_figures(res, tri, out, thresh)                     # figures re-render later WITHOUT a GPU
    N = res["n_inst"]; loose = res["captures_by_thresh"][f"{2 * thresh:.3f}"]
    print(f"[eval] FVU(mix)={res['fvu']:.4f} act_rank={res['act_rank']:.1f} "
          f"active/sample={res['active_count_mean']:.1f}+-{res['active_count_std']:.1f} dead={res['dead_atoms']}\n"
          f"  CAPTURE (isolated @{thresh:g}): single={res['captured_single']}/{N} "
          f"(at-di={res['captured_at_di']} at-ki={res['captured_at_ki']})  "
          f"tiled={res['captured_tiled']}/{N} (charts ~{res['tiled_charts_mean']:.1f}, hist {res['tiled_charts_hist']})  "
          f"[full(union)={res['captured_full']}/{N}]\n"
          f"  @{2*thresh:g}: single={loose['single']}/{N} tiled={loose['tiled']}/{N}   "
          f"in-mixture(deployment): single={res['inmix_captured_single']}/{N}   "
          f"mean_single_FVU={res['mean_single_fvu']:.3f}", flush=True)
    # per-family: the TWO standard metrics (single, tiled) + charts needed for the tiling, mean FVUs, rank
    print("  [per-family] type        single  tiled  charts  mean_single  mean_full  rank", flush=True)
    for t in sorted(res["per_family"]):
        d = res["per_family"][t]
        print(f"   {t:13s} {d['captured_single']}/{d['n']}    {d['captured_tiled']}/{d['n']}    "
              f"{d['tiled_charts_mean']:.1f}     {d['mean_single_fvu']:.3f}      {d['mean_full_fvu']:.3f}     "
              f"{d['mean_rank']:.1f}", flush=True)
    # per-manifold: which atoms carry it (best single vs the full participating set) -> splitting
    for m in sorted(res["per_instance"], key=lambda r: -r["full"]):
        ids = " ".join(f"{a['atom']}(r{a['rank']},{a['frac']*100:.0f}%)" for a in m["atoms"])
        print(f"   {m['name']:13s} k{m['ki']}  single={m['single']:.3f}(atom {m['best_atom']}) "
              f"full={m['full']:.3f}  uses {m['n_atoms_used']} atoms: {ids}", flush=True)
    return res


def load_checkpoint(ckpt_path, device="cpu"):
    """Canonical loader: rebuild (model, zoo, scale, l0, ck) from a saved ckpt.pt. d_model is read back
    from the encoder weight so it is not hard-coded. Used by the CLI, reeval_campaign, and smoke tests."""
    from manifold_ae.manifold_zoo import ManifoldZoo
    from manifold_ae.manifold_sae import ManifoldSAE
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["state_dict"]
    if "enc1.weight" in sd:        # legacy fixed-depth naming -> variable-depth ModuleList names
        ren = {"enc1": "encs.0", "enc2": "encs.1", "enc3": "encs.2",
               "dec1": "decs.0", "dec2": "decs.1", "dec3": "decs.2", "dec4": "decs.3"}
        sd = {(ren[k.split(".", 1)[0]] + "." + k.split(".", 1)[1]
               if k.split(".", 1)[0] in ren else k): v for k, v in sd.items()}
        ck["state_dict"] = sd
    d_model = int(sd["encs.0.weight"].shape[2])
    m = ManifoldSAE(d_model=d_model, rank_dist=ck["pool"], enc_dims=ck["enc_dims"],
                    jump_eps=ck["jump_eps"], learn_rank=ck["learn_rank"],
                    gate_grad=ck.get("gate_grad", "rect"),
                    residual=ck.get("residual", False)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    zoo = ManifoldZoo(d=d_model, seed=0, variants_per_type=ck["variants_per_type"])
    return m, zoo, ck["scale"], ck.get("l0", None), ck


if __name__ == "__main__":
    # Canonical single-checkpoint eval: `python -m manifold_ae.eval_and_viz <ckpt.pt|run_dir>`
    # -> writes metrics.json + report.md + all figures into the run dir and prints the full suite.
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a manifold-SAE checkpoint (full capture suite).")
    ap.add_argument("ckpt", help="path to ckpt.pt, or a run dir containing ckpt.pt")
    ap.add_argument("--out-dir", default=None, help="output dir (default: the ckpt's own dir)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--figures-only", action="store_true",
                    help="re-render figures + report from the run dir's cached metrics.json + "
                         "viz_arrays.npy -- CPU-only, no model forward")
    a = ap.parse_args()
    cpath = Path(a.ckpt); cpath = cpath / "ckpt.pt" if cpath.is_dir() else cpath
    if a.figures_only:
        rd = Path(a.out_dir or cpath.parent)
        res = json.load(open(rd / "metrics.json"))
        tri = np.load(rd / "viz_arrays.npy", allow_pickle=True).item()
        render_figures(res, tri, rd, res.get("thresh", 0.05))
        print(f"[figures-only] re-rendered figures + report in {rd}")
        raise SystemExit(0)
    model, zoo, scale, l0, ck = load_checkpoint(cpath, a.device)
    # carry the ckpt's training hyperparams into metrics.json (same provenance the trainer's own
    # inline eval writes) so re-evals don't strip lam/lam_atom/... from downstream aggregation.
    extra = {k: ck[k] for k in ["lam", "lam_preact", "jump_eps", "lr", "lr_schedule",
                                "lam_warmup_steps", "lam_decorr", "learn_rank", "lam_preact_dim",
                                "l0", "lam_atom", "l0_rank_floor", "gate_grad", "seed",
                                "variants_per_type", "pool"] if k in ck}
    eval_and_viz(model, zoo, scale, a.out_dir or cpath.parent,
                 p_active=(None if l0 is not None else ck.get("p_active", 0.25)), l0=l0,
                 extra=extra)
