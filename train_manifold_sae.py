"""THE CANONICAL TRAINER for the manifold-SAE: a sparse mixture of nonlinear manifold atoms
(manifold_ae/manifold_sae.py) fit to the synthetic manifold zoo (manifold_ae/manifold_zoo.py).

Each atom = per-atom deep encoder -> r-dim latent + binary presence gate -> per-atom deep
decoder; x_hat = b_dec + sum over ACTIVE atoms. The gate thresholds the raw pre-activation at
0 with a straight-through Heaviside (gpre>0 IS the boundary; no sigmoid, no theta). The
reconstruction uses the {0,1} mask DIRECTLY (active_i * dec_i): the gate only SELECTS, all
magnitude lives in the decoder, so soft==hard by construction. With --learn-rank each atom
also learns WHICH of its latent dims to use (a second gate on a static per-dim bias), so the
effective rank of every atom emerges from the sparsity pressure.

    loss = MSE(x_hat, x)
         + lambda(t)   * mean_b sum_i rank_i * active_i    rank-weighted L0 ("dims are priced");
                                                           --l0-rank-floor charges >=1 per firing
                                                           atom so rank-0 atoms aren't free
         + lam_atom(t) * mean_b sum_i active_i             per-ATOM count cost: breaks the tie
                                                           between 1 rank-2k atom and 2 rank-k
                                                           atoms -> consolidation / clean tilings
         + lam_preact     * mean_b sum_i ReLU(-gpre_i)     presence-gate REACTIVATING loss (dead atoms stay
                                                           revivable; Conerly et al., simplified)
         + lam_preact_dim * sum_(i,k)  ReLU(-dim_bias_ik)  per-dim rank reactivating loss (rule: lambda/30)

lambda(t), lam_atom(t): linear ramp 0 -> target over --lam-warmup-steps (default: whole run).
--gate-grad picks the gate's backward estimator: rect (JumpReLU STE, window jump_eps) or
sigmoid (exact-hard forward, sigmoid'(pre/(eps/4)) backward, nonzero everywhere).

Best known recipe (the "gentle-lambda floor"; 48-manifold zoo, constant-L0=4 mixtures):
  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python train_manifold_sae.py --steps 150000 --lr 3e-4 --lr-schedule warmup_cosine --lam 0.003 --lam-preact 3e-4 --lam-preact-dim 0.0001 --learn-rank --pool 8:64 --l0-rank-floor --l0 4 --variants-per-type 6 --eval-every 30000 --out-dir <run_dir>
  (on SLURM: wrap in ebatch <name> slconf/slconf40s "<cmd>")

Writes into --out-dir: ckpt.pt + the full eval suite (metrics.json, report.md, fvu_strip.png,
triptych_<family>.png, tiling_<name>.png). Re-evaluate any checkpoint later with
`python -m manifold_ae.eval_and_viz <run_dir>`; suite spec/smoke test: eval_smoke_test.py.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from manifold_ae.manifold_zoo import ManifoldZoo
from manifold_ae.manifold_sae import ManifoldSAE, reactivating_loss
from manifold_ae.eval_and_viz import eval_and_viz, compute_capture

POOL = {1: 16, 2: 16, 3: 16, 4: 16}     # 64 atoms (4x oracle's 16; ample per-rank capacity, no rank-1 starvation)
D, STEPS, BATCH = 256, 20000, 2048
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--lam-preact", type=float, default=3e-4)  # pre-act/dead-atom loss
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=0)      # periodic capture eval (0 = only at end)
    ap.add_argument("--jump-eps", type=float, default=2.0)    # STE bandwidth, in raw pre-activation units
    ap.add_argument("--variants-per-type", type=int, default=2)  # 2 -> 16 manifolds
    ap.add_argument("--p-active", type=float, default=0.25)   # independent Bernoulli presence prob (default mode)
    ap.add_argument("--l0", type=int, default=None)           # constant-L0 presence (paper); set -> overrides p-active
    ap.add_argument("--enc-dims", type=str, default="128,64,32")  # per-atom encoder funnel (decoder mirrors)
    ap.add_argument("--lr-schedule", choices=["constant", "warmup_cosine"], default="constant")
    ap.add_argument("--lr-warmup-steps", type=int, default=2000)   # linear warmup before cosine decay
    ap.add_argument("--lam-warmup-steps", type=int, default=None)  # ramp lambda 0->target then HOLD (default: whole run)
    ap.add_argument("--lam-decorr", type=float, default=0.0)       # off-diag activation-correlation penalty (ramped like lambda)
    ap.add_argument("--learn-rank", action="store_true")           # learn per-dim rank (pool rank = MAX); else fixed pool
    ap.add_argument("--pool", default="1:16,2:16,3:16,4:16")       # "rank:count,..." ; learn-rank -> homogeneous e.g. "4:32"
    ap.add_argument("--lam-preact-dim", type=float, default=0.0)   # per-dim gate reactivating loss (learn-rank): keeps pruned dims reachable
    ap.add_argument("--lam-atom", type=float, default=0.0)         # per-ATOM count cost (rank-independent): favor 1 higher-dim atom over a spanning split
    ap.add_argument("--l0-rank-floor", action="store_true")        # charge >=1 per FIRING atom (rank-0 atoms cost 0 otherwise -> free always-on bias atoms)
    ap.add_argument("--gate-grad", choices=["rect", "sigmoid"], default="rect")  # gate backward: rect STE vs sigmoid' surrogate (both gate levels)
    ap.add_argument("--seed", type=int, default=0)                 # init + batch-order seed; zoo GEOMETRY stays seed=0
    ap.add_argument("--sigma-eps", type=float, default=1e-5)       # ambient Gaussian noise in TRAINING mixtures (eval stays clean)
    ap.add_argument("--lam-latent-moment", type=float, default=0.0)  # push each USED latent dim to zero-mean/unit-var over its firing samples
    ap.add_argument("--residual", action="store_true")             # skip connections where consecutive funnel widths match (deep winding stacks)
    a = ap.parse_args()
    enc_dims = tuple(int(x) for x in a.enc_dims.split(","))
    pool = {int(k): int(v) for k, v in (kv.split(":") for kv in a.pool.split(","))}
    # presence mode: --l0 set -> constant-L0 (paper, exactly L0 active); else independent Bernoulli p_active.
    samp_l0, pa = (a.l0, None) if a.l0 is not None else (None, a.p_active)  # 'l0' alone collides w/ forward_jump's L0 tensor
    print(f"[presence] {'constant-L0=%d' % a.l0 if a.l0 is not None else 'Bernoulli p_active=%g' % a.p_active}", flush=True)
    torch.manual_seed(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    zoo = ManifoldZoo(d=D, seed=0, variants_per_type=a.variants_per_type)
    x0, _ = zoo.sample(8192, samp_l0, np.random.default_rng(7), p_active=pa)
    scale = float(np.sqrt((x0 ** 2).mean()))
    rng = np.random.default_rng(a.seed + 1)

    def batch():
        x, _ = zoo.sample(BATCH, samp_l0, rng, sigma_eps=a.sigma_eps, p_active=pa)
        return torch.tensor(x, device=DEV) / scale

    m = ManifoldSAE(d_model=D, rank_dist=pool, enc_dims=enc_dims,
                    jump_eps=a.jump_eps, learn_rank=a.learn_rank,
                    gate_grad=a.gate_grad, residual=a.residual).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    # lambda schedule: ramp 0 -> target over lam_warmup_steps, then HOLD. Default (None) =
    # ramp over the whole run (legacy). Holding at target is what lets the model actually
    # converge AT the sparsity we care about instead of only grazing it at the very end.
    lam_warmup = a.lam_warmup_steps if a.lam_warmup_steps is not None else a.steps

    def lr_mult(step):                            # base lr = a.lr (the peak)
        if a.lr_schedule == "constant":
            return 1.0
        if step < a.lr_warmup_steps:              # linear ramp up
            return (step + 1) / max(1, a.lr_warmup_steps)
        prog = (step - a.lr_warmup_steps) / max(1, a.steps - a.lr_warmup_steps)  # cosine decay -> 0
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    progress = []
    for step in range(a.steps):
        lam = a.lam * min(1.0, step / lam_warmup)  # ramp 0 -> target over lam_warmup, then hold
        xt = batch()
        xh, z, gpre, active, dec, l0 = m.forward_jump(xt)
        recon = ((xh - xt) ** 2).mean()
        # pre-act / dead-atom loss: push non-firing pre-activations (gpre < 0) up toward the
        # boundary (0) so dead atoms stay revivable. Uniform (no decoder-norm weighting): the
        # binary gate carries no magnitude, so the reparam-invariance reason for it is gone.
        preact = reactivating_loss(gpre).sum(-1).mean()    # HIGH-level gate reactivating loss (shared helper)
        # decorrelation: push off-diagonal activation CORRELATION toward 0. Under independent
        # p_active, the dedicated (one-atom-per-manifold) code is the independent one, so this is
        # minimized at dedication; it bites only the within-manifold spanning splits (co-firing
        # atoms). Soft prob sigmoid(gpre) for smooth gradient; forward stays binary. Ramped like lam.
        decorr = gpre.new_zeros(())
        if a.lam_decorr > 0:
            # DeCov: penalize off-diagonal of the activation COVARIANCE (no std normalization, so no
            # 1/std gradient -> stable by construction; dead atoms self-exclude via ~zero variance).
            # Soft prob sigmoid(gpre) for a smooth gradient everywhere; forward stays binary.
            p = torch.sigmoid(gpre)
            pc = p - p.mean(0)
            cov = (pc.t() @ pc) / pc.shape[0]
            decorr = (cov ** 2).sum() - (cov.diag() ** 2).sum()        # off-diagonal squared
        # l0 = sum_i rank_i*active_i (SUM over atoms -> capacity-invariant); .mean() over batch only
        lam_dec = a.lam_decorr * min(1.0, step / lam_warmup)           # same ramp as lam
        # rank-0 atoms cost 0 in the rank-weighted l0, so an always-on rank-0 atom is FREE -> the
        # optimizer parks "bias" crutches there. The floor charges >=1 per FIRING atom (rank>=1
        # unchanged), so the sparsity grips the gate of a rank-0 atom and it dies or grows a dim.
        l0_pen = (active * m.atom_ranks().clamp(min=1.0)).sum(-1) if a.l0_rank_floor else l0
        loss = recon + lam * l0_pen.mean() + a.lam_preact * preact + lam_dec * decorr
        if a.lam_atom > 0:
            # per-ATOM count cost (rank-independent, sum over atoms): the rank-weighted l0 is
            # indifferent to consolidation (1 rank-4 atom == 4 rank-1 atoms), so this fixed cost
            # per active atom breaks the tie toward ONE higher-dim atom vs a spanning split. Ramped like lam.
            loss = loss + a.lam_atom * min(1.0, step / lam_warmup) * active.sum(-1).float().mean()
        if a.lam_latent_moment > 0:
            # latent shaping: each USED latent dim should be zero-mean/unit-variance over the
            # samples its atom fires on (firing-weighted batch moments). First-two-moments
            # standardization only -- NOT full Gaussianity; a cheap prior toward comparable,
            # centered charts. Masked to on-dims (detached gate) and atoms with enough firing.
            w = (active / active.sum(0).clamp(min=1.0)).unsqueeze(-1)      # (B,N,1) per-atom weights
            mu = (w * z).sum(0)                                            # (N,K) firing-weighted mean
            var = (w * (z - mu) ** 2).sum(0)                               # (N,K) firing-weighted var
            mmask = m.rank_gate().detach() * (active.sum(0) > 8).float().unsqueeze(-1)
            mom = ((mu ** 2 + (var - 1.0) ** 2) * mmask).sum() / mmask.sum().clamp(min=1.0)
            loss = loss + a.lam_latent_moment * mom
        if a.learn_rank and a.lam_preact_dim > 0:
            # per-dim reactivating loss: constant upward push on pruned dim-biases (bias<0) so a dim driven
            # below the STE window stays revivable instead of freezing off forever. Analog of the
            # presence-gate preact, one level down. Constant (not ramped): a revivability floor.
            loss = loss + a.lam_preact_dim * reactivating_loss(m.dim_bias).sum()  # LOW-level gate reactivating loss (same helper)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            print(f"step {step:>6} recon={recon.item():.4f} "
                  f"act_atoms={active.sum(1).float().mean().item():.1f} "
                  f"act_rank={l0.mean().item():.1f} preact={preact.item():.3f} "
                  f"decorr={float(decorr):.3f} lam={lam:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            if a.learn_rank:
                ar = m.atom_ranks().detach().round().long().clamp(0, m.max_rank)
                frozen = (m.dim_bias.detach() < -m.jump_eps / 2).float().mean().item()
                # split over ALL atoms (whole pool, dilutes with the dead half) vs the ACTIVE
                # subset (atoms that fired at least once this batch -- the rank that actually
                # matters). n_active itself is a spanning diagnostic: ~16 = one-per-manifold,
                # ~32 = heavy spanning. dims/atom(active) = avg learned rank among used atoms.
                ever = active.sum(0) > 0                                  # (N,) distinct atoms used
                ara = ar[ever]; n_act = int(ever.sum())
                dpa_act = float(ara.float().mean()) if n_act else 0.0
                print(f"          [learn-rank] dims/atom all={ar.float().mean().item():.2f} "
                      f"active={dpa_act:.2f}  n_active={n_act}/{m.n_atoms}  "
                      f"hist_all(0..{m.max_rank})={torch.bincount(ar, minlength=m.max_rank + 1).tolist()} "
                      f"hist_active={torch.bincount(ara, minlength=m.max_rank + 1).tolist()} "
                      f"frozen_frac={frozen:.2f}", flush=True)
        if a.eval_every and step > 0 and step % a.eval_every == 0:
            r, _ = compute_capture(m, zoo, scale, pa, n=2000, n_iso=1000, want_tri=False, l0=samp_l0)
            progress.append(dict(step=step, fvu=r["fvu"], captured_single=r["captured_single"],
                                 captured_full=r["captured_full"], dead=r["dead_atoms"]))
            print(f"  [progress {step:>6}] FVU={r['fvu']:.4f} cap_single={r['captured_single']} "
                  f"cap_full={r['captured_full']} dead={r['dead_atoms']}", flush=True)
    m.eval()
    torch.save({"state_dict": m.state_dict(), "pool": pool, "enc_dims": enc_dims,
                "scale": scale, "lam": a.lam, "lam_preact": a.lam_preact,
                "jump_eps": a.jump_eps, "lr": a.lr, "lr_schedule": a.lr_schedule,
                "lr_warmup_steps": a.lr_warmup_steps, "lam_warmup_steps": lam_warmup,
                "lam_decorr": a.lam_decorr, "learn_rank": a.learn_rank,
                "lam_preact_dim": a.lam_preact_dim,
                "variants_per_type": a.variants_per_type, "p_active": a.p_active,
                "l0": a.l0, "lam_atom": a.lam_atom,
                "l0_rank_floor": a.l0_rank_floor, "gate_grad": a.gate_grad,
                "seed": a.seed, "sigma_eps": a.sigma_eps,
                "lam_latent_moment": a.lam_latent_moment,
                "residual": a.residual}, out / "ckpt.pt")

    # ---- inline eval + viz: per-manifold single/full FVU strip + canonical|latent|decoder ----
    eval_and_viz(m, zoo, scale, out, p_active=pa, l0=samp_l0,
                 extra={"lam": a.lam, "lam_preact": a.lam_preact, "jump_eps": a.jump_eps,
                        "lr": a.lr, "lr_schedule": a.lr_schedule, "lam_warmup_steps": lam_warmup,
                        "lam_decorr": a.lam_decorr, "learn_rank": a.learn_rank,
                        "lam_preact_dim": a.lam_preact_dim, "l0": a.l0,
                        "lam_atom": a.lam_atom, "gate_grad": a.gate_grad,
                        "seed": a.seed, "sigma_eps": a.sigma_eps,
                        "lam_latent_moment": a.lam_latent_moment, "progress": progress})


if __name__ == "__main__":
    main()
