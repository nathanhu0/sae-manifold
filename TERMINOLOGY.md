# Terminology (canonical)

Use these terms in code, reports, plots, and discussion. Where an older synonym exists it is
listed as DEPRECATED — don't introduce it in new writing.

## Model

- **manifold atom** (or just **atom**) — one dictionary element: a per-atom deep encoder →
  r-dim latent → per-atom deep decoder, plus a binary presence gate. The nonlinear
  generalization of an SAE feature direction.
- **rank** (of an atom) — the number of latent dimensions the atom uses. With **learn-rank**,
  each atom learns which of its `max_rank` dims are on via a per-dim gate on a static bias;
  the **effective rank** is the on-dim count. Never report `max_rank` as an atom's rank.
- **presence gate** — the binary on/off gate per atom: Heaviside of the raw gate
  pre-activation at 0 (`gpre > 0` fires). Carries no magnitude; it only selects.
  (Historical name "JumpReLU gate" is inaccurate — no learned threshold, no magnitude
  pass-through; prefer "binary presence gate".)
- **gate backward / STE** — the gradient estimator for the gate: **rect** (rectangular
  window of width `jump_eps`, the default and empirical winner) or **sigmoid surrogate**
  (exact-hard forward, `sigmoid'(pre/(eps/4))` backward).

## Losses (see train_manifold_sae.py docstring for the full objective)

- **rank-weighted L0** (`--lam`, λ) — sparsity term `λ · Σ rank_i · active_i`: *dimensions
  are priced*. The **rank floor** (`--l0-rank-floor`) charges ≥1 per firing atom so rank-0
  atoms aren't free riders.
- **per-atom count cost** (`--lam-atom`) — `lam_atom · Σ active_i`: *naming an atom is
  priced*. The consolidation / clean-tiling lever. MDL reading: λ ≈ bits per coordinate,
  lam_atom ≈ bits per atom identity.
- **reactivating loss** (DEPRECATED: "revival", "revival loss", "pre-act loss") — the
  constant upward push `ReLU(-pre)` on gate pre-activations that keeps switched-off
  components reachable by gradient. Two levels: **presence reactivating loss**
  (`--lam-preact`) on the atom gates, and **per-dim reactivating loss**
  (`--lam-preact-dim`, rule of thumb λ/30) on the learn-rank dim gates. Both are
  necessary (2026-06-09 bake-off: dropping them collapses dedication under every gate
  backward; the sigmoid surrogate does not substitute).
- **gentle ramp** — λ and lam_atom ramp linearly 0 → target over the whole run (no hold).
  Ramp-then-hold ("floorhold") over-prunes; the reactivating losses are NOT ramped.

## Evaluation (manifold_ae/eval_and_viz.py; FVUs are AGGREGATE Σ‖r‖²/Σ‖x‖², never
per-sample means of ratios)

- **single capture** — one atom reconstructs the WHOLE manifold (best atom's FVU < thresh,
  default 0.05) on isolated samples. Strict dedication.
- **tiled capture** — single, OR the union reconstructs (full < thresh) with charts firing
  one-at-a-time (co-fire overlap < 5%). Credits clean disjoint **atlases**; rejects atoms
  that sum. **#charts** = atoms in the tiling.
- **full / union capture** — the whole model reconstructs the isolated manifold, any number
  of co-firing atoms. A SEPARATE criterion (span-learning signal), not the third rung of a
  ladder.
- **in-mixture (deployment) capture** — same dedication question asked on real L0=4
  mixtures: best atom's output vs that manifold's ground-truth CONTRIBUTION, over samples
  where it is present. The honest deployment number; collapses before isolated metrics do.
- **conditional FVU** — an atom's error only on the samples it fires on (its own chart).
  Separates "bad atom" from "good chart that doesn't span".
- **atlas recon** — the one-chart-at-a-time stitched reconstruction (each point = its best
  FIRING atom's output alone, never the sum) used in tiling figures.
- **atoms/sample** (`atoms_per_sample_mean`) — MODEL atoms firing per mixture sample
  (dedication target = data L0 = 4). Distinct from `active_count_mean` (the DATA's
  ground-truth active-manifold count). **act_rank** — rank-weighted dof firing per sample.

## Failure modes / regimes

- **dedication** — one atom per manifold (the goal; atoms/sample ≈ L0).
- **spanning** — several atoms co-fire and SUM to reconstruct one manifold (redundant
  overlap; weak-λ regime).
- **clean tiling** — several atoms partition a manifold into disjoint one-at-a-time charts
  (legitimate; required for closed surfaces like the sphere).
- **fragmentation** — atoms collapse to rank ~1 shards (no-reactivating-loss failure;
  many atoms, low rank each).
- **mixture-context entanglement** — atoms that look dedicated in isolation but explain
  co-occurring COMBINATIONS in mixtures (over-consolidation failure, lam_atom ≥ 0.03;
  visible only through the in-mixture lens).

## Zoo (manifold_ae/manifold_zoo.py)

- **family / type** — one of the 8 manifold kinds (circle, sphere, torus, mobius,
  swiss_roll, helix, flat_disk, segment); **variant/instance** — one of 6 parameterizations
  of a family. 48 instances total.
- **intrinsic dim `di`** — the manifold's free parameters (sphere 2, helix 1): chart rank.
- **embedding dim `ki`** — the ambient subspace dim (sphere 3, helix 3, torus 4): the rank
  one atom needs for SEAMLESS single-atom capture. Don't conflate di and ki; learned ranks
  typically land between them.
