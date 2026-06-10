# Manifold-SAE — paper 48-zoo (L0=4) campaign + scoring audit — 2026-06-09

The LR / K / λ / preact sweeps, oracle floor, data-scaling, per-atom-penalty and rank-0-floor
experiments — **re-scored after a capture-metric audit**. Code on branch `manifold-sae`; runs in
`/nlp/scr/nathu/sae-manifold/runs/2026-06-09_paper48_*`; figures + this doc in this folder; the
full per-cell old→new table in `reeval_corrected/reeval_comparison.md`.

## ⚠ Scoring corrections (this session) — READ FIRST
Two scoring issues and one training-config issue were found and fixed. **All capture numbers below
are the corrected ones**; the old per-sample numbers are preserved in each run's
`metrics_persample_buggy.json`.

1. **Per-sample FVU bug (the big one, commit 432af3a).** Isolated single/full/per-atom FVU used a
   per-sample mean of ratios `mean_n[‖r_n‖²/‖x_n‖²]`, which explodes on low-norm samples and was
   **pervasive — it deflated EVERY cell, `full` (union) hardest**. Fixed to aggregate
   `Σ‖r‖²/Σ‖x‖²`. Re-evaluated all 61 ckpts on GPU: e.g. the lrsweep `k8_lr3e-4` cell went
   single 28→**36**, full 32→**48**; `stage2 k8_lam0.003_rev2x` single 28→**44**. The campaign
   captured far more than originally reported.
2. **Standardized to TWO capture metrics: `single` and `tiled` (commits fec3b90, d90ffe5).**
   - **single** = one atom spans the WHOLE manifold (`single<thresh`) — strict dedication.
   - **tiled** = captured by a clean **one-chart-at-a-time atlas**: `single<thresh` OR
     (`full<thresh` AND `frac_overlap<0.05`), where `frac_overlap` = fraction of the manifold's
     samples on which ≥2 atoms fire simultaneously. So `single ⊆ tiled`; the extra credit is for
     **disjoint** multi-chart covers, and we also track **how many charts** each tiling needs
     (`tiled_charts_hist`). `full` (union) is kept only as a secondary diagnostic.
   This rejects **redundant overlap** (`flat_disk_1`: two atoms both firing ~80%, overlap 0.60 → they
   ADD, not tile → NOT tiled) while crediting clean tilings (`segment_0`: two θ-half-charts, overlap
   0.00, 2 charts → tiled). Key consequence: **genuine disjoint tilings are RARE** — on the floor
   winner tiled is only 39 (= single 38 + segment_0), so the single→full gap (38→45) is **mostly
   redundant overlap, not atlases**. Metric also made reproducible (per-instance rng; FVUs were
   order/`n`-dependent, max|Δ|1.8e-2 → 0) + provenance + stable firing-floor dead count.
3. **λ never reached target (training-config).** Every campaign run left `--lam-warmup-steps`
   unset → λ ramped 0→target over the **whole** 150k run, hitting target only at the final step as
   the cosine LR decayed to ~0. **The runs never actually trained AT their nominal λ.** A fix grid
   (`--lam-warmup-steps 50000` to HOLD λ + anti-split levers) is running in
   `runs/2026-06-09_paper48_floorhold/`.

**The within-family "catastrophic failures" are GENUINE model dedication failures, not metric/OOD
bugs** — confirmed by conditional-FVU, in-mixture ground-truth attribution, gate-margin probes, and
cross-seed variation (which variant loses the dedication race is random per run). The metric is
correctly reporting that no single atom spans those manifolds.

## Setup
48 manifolds (8 types × 6 variants), constant **L0=4** presence (uniform without replacement,
σε=1e-5), per-instance RMS-1 normalization, random-orthonormal V_i. All runs use the **learnable
per-dim rank** gate, pool `K:64` (64-atom pool, max-rank K). Recipe: tune λ; set both preacts by
rule (presence `lam_preact=3e-4`; per-dim `lam_preact_dim ≈ λ/30 = 0.8·λ·L0/N`; "1×/2×" = that
rule ×1/×2).

## TL;DR (corrected)
- **Two standard metrics:** `single` (one atom spans the whole manifold) ⊆ `tiled` (clean
  one-chart-at-a-time atlas; also tracks #charts). `full` (union) kept only as a diagnostic.
- **Clean disjoint tilings are RARE — `lam_atom` is the only lever that makes them.** `atomcost
  k8/lamatom0.01` → single 38, **tiled 46, full 46** (zero overlap-only) at act_rank 8.8. Everywhere
  else tiled ≈ single and the union captures are redundant overlap (floor: tiled 39 vs full 45).
- **Most rank-efficient clean cell:** floor `k8/lr3e-4/λ0.003/1×` → **single 38, tiled 39, full 45**
  at **act_rank 8.3**, FVU 0.025, dead 14.
- **Peak raw capture:** `stage2(lr1e-3) k8/λ0.003/2×` → **single 44, tiled 45** — but at **act_rank
  15.4** (≈2× the floor's rank). Higher preact (2×) buys capture by spending rank.
- **λ-hold fix REFUTED** (`floorhold`): holding λ=0.003 over-prunes (single 38→20-31). The gentle ramp
  was beneficial; use `lam_atom` on the gentle recipe, not a hold.
- **"lr3e-4 ≫ lr1e-3" is OVERTURNED:** corrected, lr1e-3 single 39 ≳ lr3e-4 36, and lr1e-3's best
  cell (44/45/47) tops lr3e-4's best (37/43/47). lr3e-4 still wins on `full` (48 vs 33 at the
  lrsweep cells) and rank-efficiency; the two are close, not a blowout.
- **K8 ≫ K4 still holds** for single/tiled (K4 best tiled ~33 vs K8 ~45); K4's `full` is fine.
- **The gap is optimization, not objective:** oracle floor unreached, data-scaling flat, but the
  λ-never-held config error + 14 idle dead atoms are concrete, fixable levers (training fix running).

## 1. LR sweep (lrsweep, K8) — corrected single / tiled / full
| lr | single | tiled | full | FVU | act_rank | dead |
|---|---|---|---|---|---|---|
| 1e-4 | 15 | 18 | 38 | 0.055 | 9.4 | 0 (underfit) |
| **3e-4** | 36 | 39 | **48** | **0.021** | 9.2 | 13 |
| 1e-3 | **39** | **40** | 33 | 0.025 | 7.9 | 19 |
| 3e-3 | 28 | 34 | 32 | 0.039 | 7.8 | 15 |
lr3e-4 vs lr1e-3 is a wash on single/tiled; lr3e-4 wins `full` (48 vs 33 — fewer dead/co-firing,
lr1e-3 has 19 dead) and FVU. K4 cells (not shown) top out far lower (best K4 lr3e-4 single 18).

## 2. K4 vs K8 — headroom still required
K8 reaches single 38-44 / tiled 40-45; K4 tops out around single 24 / tiled 33 (best K4 =
`stage2 k4_lam0.001_rev2x` 31/33/full 48 at act_rank 10.7). K4 (max_rank=4=torus ki, zero slack)
still forces spanning/low single. K8's slack enables one-atom dedication.

## 3. Stage-2 grid @ K8 (λ × preact) — corrected (floor-ON, lr3e-4)
| λ | preact | single | tiled | full | mean_cond | act_rank | dead |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.5× | 14 | 16 | 46 | 0.309 | 10.2 | 1 |
| 0.001 | 1× | 15 | 19 | **48** | 0.241 | 10.7 | 3 |
| 0.001 | 2× | 27 | 34 | **48** | 0.107 | 17.8 | 6 |
| 0.003 | 0.5× | 13 | 21 | 45 | 0.223 | 8.1 | 3 |
| **0.003** | **1×** | **38** | **40** | 45 | 0.048 | **8.3** | 14 |
| 0.003 | 2× | 37 | 42 | 46 | 0.033 | 17.1 | 15 |
| 0.01 | 0.5× | 27 | 29 | 25 | 0.094 | 5.7 | 8 |
| 0.01 | 1× | 34 | 35 | 30 | 0.056 | 6.6 | 18 |
| 0.01 | 2× | 20 | 22 | 21 | 0.027 | 12.7 | 13 |
Higher preact (2×) lifts capture but inflates rank (act_rank 8→17); `full` is near-ceiling (45-48)
across most cells — the model reconstructs almost everything in union, the question is rank +
dedication. Sub-rule preact (0.5×) collapses single while keeping `full` high (heavy spanning).
**Two operating points:** rank-efficient = λ0.003/1× (single 38 @ rank 8.3); max-capture = trade
rank for the 2× cells or lr1e-3 λ0.003/2× (single 44 @ rank 15.4).

## 4. Oracle floor (reference)
Sum of min-capturing ranks over 48 = 70 → at L0=4 oracle ≈ **act_rank 5.83, FVU 0.007**. The floor
winner (act_rank 8.3, FVU 0.025, single 38) spends ~1.4× the oracle rank, ~3.5× the FVU floor.

## 5. Data-scaling — NOT data-limited
K8/lr3e-4/λ0.003, 150k→300k→600k: single 36→38→34, full 48→41→41 (flat / noisy). 4× the data finds
no more dedicated atoms — the bottleneck is optimization/config (λ-hold), not data.

## 6. Per-atom count penalty (`--lam-atom`) — the ONLY lever that yields clean disjoint tilings
Under the strict one-at-a-time `tiled` metric, **`atomcost k8_lamatom0.01` = single 38, tiled 46,
full 46** (act_rank 8.8, mean_cond 0.003). tiled == full means **zero overlap-only captures** — every
captured manifold is a clean atlas. Contrast the floor winner (single 38, **tiled 39**, full 45: 6 of
its captures are redundant overlap) and lr3e-4 (single 36, tiled 36, full 48: all 12 extra are overlap).
So `lam_atom` doesn't merely consolidate to single atoms — it makes the multi-atom captures **disjoint**
(charts that tile, not atoms that sum). The old per-sample metric hid this entirely (read this cell as
single 29). **This is the strongest lever found; `lam_atom` on the gentle-λ floor is the top next experiment.**

**`floorhold` (λ-hold fix) — NEGATIVE.** Holding λ=0.003 (`--lam-warmup-steps 50000`) HURT on every lens
(single 38→20-31, tiled 40→27-36, full 45→37): sustained λ over-prunes; the gentle ramp's lower
effective-λ was beneficial. `lam_atom` helped *within* the hold (base→atom: single 20→31) but couldn't
overcome it. The λ-never-held root-cause hypothesis is REFUTED. → use lam_atom on the GENTLE recipe (no hold).

## 7. Rank-0 floor fix (`--l0-rank-floor`) — adopted; the floor grid is the canonical recipe
Charges ≥1 per firing atom so always-on rank-0 "bias" atoms (a free-rider loophole) get gated down.
Kills the 2 always-on atoms, collapses spanning to ~1 atom/manifold, best on rank-efficiency. The
floor `k8/λ0.003/1×` cell is the canonical low-rank operating point (single 38 / tiled 40 / full 45
@ act_rank 8.3).

## 8. The within-family failures (diagnosis — genuine, not a metric bug)
- `segment_0` single 0.49 but **full 0.014** — a clean 2-atom tiling (atom 6 covers θ<0.24, atom 30
  covers θ>0.27, each cond-FVU ~0.001). Now counted as **tiled**.
- `flat_disk_1` single 0.50, full 0.027 — a **broken** overlapping split (both atoms cond-FVU ~0.42,
  one is rank-1 and structurally can't span a 2-D disk). NOT tiled.
- failing spheres (2/6 captured) **squat on atoms another family owns** (atom 58 = sphere_0 0.96 AND
  swiss_roll_1 1.0; atom 25 = sphere_3 AND helix_5). Not topology: sphere_2/sphere_4 ARE single-atom
  captured. Which variant fails is **random across seeds** (flat_disk_1 fails here, is perfect in
  `lrsweep:k8_lr1e-3` and `stage2:k8_lam0.003_rev2x`).
- Mechanism: dedication collision / spanning split under L0=4 with **14/64 atoms dead at max rank**
  (idle capacity the model split instead of dedicating). Lever is the training side (§ scoring #3).

## Figures (regenerated from corrected metrics)
- `fig1_frontier.png` — FVU(mix) & single-capture vs act_rank (lr3e-4 / lr1e-3 / floor; K8 filled,
  K4 hollow; @0.05 filled, @0.10 hollow; oracle ★).
- `fig2_heatmap.png` — K8 single-capture over λ × preact, **tiled count in parens** per cell.
- `fig3_datascale.png` — single-capture & FVU vs steps (flat).
- `rank_dist.png`, `firing_matrix*.png` — per-atom rank / firing structure (rank-based, unaffected
  by the FVU fix).

## Open threads / next
1. **Training fix (running, `floorhold`):** does holding λ at target (`--lam-warmup-steps 50000`) +
   anti-split (`lam_atom` / `lam_decorr`) collapse the splits and revive the 14 dead atoms?
2. **Report over seeds:** which variant loses dedication is random — average capture over ≥3 seeds.
3. **0-D feature test** — seed the zoo with discrete features; check learn-rank parks them at rank-0.
4. Sphere/closed-surface multi-chart handling remains the genuine hard case (needs ≥2 charts).
