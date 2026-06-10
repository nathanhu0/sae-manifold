# Manifold-AE — progress log

Watch this dir. `figures/` holds the latest plots (overwritten each update);
newest narrative entry is at the top.

---

## 2026-06-08 · Milestone 4 — mixture-of-manifolds SAE built + trains on real Llama acts

`manifold_ae/manifold_sae.py` + `train_sae.py`. Architecture (per user's design):
fully **per-atom** funnel encoder `4096→256→128→64→[rank]` + sigmoid on/off gate off
the 64-d layer; mirrored decoder. **Non-uniform rank pool** `{1:128,2:64,3:32,4:16}`
= 240 atoms (MOLT-style: variable rank beats uniform). **Rank-budgeted TopK** =
complexity-weighted sparsity (a rank-k atom costs k; MOLT's Frobenius-norm penalty,
realized as a budget) with **sparsity warmup** R: 416→100 exp-decay.

Data: 8 concept manifolds + C4 background corpus (500k×4096) cached to
`/nlp/scr/nathu/sae-manifold/cache/`. First C4 run (1500 steps, 524M params, lr 4e-4):
FVU **1.00→0.63**, R-warmup tracks (atoms 240→58, rank-used→100.6), gate binarizes
(presence 0.50→0.99). Validates the pipeline learns; not yet a strong reconstructor
(small/short debug run). Checkpoint → scratch `manifold_sae_ckpt.pt`.

Connection: this is MOLT (transformer-circuits 2025) with **nonlinear manifold
atoms** instead of low-rank linear transforms — same anti-shatter, complexity-
weighted-sparsity philosophy. Next: bridge eval (encode concept manifolds →
which atoms fire, do they recover known topology e.g. days=loop, years=helix);
longer/bigger training for real FVU; nested ordering (deferred).

---

## 2026-06-08 · ⚠ CORRECTION 2 (steerability verify, wf wlhjtm1zc) — flat IS generatively complete

Milestone-3's "flat is generatively incomplete / cold generation is lossy" was a
**Gaussian-prior artifact**, caught + adversarially confirmed. Flat `gen` sampled
a full-covariance Gaussian over the encoded codes; for a non-convex support
(loop/shell) the Gaussian fills the convex hull and decodes off-manifold. Swap in
a **support-matching prior** (resample encoded codes + small noise, or KDE) and
flat gen collapses to structured-level or better: circle 0.23→0.006 (struct
0.008), sphere 0.37→0.040 (0.034), torus 0.34→0.037 (**beats** struct 0.120).
Verified NOT memorization: fresh novel on-support samples (interp between encoded
neighbours) decode on-manifold too, robust to 30× noise. The Gaussian artifact
even **grows with fit quality** (thinner support → more off-support volume:
s_curve 0.43→0.77 at lower FVU) while the support-prior number is stable.
gen_proj (S4) was a weak fix; the support-prior fixes all manifolds.

⇒ **Flat@embedding is generatively complete in practice** — the decoder's image
IS the manifold; you just need a support-conforming sampler (perturb encoded
codes), NOT structured topology. Flat wins on fit AND generation. For the
mixture: to steer/generate an atom, sample from a support prior over its encoded
codes, not a Gaussian. (3rd metric artifact the verification caught — C3
fairness, box-sampling, Gaussian prior. Completeness metrics are subtle.)

---

## 2026-06-08 · Milestone 3 — steerability / generative completeness (verifying)

Question (user): "pretend every manifold is open" (plain flat latent, no
topology) — how bad is it for STEERING, not fit? Measured local-steer / interp /
generative-completeness (sample latent's natural domain → decode → NN-dist to
manifold; scale 1). `plotting_scripts/.../steerability.py`, fig11.

- **Local steering fine everywhere** (0.03–0.22, open & closed) — nudge a real
  point, stay near manifold. Pretend-open is cheap here.
- **Cold generation is where flat fails** — flat gen 0.16–0.43; structured gen
  0.01–0.12 (complete by construction: its domain = the sampling distribution).
- **CORRECTION to "completeness is free for open manifolds" (was too strong):**
  curved OPEN manifolds (swiss_roll 0.41, s_curve 0.43) are as incomplete as the
  closed ones. Completeness tracks whether the latent **support is simple/
  sample-able**, NOT open-vs-closed. Open topology removes the seam; it doesn't
  give a sample-able latent.
- One projection `dec∘enc` recovers some (circle 0.17→0.04 ≈ structured) not all
  (s_curve, torus barely move).
- ⚠ First metric cut was confounded (sampled the latent bounding BOX → box-corner
  extrapolation + fed structured invalid un-normalized inputs). Fixed to sample
  each latent's natural domain. Now verifying whether the Gaussian-prior `gen`
  is itself a fair proxy (workflow wlhjtm1zc).

**Takeaway:** pretend-open is cheap for local steering, lossy for cold
generation — and the lever for generation is a sample-able support (structured
domain / flexible prior / projection), not closedness.

---

## 2026-06-08 · ⚠ CORRECTION (adversarial audit, 6 lenses) — "structured beats flat" was an artifact

The Milestone-1 headline ("structured/matched latent beats flat at fixed
capacity on closed manifolds") was WRONG — caught by the audit workflow and
independently verified. A structured `S^n` latent secretly uses **n+1**
coordinates (it L2-normalizes an (n+1)-vector), but the "flat@k" baseline used
only **k = intrinsic_dim** coordinates. So it was a 2-coord decoder vs a 1-coord
decoder — the gap was capacity, not topology. The structured circle net and a
flat-with-2-coords net are *byte-for-byte identical* except one L2-norm.

**At equal latent width** (flat @ minimal seam-free *embedding* dim vs structured):
circle flat@2=0.0028 vs structured 0.0017 (tie); sphere flat@3=0.0029 vs 0.0021
(tie); **torus flat@4=0.003 vs structured 0.094 — flat WINS ~30×, and structured
is unstable (works in 1/3 seeds).** ⇒ **The topology constraint buys no fit.**
Audit also flagged: membership-AUROC may be a norm artifact (off-manifold probes
are full Gaussians) — to be hardened. See fig10_fair_comparison.png. Full audit:
`tasks/wki4xed7v.output` (14 findings, C1/C5 survived, C3 invalidated).

**Reframed conclusions:**
- Minimal *seam-free* latent size = the **embedding dim** (circle 2, sphere 3),
  not the intrinsic dim. "As small as possible" bottoms out there.
- Flat @ embedding-dim fits anything (Whitney) and ties/beats structured. Drop
  structured as a fit tool.
- **Atlas dropped** (user pref): it fragments the latent across charts+gating —
  opposite of "the latent tells the whole story."
- Open goal: make the flat latent *canonical/readable* ("latent tells the
  story") via soft pressure, not hard topology. → testing the per-dimension
  learnable flat-or-circular latent (`PerDimAutoencoder`,
  `claude_scripts/perdim_experiment.py`): spans R^a×T^b, should recover per-dim
  types and expose the sphere as the boundary.

---

## 2026-06-08 · Milestone 2 — latent dim, high-dim scaling, flat-first (IN PROGRESS)

Standard naming locked: ambient **D**, intrinsic **k\***, latent **k**;
under/exact/over-complete; latent geometry **flat** (`R^k`) vs **structured**
(topology baked in). `matched`→`structured` in code.

**Latent-dim sweep (done).** Under-complete (k<k\*) = hard irreducible failure
(FVU floors at dropped-variance). Over-complete (k>k\*) = safe and often *better*
— no blow-up even at 2k\*. The win for the flat-first plan: **flat@(k\*+1)
≈ structured** on single closed manifolds (sphere flat@3 = 0.0028 vs structured
0.0021; swiss_roll 0.053→0.003; torus 0.151→0.054). The extra dim lets the loop
close in a higher-dim *flat* latent without a seam. ⇒ default = **flat,
over-provisioned by 1–2 dims**; structured is a reference, not the recipe.

**High-dim sweep (done; width 384, depth 4, 2000 ep, n=16384).**
- **flat patches R^k:** trivial at every k (FVU ~0.003 up to k=25); AE ≈ PCA
  (flat data is linear, nothing to beat). PR recovers k\* exactly.
- **spheres S^k:** fit well at every k (FVU ~0.003 to S^25). AE-over-PCA margin
  shrinks with dim (S^2 ~150×, S^25 ~12×) but stays large. flat@k\* nearly
  matches structured.
- **tori T^k = (S^1)^k:** the wall. flat@k\* DEGRADES with dim — T^2 0.023,
  T^10 **0.485**, T^25 **1.17** (worse than the mean). structured holds far
  better (T^10 0.041, T^25 0.70). A k-torus is k independent loops; flat@k\*
  can't give each loop room to close. **This is the one regime that may
  genuinely need structured latents.**

**Decisive test (RESOLVED — flat-first wins).** Over-complete flat at ~2k\*
recovers and *beats* structured on high-dim tori, at EQUAL latent width:
T^10 flat@20 = **0.0035** vs structured@10 (also 20 coords) = 0.049 (14×);
T^5 flat@10 = 0.0028 vs structured@5 = 0.0086. Each circle needs one extra
latent dim to close; given ~2k\* dims the flat net learns the wrap-around better
than the hard topology constraint (which can hurt optimization). flat@(k\*+1)
is not enough for many-loop tori (T^10 flat@11 still 0.36) — need ~2k\*.
Caveat: flat@2k\* is not parsimonious (PR ~14 for a 10-d manifold); structured
trades fit for a minimal chart. AE/PCA is uninformative at 2k\* (a T^k sits in a
2k-d linear hull). → `claude_scripts/torus_overcomplete_probe.py`,
`results/manifold_ae/torus_overcomplete.csv`.

**⇒ Design conclusion: flat-first holds everywhere.** Default recipe = flat
latent, over-provisioned toward ~2k\* when many periodic factors are expected.
Structured stays as an optional parsimony tool, not a requirement.

**Also running:** arch sweep (lr × width × depth, flat vs structured, 576 runs)
— how deep/expressive the net must be; adversarial audit workflow (6 lenses).

---

## 2026-06-08 · Milestone 1 — single-manifold de-risk (DONE)

**Question:** can a single deep autoencoder `R^D → R^k → R^D` fit a known
k-dim manifold, and does giving the latent the manifold's *topology* matter?

**Setup:** 7 toy manifolds (4 contractible: helix, swiss_roll, s_curve,
wavy_sheet; 3 closed: circle, sphere, torus) × 3 regimes (native / random
orthonormal-embedded into R^256 / embedded+noise) × {flat R^k latent,
topology-matched latent} × k-sweep × 3 seeds = 297 runs, ~5 min on 4 GPUs.

**Findings (embedded+noise, the realistic regime):**
- **The AE beats linear PCA wherever there's curvature** — helix 18×, sphere
  (matched) 159×, circle (matched) 298×. s_curve/torus barely beat PCA at the
  default capacity (they need more — see Milestone 2). Core thesis holds.
- **Contractible manifolds: flat latent is enough** (matched ≡ flat there by
  construction — a good sanity check that the matched machinery is a no-op on
  R^n factors).
- **Closed manifolds: topology-matched latent wins on every axis** at fixed
  capacity — FVU (circle 145× lower, sphere 15×), denoising (circle projection
  0.065 vs 0.616 — lower is better), membership AUROC (1.00 vs 0.95), and
  intrinsic-dim recovery (matched recovers full dim; flat under-uses it).
- **The flat obstruction is about efficiency + chart quality, not absolute
  fittability.** Capacity control: a flat R¹ latent *can* drive circle FVU to
  ~0 — but needs width 512+/depth 4+/2000 epochs, where matched S¹ gets there
  at width 64. And flat always carries a **seam** (see fig4: circle-flat latent
  collapses to a colored line; circle-matched traces a clean ring).
- **Membership residual is a clean gate** (AUROC ≈ 1.0) — the Step-2 gating
  signal is there. (Caveat under verification: off-manifold probes are full
  Gaussians; a harder near-manifold probe is being tested.)
- **Torus is the laggard** (matched FVU 0.094) — underfit at default capacity;
  motivates the depth/expressiveness sweep.

**Figures:**
- `fig1_fvu_flat_vs_matched.png` — recon error, flat vs matched, per regime.
- `fig2_ae_vs_pca.png` — how many × better than the linear baseline.
- `fig3_k_sensitivity.png` — flat latent dim vs error.
- `fig4_faithfulness.png` — learned latent colored by true coordinate.

**Next:** (2) LR + depth/width expressiveness sweep; (3) higher-dim manifolds
(S^k, T^k, flat for k=5,10,25); (4) 3D visuals of true manifold + learned fit;
(5) adversarial verification of the harness + claims.
