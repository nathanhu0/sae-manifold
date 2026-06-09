"""Manifold zoo + sparse-mixture generator (synthetic benchmark).

8 manifold types x 6 variants = 48 instances. Each instance i has intrinsic dim
d_i, embedding dim k_i, a parametric embedding gamma_i(theta) in R^{k_i}, a
per-instance isotropic normalization to RMS-norm 1, and a random orthonormal
ambient embedding V_i (k_i x d, orthonormal ROWS; norm-preserving).

Generative model (eq. 13):
    x = sum_{i in S} gammatilde_i(theta_i) @ V_i + eps,   |S| = L0
S drawn uniformly without replacement over the 48 instances; theta_i uniform on
manifold i; eps ~ N(0, sigma_eps^2 I_d) with sigma_eps tiny (geometry, not
denoising).

Normalization (the design choice that makes every instance contribute equally to
the loss): for each instance, draw a 50k calibration sample of the raw embedding,
take mean mu_i and RMS norm sigma_i = sqrt(E||gamma - mu||^2), set
gammatilde = (gamma - mu)/sigma. After this every instance has RMS norm exactly 1,
regardless of type/variant scale. (This is the general form of the per-coordinate
noise-scaling lesson: contributions must be comparable in NORM, not per-coord.)

di vs ki: di governs the manifold's free parameters (expected # localized
detectors in the tiling regime); ki is the ambient subspace dim (atoms needed for
subspace capture). Circle: di=1, ki=2 -> two atoms to span. Torus: di=2, ki=4.

theta is exposed (sample_full / embed_theta): the intrinsic coordinate is the
ground truth for per-manifold capture/seam visualization. theta-sampling is
factored out from the embedding so the rng draw order is IDENTICAL to the prior
internal-sampling version -> mu/sigma/V and all training data are unchanged.
"""
import numpy as np

PI = np.pi

# ── variant parameter sets ────────────────────────────────────────────────────
_R6 = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
_TORUS = [(2, 0.5), (2, 1.0), (3, 1.0), (3, 0.5), (2, 1.5), (3, 2.0)]
_MOBIUS = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
_SWISS = [(2 * PI, 1.5), (2.5 * PI, 2.5), (3 * PI, 3.0),
          (3.5 * PI, 4.0), (4 * PI, 5.0), (4.5 * PI, 6.0)]
_HELIX = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# ── theta samplers: draw n intrinsic coords -> (n, d_i). rng draw order matches
#    the prior in-sampler draws exactly (so calibration / V / training are stable).
def _th_circle(n, r, rng):       return rng.uniform(0, 2 * PI, n)[:, None]
def _th_sphere(n, r, rng):
    u = rng.uniform(-1, 1, n); t = rng.uniform(0, 2 * PI, n)
    return np.stack([u, t], 1)
def _th_torus(n, Rr, rng):
    t = rng.uniform(0, 2 * PI, n); p = rng.uniform(0, 2 * PI, n)
    return np.stack([t, p], 1)
def _th_mobius(n, w, rng):
    phi = rng.uniform(0, 2 * PI, n); t = rng.uniform(-w, w, n)
    return np.stack([phi, t], 1)
def _th_swiss(n, th_h, rng):
    u = rng.uniform(0, 1, n); h = rng.uniform(0, th_h[1], n)
    return np.stack([u, h], 1)
def _th_helix(n, alpha, rng):    return rng.uniform(0, 6 * PI, n)[:, None]
def _th_disk(n, R, rng):
    u = rng.uniform(0, 1, n); t = rng.uniform(0, 2 * PI, n)
    return np.stack([u, t], 1)
def _th_segment(n, L, rng):      return rng.uniform(0, L, n)[:, None]


# ── embeddings: explicit intrinsic coords theta (n, d_i) -> raw (n, k_i) ────────
def _emb_circle(th, r):
    t = th[:, 0]
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)
def _emb_sphere(th, r):
    u, t = th[:, 0], th[:, 1]
    s = np.sqrt(np.clip(1 - u * u, 0, 1))
    return np.stack([r * s * np.cos(t), r * s * np.sin(t), r * u], 1)
def _emb_torus(th, Rr):
    R, r = Rr; t, p = th[:, 0], th[:, 1]
    return np.stack([R * np.cos(t), R * np.sin(t), r * np.cos(p), r * np.sin(p)], 1)
def _emb_mobius(th, w):
    phi, t = th[:, 0], th[:, 1]
    a = 1.0 + t * np.cos(phi / 2)
    return np.stack([a * np.cos(phi), a * np.sin(phi), t * np.sin(phi / 2)], 1)
def _emb_swiss(th, th_h):
    th_max = th_h[0]; u, h = th[:, 0], th[:, 1]
    ang = np.sqrt(u * (th_max ** 2 - PI ** 2) + PI ** 2)   # area-uniform over [pi, th_max]
    return np.stack([ang * np.cos(ang), h, ang * np.sin(ang)], 1)
def _emb_helix(th, alpha):
    t = th[:, 0]
    return np.stack([np.cos(t), np.sin(t), alpha * t], 1)
def _emb_disk(th, R):
    u, t = th[:, 0], th[:, 1]
    rad = R * np.sqrt(u)
    return np.stack([rad * np.cos(t), rad * np.sin(t)], 1)
def _emb_segment(th, L):
    return th[:, :1]


# name, intrinsic d_i, embedding k_i, variants, theta-sampler, embedding
TYPES = [
    ("circle",     1, 2, _R6,    _th_circle,  _emb_circle),
    ("sphere",     2, 3, _R6,    _th_sphere,  _emb_sphere),
    ("torus",      2, 4, _TORUS, _th_torus,   _emb_torus),
    ("mobius",     2, 3, _MOBIUS, _th_mobius, _emb_mobius),
    ("swiss_roll", 2, 3, _SWISS, _th_swiss,   _emb_swiss),
    ("helix",      1, 3, _HELIX, _th_helix,   _emb_helix),
    ("flat_disk",  2, 2, _R6,    _th_disk,    _emb_disk),
    ("segment",    1, 1, _R6,    _th_segment, _emb_segment),
]


class Instance:
    def __init__(self, name, typ, di, ki, params, th_fn, emb_fn, d, rng, calib_n=50_000):
        self.name, self.type, self.di, self.ki, self.params = name, typ, di, ki, params
        self._th, self._emb = th_fn, emb_fn
        calib = emb_fn(th_fn(calib_n, params, rng), params)     # same rng draws as before
        self.mu = calib.mean(0).astype(np.float32)
        self.sigma = float(np.sqrt(((calib - self.mu) ** 2).sum(1).mean()))
        G = rng.standard_normal((d, ki))
        Q, _ = np.linalg.qr(G)                  # (d, ki), orthonormal columns
        self.V = Q.T.astype(np.float32)         # (ki, d), orthonormal rows

    def gtilde(self, n, rng):
        raw = self._emb(self._th(n, self.params, rng), self.params)
        return ((raw - self.mu) / self.sigma).astype(np.float32)

    def embed_theta(self, theta):
        """Explicit intrinsic coords (n, d_i) -> normalized ambient (n, d)."""
        raw = self._emb(np.asarray(theta, np.float64), self.params)
        return (((raw - self.mu) / self.sigma) @ self.V).astype(np.float32)

    def sample_full(self, n, rng):
        """-> (theta (n, d_i), ambient (n, d)). theta is the ground-truth coord."""
        theta = self._th(n, self.params, rng)
        raw = self._emb(theta, self.params)
        amb = ((raw - self.mu) / self.sigma) @ self.V
        return theta.astype(np.float32), amb.astype(np.float32)


class ManifoldZoo:
    def __init__(self, d=256, seed=0, variants_per_type=None):
        """variants_per_type: use only the first k variants of each type (None = all 6).
        e.g. variants_per_type=2 -> 8 types x 2 = 16 distinct manifolds (a simpler zoo)."""
        self.d = d
        rng = np.random.default_rng(seed)
        self.instances = []
        for name, di, ki, variants, th_fn, emb_fn in TYPES:
            vs = variants if variants_per_type is None else variants[:variants_per_type]
            for vi, params in enumerate(vs):
                self.instances.append(
                    Instance(f"{name}_{vi}", name, di, ki, params, th_fn, emb_fn, d, rng))
        self.n_inst = len(self.instances)

    def sample(self, batch, L0, rng, sigma_eps=1e-5, return_truth=False, return_theta=False,
               p_active=None):
        """x, masks [, truth (batch,n_inst,d)] [, theta (batch,n_inst,max_di)].
        truth[b,i]/theta[b,i] are populated only for active i. Backward compatible:
        (x,masks) and (x,masks,truth) contracts are unchanged.

        p_active: if set, each manifold is present INDEPENDENTLY with this probability
        (Bernoulli) -> the active count is Binomial(n_inst, p_active), a distribution
        rather than a fixed L0. (L0 is ignored in this mode.)"""
        d, n_inst = self.d, self.n_inst
        max_di = max(inst.di for inst in self.instances)
        x = np.zeros((batch, d), np.float32)
        if p_active is not None:
            masks = rng.random((batch, n_inst)) < p_active   # independent presence
        else:
            rr = rng.random((batch, n_inst))
            active = np.argsort(rr, axis=1)[:, :L0]          # L0 distinct per row
            masks = np.zeros((batch, n_inst), bool)
            masks[np.arange(batch)[:, None], active] = True
        truth = np.zeros((batch, n_inst, d), np.float32) if return_truth else None
        theta = np.zeros((batch, n_inst, max_di), np.float32) if return_theta else None
        for i, inst in enumerate(self.instances):
            sel = np.where(masks[:, i])[0]
            if len(sel) == 0:
                continue
            th_i, contrib = inst.sample_full(len(sel), rng)   # same rng draws as gtilde@V
            x[sel] += contrib
            if return_truth:
                truth[sel, i] = contrib
            if return_theta:
                theta[sel, i, :inst.di] = th_i
        if sigma_eps:
            x += (sigma_eps * rng.standard_normal((batch, d))).astype(np.float32)
        out = (x, masks)
        if return_truth:
            out = out + (truth,)
        if return_theta:
            out = out + (theta,)
        return out


def _self_check():
    zoo = ManifoldZoo(d=256, seed=0)
    print(f"{'instance':14s} di ki   sigma   ||gt||rms  V@Vt-I   embed_theta   params")
    rng = np.random.default_rng(7)
    bad = 0
    for inst in zoo.instances:
        gt = inst.gtilde(20_000, rng)
        rms = float(np.sqrt((gt ** 2).sum(1).mean()))           # should be ~1
        orth = float(np.abs(inst.V @ inst.V.T - np.eye(inst.ki)).max())
        contrib = gt @ inst.V
        norm_pres = float(np.abs(np.linalg.norm(contrib, axis=1)
                                 - np.linalg.norm(gt, axis=1)).max())
        # sample_full consistency: theta -> embed_theta reproduces the ambient point
        th, amb = inst.sample_full(2000, np.random.default_rng(3))
        amb2 = inst.embed_theta(th)
        recon = float(np.abs(amb - amb2).max())
        ok = abs(rms - 1) < 0.05 and orth < 1e-4 and norm_pres < 1e-3 and recon < 1e-5
        bad += not ok
        flag = "" if ok else "  <-- FAIL"
        print(f"{inst.name:14s} {inst.di}  {inst.ki}  {inst.sigma:7.3f}  "
              f"{rms:7.4f}   {orth:.1e}  {recon:.1e}     {inst.params}{flag}")
    # mixture sanity: E||x||^2 ~ L0 (each contribution has unit expected sq-norm)
    for L0 in (1, 4, 8):
        x, masks = zoo.sample(4096, L0, np.random.default_rng(L0), sigma_eps=0.0)
        print(f"L0={L0}: mean||x||^2={float((x**2).sum(1).mean()):.3f} (expect ~{L0}), "
              f"active/sample={float(masks.sum(1).mean()):.1f}")
    print(f"\n{zoo.n_inst} instances, sum k_i = {sum(i.ki for i in zoo.instances)}, "
          f"FAILS = {bad}")


if __name__ == "__main__":
    _self_check()
