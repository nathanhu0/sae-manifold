"""Mixture-of-manifolds SAE (MOLT-style, with nonlinear manifold atoms).

Each atom is a fully per-atom nonlinear manifold autoencoder:
  encoder funnel:  d_model -> 256 -> 128 -> 64        (GELU)
     coordinate head: Linear(64 -> max_rank)          -> z_i (masked to atom's rank)
     gate head:       Linear(64 -> 1) -> sigmoid      -> g_i in [0,1] (on/off presence)
  decoder (mirror):   max_rank -> 64 -> 128 -> 256 -> d_model (GELU)

Atoms come from a NON-UNIFORM rank pool (e.g. {1:128, 2:64, 3:32, 4:16}); each
atom's coordinate is masked to its own rank, so a rank-r atom is a curved
r-manifold. This mirrors MOLT (transformer-circuits 2025): variable rank beats
uniform, and sparsity is budgeted by COMPLEXITY, not atom count.

Sparsity = RANK-BUDGETED TopK: per sample, take highest-presence atoms until their
summed rank reaches R (total active degrees of freedom). A rank-3 atom costs 3x a
rank-1, so the model only spends on high-rank manifolds where the data earns it.
R is annealed (full -> target, e.g. 416 -> 100) by the trainer (sparsity warmup).

Output: x_hat = b_dec + sum_{active} g_i * dec_i(z_i)

DESIGN NOTES (debug-scale; easy to change):
  * gate off the 64-d layer -> all N encoders run every input (only decode is sparse).
  * fully per-atom (~2.2M/atom); shared-trunk / conditional-decoder = scale path.
  * dense decode + mask (mem ~B*N*d_model); gather-active is the scale optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STEHeaviside(torch.autograd.Function):
    """Heaviside H(a) with a straight-through rectangular pseudo-gradient of width
    `eps` centered at the threshold boundary (a=0). Gives gradient to both the
    presence and the threshold for atoms whose presence sits within eps/2 of their
    threshold (the JumpReLU estimator, Rajamanoharan et al. 2024)."""

    @staticmethod
    def forward(ctx, a, eps):
        ctx.save_for_backward(a)
        ctx.eps = eps
        return (a > 0).to(a.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (a,) = ctx.saved_tensors
        pseudo = (a.abs() < ctx.eps / 2).to(a.dtype) / ctx.eps
        return grad_out * pseudo, None


def preact_revival(pre):
    """Revival / dead-component aux loss, shared by BOTH gate levels (presence gpre and per-dim
    rank dim_bias) -- the committed companion to STEHeaviside: every gate pre-activation gets a
    Heaviside forward + this revival. Pushes any pre-activation that has gone negative back up
    toward the gate boundary (0) so a component driven off stays revivable instead of freezing
    past the STE window. Returns the per-element penalty relu(-pre); the caller applies its own
    reduction (presence: sum over atoms then mean over batch; per-dim: sum over the static (N,K))."""
    return torch.relu(-pre)


# --- SET ASIDE (commented out while focusing on the binary pre-activation JumpReLU gate) ---
# Leaky-hard-sigmoid STEs for the SPD causal-importance gate (used by forward_clamp, also
# parked below). Kept verbatim for reference.
#
# class LowerLeakyHardSigmoid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, z, alpha):
#         ctx.save_for_backward(z); ctx.alpha = alpha
#         return z.clamp(0.0, 1.0)
#     @staticmethod
#     def backward(ctx, g):
#         (z,) = ctx.saved_tensors
#         grad = g * ((z > 0) & (z < 1)).to(z.dtype)
#         grad = grad + ctx.alpha * g * ((z <= 0) & (g < 0)).to(z.dtype)   # revive, neg-grad only
#         return grad, None
#
# class UpperLeakyHardSigmoid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, z, alpha):
#         ctx.save_for_backward(z); ctx.alpha = alpha
#         return z.clamp(0.0, 1.0)
#     @staticmethod
#     def backward(ctx, g):
#         (z,) = ctx.saved_tensors
#         grad = g * ((z > 0) & (z < 1)).to(z.dtype)
#         grad = grad + ctx.alpha * g * (z >= 1).to(z.dtype)
#         return grad, None
# -------------------------------------------------------------------------------------------


class BatchedLinear(nn.Module):
    """N independent affine maps in_f->out_f. Input (B, in_f) [shared] or
    (B, N, in_f); output always (B, N, out_f)."""

    def __init__(self, n, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(n, out_f))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        eq = "noi,bi->bno" if x.dim() == 2 else "noi,bni->bno"
        return torch.einsum(eq, self.weight, x) + self.bias


class ManifoldSAE(nn.Module):
    def __init__(self, d_model=4096, rank_dist=None, R_target=100,
                 enc_dims=(256, 128, 64), jump_eps=2.0, learn_rank=False):
        super().__init__()
        rank_dist = rank_dist or {1: 128, 2: 64, 3: 32, 4: 16}
        ranks = [r for r, c in sorted(rank_dist.items()) for _ in range(c)]
        self.register_buffer("ranks", torch.tensor(ranks, dtype=torch.long))
        N = len(ranks)
        self.n_atoms, self.max_rank, self.R_target = N, max(ranks), R_target
        self.full_R = int(self.ranks.sum())
        # per-atom coordinate mask: rank-r atom keeps its first r coordinate dims
        rmask = torch.zeros(N, self.max_rank)
        for i, r in enumerate(ranks):
            rmask[i, :r] = 1.0
        self.register_buffer("rank_mask", rmask)

        e1, e2, e3 = enc_dims
        self.enc1 = BatchedLinear(N, d_model, e1)
        self.enc2 = BatchedLinear(N, e1, e2)
        self.enc3 = BatchedLinear(N, e2, e3)
        self.coord = BatchedLinear(N, e3, self.max_rank)
        self.gate = BatchedLinear(N, e3, 1)
        self.dec1 = BatchedLinear(N, self.max_rank, e3)
        self.dec2 = BatchedLinear(N, e3, e2)
        self.dec3 = BatchedLinear(N, e2, e1)
        self.dec4 = BatchedLinear(N, e1, d_model)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # JumpReLU STE bandwidth, in RAW pre-activation units (gpre>0 is the gate boundary).
        self.jump_eps = jump_eps
        # learn_rank: each atom learns WHICH of its max_rank latent dims to use, via a static
        # per-dim bias gated by the same straight-through Heaviside (bias>0 -> dim on). The pool's
        # rank then becomes the MAX rank; the effective rank emerges. init at +0.5 (inside the STE
        # window) so all dims start ON but remain prunable by the sparsity penalty.
        self.learn_rank = learn_rank
        if learn_rank:
            self.dim_bias = nn.Parameter(torch.full((N, self.max_rank), 0.5))

    def rank_gate(self):
        """(N, max_rank) {0,1} mask of which latent dims each atom uses. Fixed pool -> static
        rank_mask; learn_rank -> straight-through Heaviside on a learned per-dim bias (bias>0 = on)."""
        if self.learn_rank:
            return STEHeaviside.apply(self.dim_bias, self.jump_eps)
        return self.rank_mask

    def atom_ranks(self):
        """Per-atom rank = number of on dims. LEARNED, differentiable count when learn_rank, else
        the static pool rank. Use THIS for all rank logging + the sparsity penalty -- NEVER max_rank."""
        if self.learn_rank:
            return self.rank_gate().sum(-1)
        return self.ranks.float()

    def encode(self, x):
        h = F.gelu(self.enc1(x))
        h = F.gelu(self.enc2(h))
        h = F.gelu(self.enc3(h))                      # (B, N, e3)
        z = self.coord(h) * self.rank_gate()          # (B, N, max_rank), masked to (learned) rank
        gpre = self.gate(h).squeeze(-1)               # (B, N) RAW gate pre-activation (no sigmoid)
        return z, gpre

    def decode_all(self, z):
        h = F.gelu(self.dec1(z))
        h = F.gelu(self.dec2(h))
        h = F.gelu(self.dec3(h))
        return self.dec4(h)                           # (B, N, d_model)

    # --- SET ASIDE (hard rank-budget TopK + sigmoid-presence JumpReLU) ---
    # These assume the OLD sigmoid-presence encode() contract and are unused by the binary
    # gate below; commented out while we focus on the one mechanism. Verbatim for reference:
    #
    # def select(self, g, R):  # rank-budgeted TopK: highest-presence atoms until sum rank >= R
    #     order = g.argsort(dim=-1, descending=True)
    #     sorted_ranks = self.ranks[order].float()
    #     start = sorted_ranks.cumsum(-1) - sorted_ranks
    #     active_sorted = (start < R).float()
    #     return torch.zeros_like(g).scatter_(-1, order, active_sorted)
    #
    # def forward(self, x, R=None):                      # hard rank-budget TopK forward (g*dec)
    #     R = self.R_target if R is None else R
    #     z, g = self.encode(x - self.b_dec)
    #     gate = g * self.select(g, R)
    #     x_hat = (gate.unsqueeze(-1) * self.decode_all(z)).sum(dim=1) + self.b_dec
    #     return x_hat, z, gate
    #
    # def select_jumprelu(self, g, theta=None):          # sigmoid-presence + learned/annealed theta
    #     if theta is None: theta = torch.sigmoid(self.theta_raw)
    #     return STEHeaviside.apply(g - theta, self.jump_eps)
    # ----------------------------------------------------------------------

    def forward_jump(self, x):
        """Canonical pure-binary JumpReLU gate: threshold the RAW gate pre-activation at 0
        with a straight-through Heaviside (bandwidth jump_eps, in pre-activation units).
        No sigmoid, no theta -- gpre>0 IS the boundary. The reconstruction uses the {0,1}
        mask DIRECTLY (active_i * dec_i): the gate only SELECTS, all magnitude lives in the
        decoder, so soft==hard by construction. Returns x_hat, z, the raw pre-activation
        gpre (the pre-act/dead-atom loss pushes gpre up toward 0), active (0/1), dec, and
        the per-sample rank-weighted L0."""
        z, gpre = self.encode(x - self.b_dec)
        active = STEHeaviside.apply(gpre, self.jump_eps)   # H(gpre > 0); bandwidth in pre-act units
        dec = self.decode_all(z)                           # (B, N, d_model)
        x_hat = (active.unsqueeze(-1) * dec).sum(dim=1) + self.b_dec
        l0 = (active * self.atom_ranks()).sum(dim=-1)      # (B,) rank-weighted dof (learned rank if on)
        return x_hat, z, gpre, active, dec, l0

    # --- SET ASIDE (implicit-k sigmoid lasso, and SPD leaky-hard-sigmoid) ---
    # Both assume the OLD sigmoid-presence / tau encode() contract or the leaky STEs above;
    # unused by the binary gate. Commented out for focus; verbatim for reference:
    #
    # def forward_soft(self, x, tau=1.0):                # implicit-k: g=sigmoid(a/tau), lasso on ||g*dec||
    #     z, g = self.encode(x - self.b_dec, tau=tau)
    #     contrib = g.unsqueeze(-1) * self.decode_all(z)
    #     x_hat = contrib.sum(dim=1) + self.b_dec
    #     return x_hat, z, g, contrib.norm(dim=-1)
    #
    # def forward_clamp(self, x, alpha=0.01):            # SPD leaky-hard-sigmoid gate
    #     h = F.gelu(self.enc3(F.gelu(self.enc2(F.gelu(self.enc1(x - self.b_dec))))))
    #     zc = self.coord(h) * self.rank_mask
    #     zg = self.gate(h).squeeze(-1)
    #     dec = self.decode_all(zc)
    #     g_rec = LowerLeakyHardSigmoid.apply(zg, alpha); g_pen = UpperLeakyHardSigmoid.apply(zg, alpha)
    #     x_hat = (g_rec.unsqueeze(-1) * dec).sum(dim=1) + self.b_dec
    #     return x_hat, zc, g_rec, (g_pen.unsqueeze(-1) * dec).norm(dim=-1)
    # ------------------------------------------------------------------------

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = ManifoldSAE()
    x = torch.randn(16, 4096)
    for R in (m.full_R, 200, 100):
        xh, z, gate = m(x, R=R)
        active = (gate > 0)
        used_rank = (active.float() * m.ranks.float()).sum(1).mean()
        print(f"R={R:>3}  params={m.n_params()/1e6:.0f}M  x_hat={tuple(xh.shape)}  "
              f"atoms/sample={active.sum(1).float().mean():.1f}  "
              f"rank-used/sample={used_rank:.1f}")
