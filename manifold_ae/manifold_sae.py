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


def reactivating_loss(pre):
    """REACTIVATING loss (dead-component aux loss; see TERMINOLOGY.md), shared by BOTH gate
    levels (presence gpre and per-dim rank dim_bias) -- the committed companion to STEHeaviside:
    every gate pre-activation gets a Heaviside forward + this loss. Pushes any pre-activation
    that has gone negative back up toward the gate boundary (0) so a switched-off component
    stays reachable by gradient instead of freezing past the STE window. Returns the per-element
    penalty relu(-pre); the caller applies its own reduction (presence: sum over atoms then mean
    over batch; per-dim: sum over the static (N,K))."""
    return torch.relu(-pre)


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
                 enc_dims=(256, 128, 64), jump_eps=2.0, learn_rank=False,
                 gate_grad="rect"):
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

        # VARIABLE-DEPTH funnel: enc_dims is any length >= 1 (e.g. (128,64,32) classic;
        # (128,64,32,32) adds a layer at the NARROW end). Decoder mirrors it.
        dims = [d_model] + list(enc_dims)
        self.encs = nn.ModuleList(BatchedLinear(N, dims[i], dims[i + 1])
                                  for i in range(len(enc_dims)))
        self.coord = BatchedLinear(N, dims[-1], self.max_rank)
        self.gate = BatchedLinear(N, dims[-1], 1)
        rdims = [self.max_rank] + list(reversed(list(enc_dims))) + [d_model]
        self.decs = nn.ModuleList(BatchedLinear(N, rdims[i], rdims[i + 1])
                                  for i in range(len(rdims) - 1))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # JumpReLU STE bandwidth, in RAW pre-activation units (gpre>0 is the gate boundary).
        self.jump_eps = jump_eps
        # gate_grad: backward estimator for BOTH gate levels (presence + learn-rank dim bias).
        # "rect" = rectangular STE of width jump_eps; "sigmoid" = sigmoid' surrogate (forward
        # stays exact-hard), nonzero EVERYWHERE -- candidate to replace the reactivating losses.
        assert gate_grad in ("rect", "sigmoid")
        self.gate_grad = gate_grad
        # learn_rank: each atom learns WHICH of its max_rank latent dims to use, via a static
        # per-dim bias gated by the same straight-through Heaviside (bias>0 -> dim on). The pool's
        # rank then becomes the MAX rank; the effective rank emerges. init at +0.5 (inside the STE
        # window) so all dims start ON but remain prunable by the sparsity penalty.
        self.learn_rank = learn_rank
        if learn_rank:
            self.dim_bias = nn.Parameter(torch.full((N, self.max_rank), 0.5))

    def _gate(self, pre):
        """{0,1} gate from a raw pre-activation. Forward is the exact Heaviside either way;
        backward depends on gate_grad: rect = rectangular pseudo-gradient of width jump_eps
        (zero outside the window); sigmoid = sigmoid'(pre/T) with T = jump_eps/4 (matches the
        rect's peak gradient 1/eps at the boundary) -- nonzero everywhere, so components driven
        far negative keep receiving gradient without a reactivating loss."""
        if self.gate_grad == "sigmoid":
            s = torch.sigmoid(pre / (self.jump_eps / 4))
            return (pre > 0).to(pre.dtype) + s - s.detach()
        return STEHeaviside.apply(pre, self.jump_eps)

    def rank_gate(self):
        """(N, max_rank) {0,1} mask of which latent dims each atom uses. Fixed pool -> static
        rank_mask; learn_rank -> straight-through Heaviside on a learned per-dim bias (bias>0 = on)."""
        if self.learn_rank:
            return self._gate(self.dim_bias)
        return self.rank_mask

    def atom_ranks(self):
        """Per-atom rank = number of on dims. LEARNED, differentiable count when learn_rank, else
        the static pool rank. Use THIS for all rank logging + the sparsity penalty -- NEVER max_rank."""
        if self.learn_rank:
            return self.rank_gate().sum(-1)
        return self.ranks.float()

    def encode(self, x):
        h = x
        for lin in self.encs:
            h = F.gelu(lin(h))                        # (B, N, enc_dims[-1])
        z = self.coord(h) * self.rank_gate()          # (B, N, max_rank), masked to (learned) rank
        gpre = self.gate(h).squeeze(-1)               # (B, N) RAW gate pre-activation (no sigmoid)
        return z, gpre

    def decode_all(self, z):
        h = z
        for lin in self.decs[:-1]:
            h = F.gelu(lin(h))
        return self.decs[-1](h)                       # (B, N, d_model)

    def forward_jump(self, x):
        """Canonical pure-binary JumpReLU gate: threshold the RAW gate pre-activation at 0
        with a straight-through Heaviside (bandwidth jump_eps, in pre-activation units).
        No sigmoid, no theta -- gpre>0 IS the boundary. The reconstruction uses the {0,1}
        mask DIRECTLY (active_i * dec_i): the gate only SELECTS, all magnitude lives in the
        decoder, so soft==hard by construction. Returns x_hat, z, the raw pre-activation
        gpre (the pre-act/dead-atom loss pushes gpre up toward 0), active (0/1), dec, and
        the per-sample rank-weighted L0."""
        z, gpre = self.encode(x - self.b_dec)
        active = self._gate(gpre)                          # H(gpre > 0); backward per gate_grad
        dec = self.decode_all(z)                           # (B, N, d_model)
        x_hat = (active.unsqueeze(-1) * dec).sum(dim=1) + self.b_dec
        l0 = (active * self.atom_ranks()).sum(dim=-1)      # (B,) rank-weighted dof (learned rank if on)
        return x_hat, z, gpre, active, dec, l0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = ManifoldSAE(d_model=64, rank_dist={4: 32}, enc_dims=(128, 64, 32), learn_rank=True)
    x = torch.randn(16, 64)
    x_hat, z, gpre, active, dec, l0 = m.forward_jump(x)
    print(f"params={m.n_params()/1e6:.1f}M  x_hat={tuple(x_hat.shape)}  "
          f"atoms/sample={active.sum(1).float().mean():.1f}  "
          f"rank-dof/sample={l0.mean():.1f}")
