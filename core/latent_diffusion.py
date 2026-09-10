"""Latent diffusion: non-recursive generative head over latent windows.

Rationale (2026-09 design pass; see bake-off evidence in
/Users/yanlu/Documents/elegans/bakeoff):
  recursive per-step samplers accumulate error over long rollouts (DeepAR
  exploded, TSMixer collapsed); segment-level diffusion conditioned on real
  context holds variance and marginals over long horizons.  Conditioning on
  an rSLDS-style state path (extra channels) further improves long-horizon
  range coverage.

This module is dimension-generic (works on any latent dim D, e.g. 1024) and
provides the "imagination" head for Brain MoE-PINN: fit on windows of the
latent trajectory z, then sample continuations conditioned on a real context
window.  Optional extra conditioning channels (e.g., a one-hot state path)
can be passed at fit/sample time via cond (T, Dc) per window.

Decoding back to observations is the caller's job (existing decoders).
"""


import torch
from torch import nn
import torch.nn.functional as Fn


class _ResBlock1d(nn.Module):
    def __init__(self, hid: int, dil: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hid, hid, 5, padding=2 * dil, dilation=dil),
            nn.SiLU(),
            nn.Conv1d(hid, hid, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x, scale, shift):
        return self.net(x) * scale + shift


def ddpm_posterior_mean(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    alpha_t: torch.Tensor,
    abar_t: torch.Tensor,
    abar_prev: torch.Tensor,
    beta_t: torch.Tensor,
) -> torch.Tensor:
    """Return the standard DDPM posterior mean ``E[x_{t-1}|x_t,x_0]``."""
    return (
        torch.sqrt(alpha_t) * (1 - abar_prev) / (1 - abar_t) * x_t
        + torch.sqrt(abar_prev) * beta_t / (1 - abar_t) * x0
    )


class LatentDiffusion(nn.Module):
    """DDPM over latent windows with context substitution.

    fit(): train on windows (n, LEN, D) [+ cond (n, LEN, Dc)].
    sample(context (Lc, D), length, n_roll, cond=None): chained segments of
    CONTEXT+TARGET, real context re-injected every reverse step.
    """

    def __init__(self, dim: int, context: int = 30, target: int = 96,
                 t_steps: int = 100, hidden: int = 48,
                 cond_dim: int = 0, seed: int = 0):
        super().__init__()
        self.D = dim
        self.Lc = context
        self.Lt = target
        self.LEN = context + target
        self.T = t_steps
        self.cond_dim = cond_dim
        torch.manual_seed(seed)
        in_ch = dim + cond_dim

        beta = torch.linspace(1e-4, 0.02, t_steps)
        self.register_buffer("beta", beta)
        alpha = 1 - beta
        self.register_buffer("abar", torch.cumprod(alpha, 0))

        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.tmlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(),
                                  nn.Linear(32, 2 * 4 * hidden))
        self.blocks = nn.ModuleList(
            [_ResBlock1d(hidden, d) for d in (1, 2, 4, 8)])
        self.out = nn.Conv1d(hidden, dim, 1)

    # -- forward: predict noise -------------------------------------------------
    def forward(self, x_noisy, cond, t):
        """x_noisy: (B, D, L); cond: (B, Dc, L) or None; t: (B,)."""
        te = self.tmlp(t[:, None])
        g = te.chunk(8, 1)
        if cond is not None:
            x = torch.cat([x_noisy, cond], dim=1)
        else:
            x = x_noisy
        h = self.inp(x)
        skips = []
        for i, blk in enumerate(self.blocks):
            z = blk(h, g[2 * i][:, :, None], g[2 * i + 1][:, :, None])
            skips.append(z)
            h = h + z
        return self.out(sum(skips) / len(skips))

    # -- training ---------------------------------------------------------------
    def fit(self, windows: torch.Tensor, epochs: int = 200, batch: int = 32,
            cond: torch.Tensor = None, lr: float = 2e-4, progress=True):
        """windows: (n, LEN, D); cond: (n, LEN, Dc) or None."""
        n = windows.shape[0]
        x = windows.transpose(1, 2).to(windows.device)          # (n, D, LEN)
        c = cond.transpose(1, 2) if cond is not None else None  # (n, Dc, LEN)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n_batch = max(1, n // batch)
        losses = []
        self.train()
        for _ in range(epochs):
            tot = 0.0
            for b in range(n_batch):
                xb = x[b::n_batch]
                B = xb.shape[0]
                t = torch.randint(0, self.T, (B,), device=x.device)
                eps = torch.randn_like(xb)
                a = self.abar[t][:, None, None]
                xnoisy = torch.sqrt(a) * xb + torch.sqrt(1 - a) * eps
                cb = c[b::n_batch] if c is not None else None
                pred = self(xnoisy, cb, t.float())
                loss = Fn.mse_loss(pred, eps)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            losses.append(tot / n_batch)
            if progress and (_ + 1) % 50 == 0:
                print(f"  diffusion epoch {_ + 1}/{epochs} loss {losses[-1]:.4f}")
        return losses

    # -- sampling ----------------------------------------------------------------
    @torch.no_grad()
    def sample(self, context, length: int, n_roll: int = 4,
               cond=None, seed: int = 0, progress=True, device=None):
        """context: (Lc, D) real latent context; returns (n_roll*segs, D) tensor.

        cond: per-segment extra channels (LEN, Dc) or None.
        All computation stays on `device` (no numpy round-trips).
        """
        torch.manual_seed(int(seed))
        dev = device or context.device if torch.is_tensor(context) \
            else torch.device("cpu")
        ctx = torch.as_tensor(context, dtype=torch.float32, device=dev)
        all_r = []
        nseg = (length + self.Lt - 1) // self.Lt
        for r in range(n_roll):
            segs = [ctx.clone()]
            cur = ctx.clone()
            for _ in range(nseg):
                if cur.shape[0] >= self.Lc:
                    cc = cur[-self.Lc:]
                else:
                    pad = cur[-1:].expand(self.Lc - cur.shape[0], -1)
                    cc = torch.cat([pad, cur], dim=0)
                xt = torch.randn(1, self.D, self.LEN, device=dev)
                ctx_t = cc
                cseg = None
                if cond is not None:
                    cseg = torch.as_tensor(cond, dtype=torch.float32,
                                           device=dev)[None].transpose(1, 2)
                # Known-context inpainting uses one forward-process noise
                # sample per segment. Reusing it across reverse steps keeps
                # q(x_t | x_0=context) Markov-consistent.
                ctx_noise = torch.randn_like(ctx_t)
                for tt in range(self.T - 1, -1, -1):
                    tb = torch.full((1,), float(tt), device=dev)
                    eps_p = self(xt, cseg, tb)
                    alpha_t = 1.0 - self.beta[tt]
                    abar_t = self.abar[tt]
                    if tt > 0:
                        abar_prev = self.abar[tt - 1]
                        x0 = (xt - torch.sqrt(1 - abar_t) * eps_p) / torch.sqrt(abar_t)
                        # DDPM posterior mean:
                        # sqrt(alpha_t) * (1-abar_prev)/(1-abar_t) * x_t
                        # + sqrt(abar_prev) * beta_t/(1-abar_t) * x_0
                        mean_xt = ddpm_posterior_mean(
                            xt, x0, alpha_t, abar_t, abar_prev, self.beta[tt])
                        var = ((1 - abar_prev) / (1 - abar_t) * self.beta[tt])
                        xt = mean_xt + torch.sqrt(var.clamp(min=1e-12)) * torch.randn_like(xt)
                    else:
                        xt = (xt - torch.sqrt(1 - abar_t) * eps_p) / torch.sqrt(abar_t)
                    # Context substitution with the same noise realization
                    # used at every reverse step.
                    a_ctx = self.abar[tt]
                    ctx_noisy = torch.sqrt(a_ctx) * ctx_t \
                        + torch.sqrt(1 - a_ctx) * ctx_noise
                    xt[0, :, :self.Lc] = ctx_noisy.T
                seg = xt[0].t()                       # (LEN, D)
                segs.append(seg[self.Lc:])
                cur = torch.cat([cur, seg[self.Lc:]], dim=0)
            all_r.append(torch.cat(segs, dim=0))
            if progress:
                print(f"  diffusion rollout {r + 1}/{n_roll} done")
        return torch.cat(all_r, dim=0)


def make_windows(z, ctx: int, tgt: int, stride: int = 2):
    """Overlapping (ctx+tgt)-long windows of a latent trajectory (torch)."""
    z = torch.as_tensor(z, dtype=torch.float32)
    m = ctx + tgt
    n = z.shape[0]
    idx = torch.arange(0, n - m + 1, stride)
    return torch.stack([z[i:i + m] for i in idx])
