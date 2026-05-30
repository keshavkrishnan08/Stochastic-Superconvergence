"""Learned-score arm (E11): the practical-reach boundary.

Two mechanisms produce a persistent, N-independent score-error FLOOR that masks
super-convergence (the umbrella D(lambda) eps^2 term):

1. miscalibrated_recursion(): a deliberately miscalibrated but still-LINEAR score
   s_hat = -x/(V+Delta).  The variance map stays affine -> EXACT recursion -> exact floor.
   Deterministic, bulletproof; validates that an O(1)-in-N score bias yields an O(1) KL floor.

2. ScoreMLP + train_dsm() + sample_learned(): a real torch-CPU MLP trained by denoising
   score matching on 1-D Gaussian data; vary width/depth/epochs and measure the residual
   score error and the resulting sampler KL floor. (Heavy arm.)
"""
from __future__ import annotations
import mpmath as mp
import numpy as np


# ---- 1. deterministic miscalibrated-score recursion (exact) ----
def miscalibrated_recursion(N, T, B, lam, s2, delta):
    """Reverse EM variance with score s_hat=-x/(V+delta) (variance miscalibrated by delta).
    Drift coefficient uses (V+delta) in the score term; map stays affine -> exact v_N."""
    N = int(N); dt = mp.mpf(T) / N; B = mp.mpf(B); lam = mp.mpf(lam); s2 = mp.mpf(s2)
    delta = mp.mpf(delta); T = mp.mpf(T); one = mp.mpf(1); u = lam ** 2
    V = lambda t: one + (s2 - 1) * mp.e ** (-B * t)
    v = V(T)
    for k in range(N):
        tk = T - k * dt
        Vt = V(tk)
        A = -B / 2 + (one + u) / 2 * B / (Vt + delta)   # miscalibrated score in the drift
        v = (1 - A * dt) ** 2 * v + u * B * dt
    return v


# ---- 2. trained MLP score (torch-CPU) ----
def _make_torch():
    import torch, torch.nn as nn
    torch.set_num_threads(4)

    class ScoreMLP(nn.Module):
        def __init__(self, width=64, depth=3):
            super().__init__()
            layers = [nn.Linear(3, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.SiLU()]
            layers += [nn.Linear(width, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x, t, B):
            abar = torch.exp(-B * t)
            feats = torch.stack([x, t, abar], dim=-1)
            return self.net(feats).squeeze(-1)

    return torch, nn, ScoreMLP


def train_dsm(s2, B, T, width=64, depth=3, epochs=5000, batch=4096, lr=1e-3, seed=0,
              t_min=0.05):
    """Denoising score matching on 1-D N(0,s2). Returns (model, torch, residual_rmse).
    residual_rmse = RMS difference between s_theta and the true marginal score -x/V over a
    held-out (t,x_t) sample (the score-approximation error that sets the floor).
    Stability: sample t in [t_min, T] (the DSM target diverges as t->0), clamp 1-abar,
    grad-clip, and a modest lr -- without these the long runs NaN."""
    torch, nn, ScoreMLP = _make_torch()
    g = torch.Generator().manual_seed(seed)
    model = ScoreMLP(width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    s2t = torch.tensor(float(s2)); Bt = torch.tensor(float(B)); Tt = float(T)
    EPS = 1e-6
    for it in range(epochs):
        t = t_min + torch.rand(batch, generator=g) * (Tt - t_min)
        abar = torch.exp(-Bt * t)
        omab = torch.clamp(1 - abar, min=EPS)          # 1 - abar, floored
        x0 = torch.randn(batch, generator=g) * torch.sqrt(s2t)
        eps = torch.randn(batch, generator=g)
        xt = torch.sqrt(abar) * x0 + torch.sqrt(omab) * eps
        # DSM target: conditional score grad log p(x_t|x_0) = -(x_t - sqrt(abar) x0)/(1-abar)
        target = -(xt - torch.sqrt(abar) * x0) / omab
        pred = model(xt, t, Bt)
        loss = ((pred - target) ** 2 * omab).mean()    # sigma^2-weighted DSM (well-scaled)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    # residual vs the TRUE marginal score -x/V on a fresh sample
    with torch.no_grad():
        t = torch.rand(20000, generator=g) * Tt
        abar = torch.exp(-Bt * t)
        V = abar * s2t + (1 - abar)
        x0 = torch.randn(20000, generator=g) * torch.sqrt(s2t)
        eps = torch.randn(20000, generator=g)
        xt = torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * eps
        true_score = -xt / V
        resid = (model(xt, t, Bt) - true_score)
        rmse = float(torch.sqrt((resid ** 2).mean()))
    return model, torch, rmse


def sample_learned(model, torch, N, T, B, lam, s2, P, seed=0):
    """Reverse EM sampler in forward-running time using the learned score. Returns variance & samples."""
    g = torch.Generator().manual_seed(seed)
    Bt = torch.tensor(float(B)); dt = T / N
    abarT = float(np.exp(-B * T)); VT = abarT * s2 + (1 - abarT)
    x = torch.randn(P, generator=g) * float(np.sqrt(VT))
    c = lam * float(np.sqrt(B)); sq = float(np.sqrt(dt))
    with torch.no_grad():
        for k in range(N):
            tk = torch.full((P,), float(T - k * dt))
            sc = model(x, tk, Bt)
            drift = (B / 2.0) * x + (1.0 + lam ** 2) / 2.0 * B * sc
            x = x + drift * dt + c * sq * torch.randn(P, generator=g)
    xv = x.numpy()
    return float(np.var(xv)), xv


if __name__ == "__main__":
    from diffusion import vp_const, set_dps
    set_dps(40)
    s2, B, T = 2.0, 4.0, 5.0
    print("miscalibrated floor (delta sweep, large N -> N-independent floor):")
    for delta in [0.0, 0.01, 0.05, 0.2]:
        v1 = miscalibrated_recursion(4096, T, B, 1.2705, s2, delta)
        v2 = miscalibrated_recursion(16384, T, B, 1.2705, s2, delta)
        kl1 = float((v1 / s2 - 1 - mp.log(v1 / s2)) / 2)
        kl2 = float((v2 / s2 - 1 - mp.log(v2 / s2)) / 2)
        print(f"  delta={delta:5}: KL(N=4096)={kl1:.3e}  KL(N=16384)={kl2:.3e}  (flat=floor)")
