"""E36 -- Tier 1: literal EDM sampler on a NON-Gaussian 2D target with a TRAINED score.

Closes the scope gap that the Gaussian-only bridge (E35) leaves: here the target is a genuine
non-Gaussian mixture (8 Gaussians on a ring), the score is a small MLP trained by EDM denoising
score matching (Karras et al. 2022 preconditioning), and the sampler is the literal EDM Algorithm 2
(stochastic Heun with the S_churn injection and the rho=7 schedule). We sweep S_churn and measure a
sample-based sliced-Wasserstein distance to held-out target samples.

Two outputs:
  (a) the churn has an interior optimum on a real trained non-Gaussian score (the EDM sweet spot,
      reproduced in a controlled toy);
  (b) the LOCAL-GAUSSIAN prediction -- the closed-form churn solving v_N=s^2 for a Gaussian of the
      data's per-coordinate variance (exact denoiser, same EDM schedule/steps, as in E35) -- tracks
      the measured optimum. This is a consistency check of the mechanism, not an exact formula
      (the exact lambda* is Gaussian-only).

CPU-only, PyTorch. Deterministic seeds; metrics averaged over seeds and projections.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import time
import numpy as np
import io_utils as io

NAME = "e36_edm_toy2d"
SQRT2_MINUS_1 = float(np.sqrt(2) - 1)


# ----------------------------- data: 8 Gaussians on a ring -----------------------------
def sample_target(n, rng, radius=2.0, std=0.2):
    ang = (2 * np.pi / 8) * rng.integers(0, 8, size=n)
    centers = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    return (centers + std * rng.standard_normal((n, 2))).astype(np.float64)


# ----------------------------- EDM preconditioned MLP denoiser -----------------------------
def _build(torch):
    import torch.nn as nn

    class FNet(nn.Module):
        def __init__(self, width=128, depth=4):
            super().__init__()
            layers = [nn.Linear(3, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.SiLU()]
            layers += [nn.Linear(width, 2)]
            self.net = nn.Sequential(*layers)

        def forward(self, x_in, c_noise):
            h = torch.cat([x_in, c_noise[:, None]], dim=1)
            return self.net(h)

    return FNet


def denoise(F, x, sigma, sigma_data, torch):
    """EDM preconditioned denoiser D(x;sigma) = c_skip x + c_out F(c_in x, c_noise)."""
    s = sigma if torch.is_tensor(sigma) else torch.full((x.shape[0],), float(sigma))
    s = s.reshape(-1, 1)
    sd2 = sigma_data ** 2
    c_skip = sd2 / (s ** 2 + sd2)
    c_out = s * sigma_data / torch.sqrt(s ** 2 + sd2)
    c_in = 1.0 / torch.sqrt(s ** 2 + sd2)
    c_noise = (torch.log(s.reshape(-1)) / 4.0)
    return c_skip * x + c_out * F(c_in * x, c_noise)


def train_edm(torch, sigma_data, steps=6000, batch=4096, lr=2e-3, seed=0,
              P_mean=-1.2, P_std=1.2, width=128, depth=4):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F = _build(torch)(width, depth)
    opt = torch.optim.Adam(F.parameters(), lr=lr)
    sd = float(sigma_data)
    loss_steps, loss_vals = [], []
    for it in range(steps):
        x0 = torch.from_numpy(sample_target(batch, rng)).float()
        ln_sigma = P_mean + P_std * torch.randn(batch)
        sigma = torch.exp(ln_sigma)
        n = torch.randn(batch, 2) * sigma[:, None]
        xt = x0 + n
        D = denoise(F, xt, sigma, sd, torch)
        weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
        loss = (weight[:, None] * (D - x0) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(F.parameters(), 10.0)
        opt.step()
        if it % 50 == 0 or it == steps - 1:
            loss_steps.append(it); loss_vals.append(float(loss))
    return F, {"step": loss_steps, "loss": loss_vals}


# ----------------------------- EDM Algorithm 2 sampler (stochastic Heun) -----------------------------
def edm_sigmas(N, sigma_min, sigma_max, rho=7):
    i = np.arange(N)
    a = sigma_max ** (1 / rho) + i / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    return np.concatenate([a ** rho, [0.0]])


def edm_sample(F, torch, sigma_data, N, S_churn, sigma_min, sigma_max, rho, P, seed,
               S_noise=1.0):
    torch.manual_seed(seed)
    sig = edm_sigmas(N, sigma_min, sigma_max, rho)
    x = torch.randn(P, 2) * float(sig[0])
    gcap = SQRT2_MINUS_1
    with torch.no_grad():
        for i in range(N):
            si = float(sig[i]); si1 = float(sig[i + 1])
            gamma = min(S_churn / N, gcap)
            shat = si * (1 + gamma)
            if gamma > 0:
                x = x + np.sqrt(max(shat ** 2 - si ** 2, 0.0)) * S_noise * torch.randn(P, 2)
            d = (x - denoise(F, x, shat, sigma_data, torch)) / shat
            x_next = x + (si1 - shat) * d
            if si1 > 0:
                d2 = (x_next - denoise(F, x_next, si1, sigma_data, torch)) / si1
                x_next = x + (si1 - shat) * (0.5 * d + 0.5 * d2)
            x = x_next
    return x.numpy()


# ----------------------------- sliced-Wasserstein metric -----------------------------
def sliced_w1(a, b, n_proj=256, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.standard_normal((n_proj, 2))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    pa = np.sort(a @ th.T, axis=0)
    pb = np.sort(b @ th.T, axis=0)
    n = min(pa.shape[0], pb.shape[0])
    return float(np.mean(np.abs(pa[:n] - pb[:n])))


# ----------------------------- Gaussian local prediction (exact denoiser, same schedule) -----------------------------
def gaussian_var_recursion(s2, N, S_churn, sigma_min, sigma_max, rho):
    """Per-coordinate terminal variance of the EDM sampler with the EXACT Gaussian denoiser
    a(sigma)=s2/(s2+sigma^2) (linear -> exact variance propagation), matching E35."""
    sig = edm_sigmas(N, sigma_min, sigma_max, rho)
    v = float(sigma_max) ** 2
    gcap = SQRT2_MINUS_1
    for i in range(N):
        si = float(sig[i]); si1 = float(sig[i + 1])
        gamma = min(S_churn / N, gcap)
        shat = si * (1 + gamma)
        vhat = v + (shat ** 2 - si ** 2)
        d1 = shat / (s2 + shat ** 2)
        m1 = 1 + (si1 - shat) * d1
        if si1 == 0:
            M = m1
        else:
            d2 = si1 / (s2 + si1 ** 2) * m1
            M = 1 + (si1 - shat) * (0.5 * d1 + 0.5 * d2)
        v = M ** 2 * vhat
    return v


def predict_churn(s2, N, sigma_min, sigma_max, rho, churn_max=40.0):
    f = lambda sc: gaussian_var_recursion(s2, N, sc, sigma_min, sigma_max, rho) - s2
    lo, hi = 0.0, churn_max
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def run(N=32, sigma_min=0.002, sigma_max=10.0, rho=7, P=20000, n_seeds=3,
        train_steps=6000):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    import torch
    t0 = time.time()
    rng = np.random.default_rng(123)
    ref = sample_target(P, rng)
    sigma_data = float(np.sqrt(np.mean(np.var(ref, axis=0))))     # EDM sigma_data
    s2_eff = float(np.mean(np.var(ref, axis=0)))                  # per-coord data variance
    io.log(f"  target: sigma_data={sigma_data:.4f} s2_eff(per-coord)={s2_eff:.4f}")

    F, loss_history = train_edm(torch, sigma_data, steps=train_steps, seed=0)
    io.log(f"  trained EDM score MLP ({train_steps} steps, {time.time()-t0:.0f}s, "
           f"final loss={loss_history['loss'][-1]:.4f})")

    # sliced-W floor: two INDEPENDENT target samples (sampling-noise baseline of the metric)
    floor_vals = [sliced_w1(sample_target(P, np.random.default_rng(2000 + k)),
                            sample_target(P, np.random.default_rng(3000 + k)), 256, k)
                  for k in range(n_seeds)]
    swd_floor = float(np.mean(floor_vals))
    io.log(f"  SWD sampling floor (real vs real) = {swd_floor:.5f}")

    churn_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 20.0, 30.0]
    swd_mean, swd_std = [], []
    for sc in churn_grid:
        vals = []
        for sd in range(n_seeds):
            gen = edm_sample(F, torch, sigma_data, N, sc, sigma_min, sigma_max, rho, P, seed=100 + sd)
            ref_s = sample_target(P, np.random.default_rng(900 + sd))
            vals.append(sliced_w1(gen, ref_s, n_proj=256, seed=sd))
        swd_mean.append(float(np.mean(vals)))
        swd_std.append(float(np.std(vals)))
        io.log(f"  S_churn={sc:5.1f}: SWD={swd_mean[-1]:.5f} +/- {swd_std[-1]:.5f} ({time.time()-t0:.0f}s)")

    imin = int(np.argmin(swd_mean))
    measured_churn = churn_grid[imin]
    interior = 0 < imin < len(churn_grid) - 1
    pred_churn = predict_churn(s2_eff, N, sigma_min, sigma_max, rho)

    # one scatter at the measured optimum (for the figure)
    gen_opt = edm_sample(F, torch, sigma_data, N, measured_churn, sigma_min, sigma_max, rho, P, seed=7)
    gen_det = edm_sample(F, torch, sigma_data, N, 0.0, sigma_min, sigma_max, rho, P, seed=7)

    res = {
        "config": {"N": N, "sigma_min": sigma_min, "sigma_max": sigma_max, "rho": rho,
                   "P": P, "n_seeds": n_seeds, "train_steps": train_steps,
                   "sigma_data": sigma_data, "s2_eff": s2_eff,
                   "target": "8 Gaussians on ring (radius 2, std 0.2)"},
        "churn_grid": churn_grid, "swd_mean": swd_mean, "swd_std": swd_std,
        "swd_floor": swd_floor, "loss_history": loss_history,
        "measured_opt_S_churn": measured_churn, "interior_optimum": bool(interior),
        "predicted_opt_S_churn_localgauss": pred_churn,
        "swd_at_churn0": swd_mean[0], "swd_at_opt": swd_mean[imin],
        "swd_improvement_factor": float(swd_mean[0] / swd_mean[imin]) if swd_mean[imin] > 0 else None,
        "scatter": {"ref": ref[:4000].tolist(),
                    "gen_opt": gen_opt[:4000].tolist(),
                    "gen_det": gen_det[:4000].tolist()},
    }
    io.save(NAME, res)
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s | interior={interior} "
           f"measured*={measured_churn} predicted*(local-Gauss)="
           f"{'NA' if pred_churn is None else round(pred_churn,2)} "
           f"SWD gain={res['swd_improvement_factor']}")
    return res


if __name__ == "__main__":
    run()
