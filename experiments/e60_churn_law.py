"""E60 -- The optimal-churn prediction law on TRAINED, NON-Gaussian scores (main-text candidate).

E36/E39 show the mechanism transfers on single trained non-Gaussian targets. E60 turns that into a
quantitative law: across a zoo of structurally diverse 2D targets (rings of several radii, two
moons, a pinwheel, grids of Gaussians) -- each with its own small EDM-trained MLP denoiser -- we
measure the empirically optimal EDM churn (literal Algorithm 2, sliced-Wasserstein objective) and
compare it to the parameter-free Gaussian local-theory prediction (root of v_N=s^2 for a Gaussian of
the data's per-coordinate variance). If the points fall on the identity line across shapes and
scales, the Gaussian theory predicts the optimal stochasticity of a trained non-Gaussian sampler --
a far stronger statement than a single consistency check.

Two panels:
  (A) measured vs predicted optimal churn across the target zoo (identity line + R^2);
  (B) optimal churn vs step count N on one target, measured vs predicted curve.

Robustness: 5 seeds, 512 projections, and a parabola-fit sub-grid optimum (denoises the shallow
minimum). CPU-only; reuses the tested E36 sampler/metric/prediction code.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io
import e36_edm_toy2d as e36

NAME = "e60_churn_law"
SMIN, SMAX, RHO = 0.002, 10.0, 7


# ------------------------------- target zoo -------------------------------
def ring(r, std=0.2):
    def f(n, rng):
        ang = (2 * np.pi / 8) * rng.integers(0, 8, size=n)
        c = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        return (c + std * rng.standard_normal((n, 2))).astype(np.float64)
    return f


def two_moons(scale=3.0, noise=0.1):
    def f(n, rng):
        t = rng.uniform(0, np.pi, n)
        half = n // 2
        x = np.empty((n, 2))
        x[:half, 0] = np.cos(t[:half]); x[:half, 1] = np.sin(t[:half])
        x[half:, 0] = 1 - np.cos(t[half:]); x[half:, 1] = 0.5 - np.sin(t[half:])
        x += noise * rng.standard_normal((n, 2))
        x -= x.mean(0)
        return (scale * x).astype(np.float64)
    return f


def pinwheel(scale=2.5, n_arms=5, std=0.25):
    def f(n, rng):
        arm = rng.integers(0, n_arms, n)
        rad = rng.uniform(0.3, 1.0, n)
        base = arm * 2 * np.pi / n_arms + rad * 2.2
        x = np.stack([rad * np.cos(base), rad * np.sin(base)], 1)
        x += std * rad[:, None] * rng.standard_normal((n, 2))
        return (scale * x).astype(np.float64)
    return f


def grid_gauss(k=3, spacing=2.2, std=0.18):
    cs = np.array([[i, j] for i in range(k) for j in range(k)], float)
    cs = (cs - cs.mean(0)) * spacing
    def f(n, rng):
        idx = rng.integers(0, len(cs), n)
        return (cs[idx] + std * rng.standard_normal((n, 2))).astype(np.float64)
    return f


ZOO = [
    ("ring r=1.4", ring(1.4)), ("ring r=2.0", ring(2.0)),
    ("ring r=2.8", ring(2.8)), ("ring r=4.0", ring(4.0)),
    ("two moons", two_moons(3.2)), ("pinwheel", pinwheel(2.6)),
    ("grid 3x3", grid_gauss(3, 2.2)), ("grid 2x2", grid_gauss(2, 3.4)),
]


def train_edm(torch, sampler, sigma_data, steps=6000, batch=4096, lr=2e-3, seed=0,
              P_mean=-1.2, P_std=1.2, width=128, depth=4):
    torch.set_num_threads(4); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F = e36._build(torch)(width, depth)
    opt = torch.optim.Adam(F.parameters(), lr=lr)
    sd = float(sigma_data)
    for it in range(steps):
        x0 = torch.from_numpy(sampler(batch, rng)).float()
        sigma = torch.exp(P_mean + P_std * torch.randn(batch))
        xt = x0 + torch.randn(batch, 2) * sigma[:, None]
        D = e36.denoise(F, xt, sigma, sd, torch)
        w = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
        loss = (w[:, None] * (D - x0) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(F.parameters(), 10.0); opt.step()
    return F


def parabola_opt(grid, ys):
    """Denoised sub-grid optimum: parabola fit on the window around the discrete argmin."""
    g = np.asarray(grid, float); y = np.asarray(ys, float)
    i = int(np.argmin(y))
    lo, hi = max(0, i - 2), min(len(g), i + 3)
    xs, ws = g[lo:hi], y[lo:hi]
    interior = 0 < i < len(g) - 1
    if len(xs) >= 3:
        a, b, _ = np.polyfit(xs, ws, 2)
        if a > 0:
            v = -b / (2 * a)
            if xs.min() <= v <= xs.max():
                return float(v), float(g[i]), interior
    return float(g[i]), float(g[i]), interior


def sweep_target(torch, sampler, N, P=12000, n_seeds=5, churn_grid=None, train_steps=6000):
    rng = np.random.default_rng(1234)
    ref = sampler(P, rng)
    sigma_data = float(np.sqrt(np.mean(np.var(ref, axis=0))))
    s2_eff = float(np.mean(np.var(ref, axis=0)))
    F = train_edm(torch, sampler, sigma_data, steps=train_steps)
    if churn_grid is None:
        churn_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    swd = []
    for sc in churn_grid:
        vals = []
        for sd in range(n_seeds):
            gen = e36.edm_sample(F, torch, sigma_data, N, sc, SMIN, SMAX, RHO, P, seed=100 + sd)
            ref_s = sampler(P, np.random.default_rng(900 + sd))
            vals.append(e36.sliced_w1(gen, ref_s, n_proj=512, seed=sd))
        swd.append(float(np.mean(vals)))
    meas, grid_opt, interior = parabola_opt(churn_grid, swd)
    pred = e36.predict_churn(s2_eff, N, SMIN, SMAX, RHO, churn_max=60)
    return {"sigma_data": sigma_data, "s2_eff": s2_eff, "churn_grid": churn_grid,
            "swd_mean": swd, "measured_opt": meas, "grid_opt": grid_opt,
            "interior": interior, "predicted_opt": pred}


def run(N=64, train_steps=6000):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    import torch
    t0 = time.time()

    # ---- Panel A: the zoo at fixed N ----
    zoo_rows = []
    for name, sampler in ZOO:
        r = sweep_target(torch, sampler, N, train_steps=train_steps)
        r["name"] = name
        zoo_rows.append(r)
        io.log(f"  [{name:10}] s2_eff={r['s2_eff']:.2f} measured*={r['measured_opt']:.2f} "
               f"predicted*={'NA' if r['predicted_opt'] is None else round(r['predicted_opt'],2)} "
               f"interior={r['interior']} ({time.time()-t0:.0f}s)")

    # ---- Panel B: N-scaling on one mid-scale target (ring r=2.8) ----
    nlad = []
    sampler = ring(2.8)
    for Nn in [32, 64, 128]:
        r = sweep_target(torch, sampler, Nn, train_steps=train_steps)
        r["N"] = Nn
        nlad.append({"N": Nn, "measured_opt": r["measured_opt"], "predicted_opt": r["predicted_opt"],
                     "s2_eff": r["s2_eff"], "interior": r["interior"]})
        io.log(f"  [N-ladder N={Nn:3}] measured*={r['measured_opt']:.2f} "
               f"predicted*={'NA' if r['predicted_opt'] is None else round(r['predicted_opt'],2)} "
               f"({time.time()-t0:.0f}s)")

    # ---- correlation stats for Panel A (interior, prediction-exists points) ----
    pts = [(d["predicted_opt"], d["measured_opt"]) for d in zoo_rows
           if d["predicted_opt"] is not None and d["interior"]]
    stats = {}
    if len(pts) >= 3:
        xp = np.array([p[0] for p in pts]); ym = np.array([p[1] for p in pts])
        slope, intc = np.polyfit(xp, ym, 1)
        ss_res = np.sum((ym - (slope * xp + intc)) ** 2)
        ss_tot = np.sum((ym - ym.mean()) ** 2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else None
        ss_id = np.sum((ym - xp) ** 2)
        r2_identity = 1 - ss_id / ss_tot if ss_tot > 0 else None
        stats = {"n_points": len(pts), "slope": float(slope), "intercept": float(intc),
                 "R2_bestfit": float(r2_fit), "R2_identity": float(r2_identity),
                 "median_abs_gap": float(np.median(np.abs(ym - xp)))}

    res = {"config": {"N": N, "sigma_min": SMIN, "sigma_max": SMAX, "rho": RHO,
                      "train_steps": train_steps, "n_targets": len(ZOO)},
           "zoo": zoo_rows, "N_ladder": nlad, "stats": stats}
    io.save(NAME, res)
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s | stats={stats}")
    return res


if __name__ == "__main__":
    run()
