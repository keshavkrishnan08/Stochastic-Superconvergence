"""E61 -- Sampler shootout on a TRAINED non-Gaussian score (baselines + Monte-Carlo bands).

A multi-line, empirical convergence figure (the kind reviewers like): on a single trained EDM
denoiser (8-Gaussian ring, s^2~4) we compare three samplers as a function of the step count N,
each scored by sliced-Wasserstein to held-out data with Monte-Carlo error bands over seeds:

  * ODE (Euler, deterministic)        -- 1st-order probability-flow baseline
  * ODE (Heun, deterministic)         -- 2nd-order probability-flow baseline
  * SDE (Heun + predicted churn)      -- our churn set to the Gaussian local-theory optimum

The stochastic sampler at the predicted churn should sit below both deterministic baselines at low
NFE, demonstrating the practical payoff of the optimal-churn theory on a trained, non-Gaussian model.

CPU-only; reuses E36's tested denoiser/metric/prediction code.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io
import e36_edm_toy2d as e36
import e60_churn_law as e60

NAME = "e61_sampler_shootout"
SMIN, SMAX, RHO = 0.002, 10.0, 7
SQRT2_MINUS_1 = float(np.sqrt(2) - 1)


def sample(F, torch, sigma_data, N, churn, order, P, seed, S_noise=1.0):
    g = torch.Generator().manual_seed(seed)
    sig = e36.edm_sigmas(N, SMIN, SMAX, RHO)
    x = torch.randn(P, 2, generator=g) * float(sig[0])
    gcap = SQRT2_MINUS_1
    with torch.no_grad():
        for i in range(N):
            si = float(sig[i]); si1 = float(sig[i + 1])
            gamma = min(churn / N, gcap) if churn > 0 else 0.0
            shat = si * (1 + gamma)
            if gamma > 0:
                x = x + np.sqrt(max(shat ** 2 - si ** 2, 0.0)) * S_noise * torch.randn(P, 2, generator=g)
            d = (x - e36.denoise(F, x, shat, sigma_data, torch)) / shat
            xn = x + (si1 - shat) * d
            if order == 2 and si1 > 0:
                d2 = (xn - e36.denoise(F, xn, si1, sigma_data, torch)) / si1
                xn = x + (si1 - shat) * (0.5 * d + 0.5 * d2)
            x = xn
    return x.numpy()


def run(P=15000, n_seeds=6, train_steps=7000):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    import torch
    t0 = time.time()
    sampler = e60.ring(2.8)
    rng = np.random.default_rng(7)
    ref = sampler(P, rng)
    sigma_data = float(np.sqrt(np.mean(np.var(ref, axis=0))))
    s2_eff = float(np.mean(np.var(ref, axis=0)))
    F = e60.train_edm(torch, sampler, sigma_data, steps=train_steps)
    io.log(f"  trained ring r=2.8 (s2_eff={s2_eff:.2f}, {time.time()-t0:.0f}s)")

    N_grid = [8, 12, 16, 24, 32, 48, 64]
    methods = {
        "ODE-Euler (det.)": dict(churn=0.0, order=1),
        "ODE-Heun (det.)": dict(churn=0.0, order=2),
        "SDE-Heun (opt. churn)": dict(churn="pred", order=2),
    }
    out = {m: {"N": [], "mean": [], "std": [], "churn": []} for m in methods}
    for N in N_grid:
        pred = e36.predict_churn(s2_eff, N, SMIN, SMAX, RHO, churn_max=60) or 0.0
        for m, cfg in methods.items():
            ch = pred if cfg["churn"] == "pred" else cfg["churn"]
            vals = []
            for sd in range(n_seeds):
                gen = sample(F, torch, sigma_data, N, ch, cfg["order"], P, seed=200 + sd)
                ref_s = sampler(P, np.random.default_rng(500 + sd))
                vals.append(e36.sliced_w1(gen, ref_s, n_proj=512, seed=sd))
            out[m]["N"].append(N); out[m]["mean"].append(float(np.mean(vals)))
            out[m]["std"].append(float(np.std(vals))); out[m]["churn"].append(float(ch))
        io.log(f"  N={N:3}: " + "  ".join(
            f"{m.split()[0]}={out[m]['mean'][-1]:.4f}" for m in methods) +
            f"  (pred churn={pred:.2f}, {time.time()-t0:.0f}s)")

    # SWD real-vs-real floor
    fl = [e36.sliced_w1(sampler(P, np.random.default_rng(11 + k)),
                        sampler(P, np.random.default_rng(99 + k)), 512, k) for k in range(n_seeds)]
    res = {"config": {"target": "8-Gaussian ring r=2.8", "s2_eff": s2_eff, "sigma_data": sigma_data,
                      "P": P, "n_seeds": n_seeds, "N_grid": N_grid, "train_steps": train_steps},
           "methods": out, "swd_floor": float(np.mean(fl))}
    io.save(NAME, res)
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")
    return res


if __name__ == "__main__":
    run()
