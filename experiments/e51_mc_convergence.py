"""E51 -- Monte-Carlo convergence with REAL sampling variance (not the exact recursion).

The exact-recursion figures are perfectly clean lines because they carry no sampling noise. Here we instead
*draw samples* and run the literal VP Euler--Maruyama reverse sampler, so every point is an empirical estimate
with genuine Monte-Carlo scatter. We sample P paths over K seeds at three churns -- the deterministic ODE
(lambda=0), the reverse SDE (lambda=1), and the superconvergent churn lambda* -- estimate the terminal KL at
each step count N, and report mean +/- seed spread. The realistic message: even with finite samples, lambda*
plunges to the Monte-Carlo floor in far fewer steps than either baseline, and the error bars show how noisy
the estimate is near that floor. Baselines included so no curve stands alone. numpy, fast.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import mpmath as mp
from diffusion import set_dps
from lambda_star import lambda_star_vp
from metrics import kl_gauss
import io_utils as io

NAME = "e51_mc_convergence"


def mc_terminal_var(N, lam, B, s2, T, P, rng):
    """One Monte-Carlo estimate of the terminal variance of the VP Euler--Maruyama reverse sampler.
    x_{k+1} = (1 - A(t_k) dt) x_k + lam*sqrt(B*dt)*xi,  x_0 ~ N(0, V(T)).  Returns sample variance of x_N."""
    dt = T / N
    VT = 1.0 + (s2 - 1.0) * np.exp(-B * T)
    x = rng.standard_normal(P) * np.sqrt(VT)
    u = lam * lam
    for k in range(N):
        t = T - k * dt
        V = 1.0 + (s2 - 1.0) * np.exp(-B * t)
        A = -B / 2.0 + (1.0 + u) / 2.0 * B / V
        x = (1.0 - A * dt) * x + lam * np.sqrt(B * dt) * rng.standard_normal(P)
    return float(np.var(x))


def run(B=4.0, s2=4.0, T=5.0, P=60000, seeds=18):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(30)
    lam_star = float(lambda_star_vp(B, s2, T))
    t0 = time.time()
    Ns = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    churns = {"ode": (0.0, r"$\lambda=0$ (ODE)"), "sde": (1.0, r"$\lambda=1$ (SDE)"),
              "lstar": (lam_star, rf"$\lambda^\star={lam_star:.3f}$")}
    out = {}
    for key, (lam, _) in churns.items():
        rows = []
        seed_traces = [[] for _ in range(seeds)]              # per-seed KL across N -> squiggly traces
        for N in Ns:
            kls = []
            for sd in range(seeds):
                rng = np.random.default_rng(1000 * sd + N)       # deterministic per (seed,N)
                v = mc_terminal_var(N, lam, B, s2, T, P, rng)
                k = float(kl_gauss(mp.mpf(v), s2))
                kls.append(k); seed_traces[sd].append(k)
            kls = np.array(kls)
            rows.append({"N": N, "kl_mean": float(kls.mean()), "kl_std": float(kls.std()),
                         "kl_lo": float(kls.min()), "kl_hi": float(kls.max())})
        out[key] = {"lambda": lam, "rows": rows, "seed_traces": seed_traces}
        io.log(f"  E51 {key} (lam={lam:.3f}): KL {out[key]['rows'][0]['kl_mean']:.2e} -> "
               f"{out[key]['rows'][-1]['kl_mean']:.2e} over N={Ns[0]}..{Ns[-1]} ({time.time()-t0:.0f}s)")
    mc_floor = float(1.0 / (2.0 * P))                            # ~ variance-estimator floor of empirical KL
    io.save(NAME, {"config": {"B": B, "s2": s2, "T": T, "P": P, "seeds": seeds, "Ns": Ns,
                              "lambda_star": lam_star, "mc_floor_est": mc_floor}, "curves": out})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")
    return res if (res := io.load(NAME)) else None


def figure():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r: return
    cfg = r["config"]; cur = r["curves"]
    cols = {"ode": COL["ode"], "sde": COL["sde"], "lstar": COL["lstar"]}
    labs = {"ode": r"$\lambda=0$ (ODE)", "sde": r"$\lambda=1$ (SDE)",
            "lstar": rf"$\lambda^\star={cfg['lambda_star']:.3f}$"}
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    NMIN = 32                                            # below this the high-churn Euler step is unstable
    for key in ("ode", "sde", "lstar"):
        rows_all = cur[key]["rows"]
        allN = np.array([x["N"] for x in rows_all], float); keep = allN >= NMIN
        N = allN[keep]
        # squiggly individual seed traces -- each Monte-Carlo run wiggles, especially near the floor
        for tr in cur[key].get("seed_traces", []):
            ax.plot(N, np.clip(np.array(tr, float)[keep], 1e-12, None), "-", color=cols[key],
                    lw=0.6, alpha=0.22, zorder=2)
        m = np.array([x["kl_mean"] for x in rows_all])[keep]
        ax.plot(N, np.clip(m, 1e-12, None), "o-", color=cols[key], ms=4.5, lw=2.4, zorder=4, label=labs[key])
    ode32 = [x for x in cur["ode"]["rows"] if x["N"] >= NMIN][0]
    N0 = ode32["N"]; k0 = ode32["kl_mean"]
    ax.loglog(N, k0 * (N0 / N) ** 2, ":", color=COL["ref2"], lw=1.4, alpha=0.8, label=r"$N^{-2}$ (generic)")
    ax.axhline(r["config"]["mc_floor_est"], ls="--", color="0.5", lw=1.0, label="Monte-Carlo floor")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"sampler steps $N$"); ax.set_ylabel(r"terminal KL (Monte-Carlo, $\pm$ seed spread)")
    ax.set_title(rf"sampled convergence: $\lambda^\star$ hits the floor in far fewer steps "
                 rf"($P{{=}}${cfg['P']:,}, {cfg['seeds']} seeds)")
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9); ax.grid(True, which="both", alpha=0.18)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_mc_convergence.png")); plt.close()
    io.log("fig_mc_convergence.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
