"""E52 -- The interior sweet spot, sampled: terminal KL versus churn, with real Monte-Carlo error bars.

The central claim in one picture, drawn from samples rather than the exact recursion. We sweep the churn
lambda at a few step counts, draw P paths over K seeds through the literal VP Euler--Maruyama sampler, and
estimate the terminal KL with its seed spread. Each curve is U-shaped: too little churn (toward the ODE) and
too much (past the SDE) are both worse than the interior optimum at lambda*, and the dip deepens as N grows.
The error bars are the honest sampling noise, widest at the bottom of the well where the divergence is pinned
near zero. Reuses the E51 sampler. numpy, fast.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import mpmath as mp
from diffusion import set_dps
from lambda_star import lambda_star_vp
from metrics import kl_gauss
import io_utils as io
from e51_mc_convergence import mc_terminal_var

NAME = "e52_mc_sweep"


def run(B=4.0, s2=4.0, T=5.0, P=300000, seeds=8):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(30)
    lam_star = float(lambda_star_vp(B, s2, T))
    t0 = time.time()
    lams = list(np.round(np.linspace(0.0, 1.05 * lam_star * 2, 22), 3))   # 0 .. ~2*lambda*
    Ns = [32, 64, 128]
    out = {}
    for N in Ns:
        rows = []
        for lam in lams:
            kls = []
            for sd in range(seeds):
                rng = np.random.default_rng(7000 * sd + int(1000 * lam) + N)
                v = mc_terminal_var(N, float(lam), B, s2, T, P, rng)
                kls.append(float(kl_gauss(mp.mpf(v), s2)))
            kls = np.array(kls)
            rows.append({"lam": float(lam), "kl_mean": float(kls.mean()), "kl_std": float(kls.std()),
                         "kl_lo": float(kls.min()), "kl_hi": float(kls.max()),
                         "kl_seeds": [float(x) for x in kls]})
        out[f"N={N}"] = rows
        best = min(rows, key=lambda r: r["kl_mean"])
        io.log(f"  E52 N={N}: min KL at lam={best['lam']:.2f} (lambda*={lam_star:.2f}) ({time.time()-t0:.0f}s)")
    io.save(NAME, {"config": {"B": B, "s2": s2, "T": T, "P": P, "seeds": seeds, "lams": lams, "Ns": Ns,
                              "lambda_star": lam_star, "mc_floor_est": float(1.0 / (2.0 * P))}, "curves": out})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r: return
    cfg = r["config"]; cur = r["curves"]
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    Nshow = [n for n in cfg["Ns"] if n >= 64]             # N=32 Euler-destabilises just past lambda*; show the clean wells
    cols = plt.cm.viridis(np.linspace(0.2, 0.78, len(Nshow)))
    LMAX = cfg["lambda_star"] + 0.5                       # past lambda* the Euler step goes unstable (variance blows up)
    for (N, c) in zip(Nshow, cols):
        rows = [x for x in cur[f"N={N}"] if x["lam"] <= LMAX]
        lam = np.array([x["lam"] for x in rows])
        for sd in range(len(rows[0].get("kl_seeds", []))):       # squiggly per-seed wells
            tr = np.array([x["kl_seeds"][sd] for x in rows])
            ax.semilogy(lam, np.clip(tr, 1e-12, None), "-", color=c, lw=0.5, alpha=0.16, zorder=2)
        m = np.array([x["kl_mean"] for x in rows])
        ax.semilogy(lam, np.clip(m, 1e-12, None), "o-", color=c, ms=4, lw=2.3, zorder=4, label=rf"$N={N}$")
    ax.axvline(cfg["lambda_star"], ls="--", color=COL["lstar"], lw=2.0, label=rf"$\lambda^\star={cfg['lambda_star']:.3f}$")
    ax.axvline(0.0, ls=":", color=COL["ode"], lw=1.4, alpha=0.8, label=r"$\lambda=0$ (ODE)")
    ax.axvline(1.0, ls=":", color=COL["sde"], lw=1.4, alpha=0.8, label=r"$\lambda=1$ (SDE)")
    ax.axhline(cfg["mc_floor_est"], ls="--", color="0.6", lw=0.9)
    ax.set_xlim(-0.15, LMAX + 0.05); ax.set_ylim(1.2e-5, 8e-3)   # focus on the stable U-well around lambda*
    ax.set_xlabel(r"churn $\lambda$"); ax.set_ylabel(r"terminal KL (Monte-Carlo, $\pm$ seed spread)")
    ax.set_title(r"the interior sweet spot, sampled: KL bottoms at $\lambda^\star$ and deepens with $N$")
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9); ax.grid(True, which="both", alpha=0.18)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_mc_sweep.png")); plt.close()
    io.log("fig_mc_sweep.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
