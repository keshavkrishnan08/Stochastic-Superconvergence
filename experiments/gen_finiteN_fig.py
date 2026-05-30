"""Finite-N optimal churn approaching lambda* (replaces the old E12 sensitivity table).
At finite step count the KL-minimising churn sits just below lambda* and climbs to it as N grows;
plotted as a convergence curve it reads far better than a row of numbers."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusion import set_dps
from learned_score import miscalibrated_recursion
from metrics import kl_gauss
from lambda_star import lambda_star_vp
import io_utils as io
from figstyle import COL

set_dps(25)
B, s2, T = 4.0, 2.0, 5.0
lstar = float(lambda_star_vp(B, s2, T))
Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192]


def lambda_opt(N):
    """KL-minimising churn at step count N: grid then parabolic refine for a sub-grid value."""
    g = np.linspace(1.05, 1.30, 140)
    kl = [float(kl_gauss(miscalibrated_recursion(N, T, B, float(l), s2, 0.0), s2)) for l in g]
    i = int(np.argmin(kl))
    if 0 < i < len(g) - 1:                       # parabolic vertex of the 3 points around the min
        y0, y1, y2 = kl[i - 1], kl[i], kl[i + 1]
        denom = (y0 - 2 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denom if denom else 0.0
        return float(g[i] + shift * (g[1] - g[0]))
    return float(g[i])


lopt = [lambda_opt(N) for N in Ns]
plt.rcParams.update({"font.size": 11, "savefig.dpi": 185, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhline(lstar, ls="--", color=COL["lstar"], lw=1.6, label=rf"$\lambda^\star={lstar:.4f}$ (asymptote)")
ax.semilogx(Ns, lopt, "o-", color=COL["sde"], ms=4.5, lw=1.7, label=r"finite-$N$ optimum $\lambda_{\mathrm{opt}}(N)$")
ax.set_xlabel("steps $N$")
ax.set_ylabel(r"KL-optimal churn")
ax.set_title(r"the finite-$N$ optimal churn climbs to $\lambda^\star$")
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, which="both", alpha=0.16)
plt.savefig(os.path.join(io.FIG_DIR, "fig_finiteN_churn.png"))
plt.close()
print(f"lambda* = {lstar:.4f}")
print("N, lambda_opt:", [(N, round(l, 4)) for N, l in zip(Ns, lopt)])
io.log("fig_finiteN_churn.png", "figs.log")
