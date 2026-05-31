"""Main-text figure: the Gaussian cancellation churn lambda* transfers off-Gaussian. For a single Gaussian
and four non-Gaussian mixtures (all mean-0, total variance s^2=2), sweep the churn and plot the terminal
variance error |Var-s^2| at a fixed step count. Every curve dips at (or just beside) the Gaussian lambda*,
so the same dial that cancels the Gaussian variance bias still minimises it for each non-Gaussian shape.
Exact 1-D density propagation (reuses the E67/E72 propagator); no sampling."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io_utils as io
from figstyle import apply_rc
from e72_mixture_shapes import ve
from lambda_star import lambda_star_vp
from diffusion import set_dps

# shapes: (means, vars, weights), all mean 0 and total variance 2; first is a single Gaussian reference
SHAPES = [
    ("Gaussian",          ([0.0],            [2.0],              [1.0]),        "#d6273b"),
    ("symmetric",         ([-0.7, 0.7],      [1.51, 1.51],       [0.5, 0.5]),   "#1769d6"),
    ("asymmetric",        ([-0.6, 1.4],      [1.16, 1.16],       [0.7, 0.3]),   "#0f9d8f"),
    ("three-component",   ([-1.4, 0.0, 1.4], [1.02, 1.02, 1.02], [0.25, 0.5, 0.25]), "#7b3fb5"),
    ("unequal-variance",  ([-0.5, 0.5],      [0.75, 2.75],       [0.5, 0.5]),   "#e8772e"),
]


def main(B=4.0, T=5.0, N=128):
    set_dps(30)
    apply_rc()
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13, "legend.fontsize": 11})
    lam_star = float(lambda_star_vp(B, 2.0, T))
    lams = np.linspace(0.05, 2.6, 34)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for name, (m, v, w), col in SHAPES:
        errs = [ve(m, v, w, B, T, float(l), N) for l in lams]
        ax.semilogy(lams, errs, "-", color=col, lw=2.2 if name == "Gaussian" else 1.8,
                    label=name, zorder=4 if name == "Gaussian" else 3,
                    alpha=1.0 if name == "Gaussian" else 0.9)
    ax.axvline(lam_star, ls=(0, (3, 3)), color="0.35", lw=1.6, zorder=2)
    ax.text(lam_star + 0.03, ax.get_ylim()[1], r"  Gaussian $\lambda^\star$", va="top", ha="left",
            color="0.3", fontsize=11)
    ax.set_xlabel(r"churn / stochasticity level $\lambda$")
    ax.set_ylabel(rf"terminal variance error $|\mathrm{{Var}}-s^2|$  ($N{{=}}{N}$)")
    ax.set_title(r"The cancellation churn $\lambda^\star$ transfers across non-Gaussian shapes")
    ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=10.5)
    ax.set_xlim(0, 2.6)
    fig.tight_layout()
    out = os.path.join(io.FIG_DIR, "fig_offgaussian.png")
    fig.savefig(out); plt.close()
    io.log("fig_offgaussian.png", "figs.log")
    print("wrote", out)


if __name__ == "__main__":
    main()
