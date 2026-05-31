"""Figure 1 -- a two-panel hero figure in the standard "intuition + result" form used across the
diffusion-sampling literature (the stochasticity sweet spot of Karras et al. EDM and Xu et al. Restart, where
injected noise contracts accumulated discretisation error). Left: what the churn does to the terminal
variance, drawn as densities -- too little noise under-shoots the target, the right amount matches it, too
much over-shoots. Right: the payoff unique to this paper -- at the matching churn lambda* the KL convergence
order does not merely improve in constant but doubles, from N^-2 to N^-4. No stock art; both panels are drawn
from the canonical numbers (e01_headline)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import io_utils as io
from figstyle import COL

GREEN = "#0f9d6b"
OUT = os.path.join(io.FIG_DIR, "fig_infographic.png")


def main():
    hl = io.load("e01_headline"); s2 = float(hl["config"]["s2"]); lstar = float(hl["lambda_star"])
    Ns = np.array(hl["Ns"], float)
    c0 = np.array(hl["curves"]["0"]["KL"], float); cs = np.array(hl["curves"]["lstar"]["KL"], float)

    plt.rcParams.update({"font.size": 12, "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- left: terminal variance as densities (too little / just right / too much noise) ----
    x = np.linspace(-5.2, 5.2, 600)
    def g(v):
        return np.exp(-x ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    target = g(s2)
    axL.plot(x, target, color="0.25", lw=2.2, ls=(0, (5, 4)), zorder=5, label="target")
    for v, col, lab in [(0.62 * s2, COL["ode"], r"$\lambda{=}0$: under-shoots"),
                        (s2,        GREEN,      r"$\lambda{=}\lambda^\star$: exact"),
                        (1.55 * s2, COL["over"], r"$\lambda$ large: over-shoots")]:
        y = g(v)
        axL.fill_between(x, y, color=col, alpha=0.18, zorder=2)
        line, = axL.plot(x, y, color=col, lw=2.6, zorder=3, label=lab)
        line.set_path_effects([pe.withSimplePatchShadow(offset=(1.5, -1.5), alpha=0.12)])
    axL.set_yticks([]); axL.set_xlabel("sample value")
    axL.set_ylabel("density"); axL.set_xlim(-5.2, 5.2); axL.set_ylim(0, None)
    for sp in ("top", "right", "left"):
        axL.spines[sp].set_visible(False)
    axL.legend(frameon=False, fontsize=10.5, loc="upper right")
    axL.set_title("Churn sets the terminal variance", fontsize=12.5)

    # ---- right: the payoff -- KL convergence order doubles at lambda* ----
    axR.loglog(Ns, c0, "o-", color=COL["ode"], lw=2, ms=5, label=r"$\lambda{=}0$ (deterministic)", zorder=3)
    axR.loglog(Ns, cs, "o-", color=COL["lstar"], lw=2.4, ms=5, label=r"$\lambda{=}\lambda^\star$ (cancellation)", zorder=4)
    axR.loglog(Ns, c0[0] * (Ns / Ns[0]) ** -2.0, ls=":", color="0.55", lw=1.4, zorder=1)
    axR.loglog(Ns, cs[2] * (Ns / Ns[2]) ** -4.0, ls=":", color="0.55", lw=1.4, zorder=1)
    axR.text(Ns[-4], c0[0] * (Ns[-4] / Ns[0]) ** -2.0 * 2.3, r"$N^{-2}$", color="0.4", fontsize=12)
    axR.text(Ns[-5], cs[2] * (Ns[-5] / Ns[2]) ** -4.0 * 0.16, r"$N^{-4}$", color="0.4", fontsize=12)
    axR.set_xlabel("number of steps $N$"); axR.set_ylabel("terminal KL to target")
    axR.legend(frameon=False, fontsize=10.5, loc="lower left")
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    axR.set_title(r"At $\lambda^\star$ the convergence order doubles", fontsize=12.5)

    fig.suptitle("The right amount of injected noise hits the target variance exactly, and converges two orders faster",
                 fontsize=13.5, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT); plt.close()
    io.log("fig_infographic.png (2-panel hero)", "figs.log")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
