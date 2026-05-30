"""Figure 1 as a simple, clean infographic of the mechanism. One curve, the discretisation bias C(lambda),
crossing zero at the superconvergence churn lambda*. Left of it the deterministic limit under-shoots the
target variance (C<0); right of it injected noise over-shoots (C>0); at lambda* the two cancel and the
convergence order doubles. Annotated with rounded callouts and a payoff badge. No data clutter; it reads
as a schematic that captures the whole paper."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = io.FIG_DIR


def render():
    cf = io.load("e18_coeff_field"); hl = io.load("e01_headline")
    s2s = np.array(cf["s2s"]); j = int(np.argmin(np.abs(s2s - 2.0)))
    lams = np.array(cf["lams"]); C = np.array(cf["C"][j]); lstar = float(hl["lambda_star"])
    m = lams <= 2.6; lams, C = lams[m], C[m]
    cmin, cmax = float(C.min()), float(C.max())

    plt.rcParams.update({"font.size": 11, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#fbfcfe")
    BLUE, RED, GOLD = "#1769d6", "#e63027", "#f5a800"        # vivid, saturated palette

    # vibrant graded fills (deepen toward the extremes)
    ax.fill_between(lams, C, 0, where=(C < 0), color=BLUE, alpha=0.34, interpolate=True, zorder=1)
    ax.fill_between(lams, C, 0, where=(C > 0), color=RED, alpha=0.34, interpolate=True, zorder=1)
    ax.axhline(0, color="0.3", lw=1.1, zorder=2)
    ax.plot(lams, C, color="#10243a", lw=3.4, zorder=4, solid_capstyle="round")
    ax.scatter([lstar], [0], s=320, marker="*", color=GOLD, edgecolor="#7a5300", lw=1.2, zorder=6)
    ax.plot([lstar, lstar], [cmin * 1.05, 0], color=GOLD, ls=(0, (2, 2)), lw=1.7, zorder=3)

    # rounded callouts
    def callout(x, y, text, fc, ec, tc):
        ax.annotate(text, xy=(x, y), fontsize=10.5, ha="center", va="center", color=tc, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=1.6))
    callout(0.46, cmin * 0.52, "deterministic ODE\nunder-shoots the variance\n($C<0$)",
            "#dcebfb", BLUE, "#0d2f5e")
    callout(2.05, cmax * 0.52, "too much noise\nover-shoots the variance\n($C>0$)",
            "#fde0dd", RED, "#7a140e")
    ax.annotate(rf"$\lambda^\star\approx{lstar:.2f}$" "\n" r"the two cancel",
                xy=(lstar, 0), xytext=(lstar, cmax * 0.66), fontsize=11, ha="center", va="center",
                color="#5a4a10", zorder=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff6da", ec=GOLD, lw=1.4),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.6))
    # payoff badge (placed well below the axis label to avoid overlap)
    ax.annotate(r"at $\lambda^\star$ the convergence order doubles:  $N^{-2}\;\Rightarrow\;N^{-4}$  (a full order faster)",
                xy=(0.5, -0.34), xycoords="axes fraction", ha="center", va="center", fontsize=11.5,
                color="#14401f", zorder=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#e9f6ec", ec="#2e7d4f", lw=1.4))

    ax.set_xlim(lams.min(), lams.max()); ax.set_ylim(cmin * 1.25, cmax * 1.18)
    ax.set_xlabel(r"churn / stochasticity level $\lambda$  ($0$: deterministic ODE, large: noisy SDE)", fontsize=10.5)
    ax.set_ylabel(r"discretisation bias $C(\lambda)$")
    ax.set_yticks([]); ax.set_title("Tuning the churn cancels the discretisation bias", fontsize=13, pad=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.30)
    fig.savefig(os.path.join(FIG, "fig_infographic.png"))
    plt.close()
    io.log("fig_infographic.png", "figs.log")


if __name__ == "__main__":
    render()
    print("fig_infographic done")
