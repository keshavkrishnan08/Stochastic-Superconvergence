"""Figure 1 -- a simple INFOGRAPHIC of the cancellation event (no plotted curves).

The sampled variance is drawn as a solid disk; the target variance as a dashed ring. Three churn settings:
  lambda=0       disk smaller than target  -> undershoots
  lambda=lambda* disk matches the target   -> exact, order N^-2 -> N^-4
  lambda large   disk larger than target   -> overshoots
A churn dial runs underneath. Icons only -- not a graph.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as pe
import io_utils as io
from figstyle import COL

OUT = os.path.join(io.FIG_DIR, "fig_infographic.png")
GREEN = "#0f9d6b"


def main():
    plt.rcParams.update({"font.size": 12, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    scenes = [
        dict(r=0.62, col=COL["ode"], head=r"$\lambda=0$", note="too little noise", sub="undershoots"),
        dict(r=1.00, col=GREEN,      head=r"$\lambda=\lambda^\star$", note="just right", sub="exact match"),
        dict(r=1.40, col=COL["over"],head=r"$\lambda$ large", note="too much noise", sub="overshoots"),
    ]
    for ax, sc in zip(axes, scenes):
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 2.05); ax.set_aspect("equal"); ax.axis("off")
        if sc["col"] == GREEN:                                                              # soft halo on the winner
            ax.add_patch(Circle((0, 0), sc["r"] + 0.16, fc=GREEN, ec="none", alpha=0.16, zorder=0))
        disk = Circle((0, 0), sc["r"], fc=sc["col"], ec=sc["col"], lw=2, alpha=0.6, zorder=2)  # sample disk
        disk.set_path_effects([pe.withSimplePatchShadow(offset=(2.5, -2.5), shadow_rgbFace="0.45", alpha=0.20)])
        ax.add_patch(disk)
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="0.38", lw=2.0, ls=(0, (5, 4)), zorder=3))  # target ring
        ax.text(0, 1.75, sc["head"], ha="center", fontsize=18, color=sc["col"], fontweight="bold")
        ax.text(0, -1.55, sc["note"], ha="center", fontsize=11, color="0.35")
        ax.text(0, -1.92, sc["sub"], ha="center", fontsize=11.5, color=sc["col"],
                fontweight="bold" if sc["col"] == GREEN else "normal")
        if sc["col"] == GREEN:
            ax.text(0, 0, r"$\checkmark$", ha="center", va="center", fontsize=26, color="white",
                    fontweight="bold")
            ax.text(0, 1.30, r"order $N^{-2}\!\to\!N^{-4}$", ha="center", fontsize=11.5,
                    color=GREEN, fontweight="bold")
    axes[0].text(0, 1.08, "target", ha="center", fontsize=9.5, color="0.4")

    # churn dial
    fig.subplots_adjust(bottom=0.17, top=0.86, wspace=0.05)
    cax = fig.add_axes([0.16, 0.05, 0.68, 0.06]); cax.axis("off")
    cax.set_xlim(0, 1); cax.set_ylim(-1, 1)
    cax.add_patch(FancyArrowPatch((0.02, 0), (0.99, 0), arrowstyle="-|>", mutation_scale=18,
                  color="0.4", lw=2.4))
    for xx, lab, c in [(0.02, r"$0$", "0.4"), (0.5, r"$\lambda^\star$", GREEN), (0.965, r"$\infty$", "0.4")]:
        cax.plot([xx], [0], "o", ms=9 if c == GREEN else 6, color=c, zorder=5)
        cax.text(xx, -0.95, lab, ha="center", fontsize=13, color=c,
                 fontweight="bold" if c == GREEN else "normal")
    cax.text(0.5, 0.9, r"injected noise (churn) $\lambda$", ha="center", fontsize=10.5, color="0.4")

    fig.suptitle("The right amount of injected noise hits the target exactly and converges two orders faster",
                 fontsize=13.5, fontweight="bold", y=1.0)
    plt.savefig(OUT); plt.close()
    io.log("fig_infographic.png (infographic)", "figs.log")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
