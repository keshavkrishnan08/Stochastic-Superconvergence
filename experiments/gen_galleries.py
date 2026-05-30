"""Stitch related standalone appendix figures into compact 'mega galleries' to save space.
Reads existing PNGs and tiles them into a single composite with (a)/(b)/... sub-labels.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io_utils as io

F = io.FIG_DIR


def tile(names, out, ncols, labels, figw=11.0, rowh=3.0):
    n = len(names); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw, rowh * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            p = os.path.join(F, names[i])
            if not os.path.exists(p):
                continue
            ax.imshow(mpimg.imread(p))
            ax.set_title(f"({labels[i]})", loc="left", fontsize=12, fontweight="bold", x=0.0, y=0.98)
    plt.subplots_adjust(wspace=0.02, hspace=0.08, left=0.005, right=0.995, top=0.97, bottom=0.01)
    plt.savefig(os.path.join(F, out), dpi=175, bbox_inches="tight")
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    # Secondary companion panels (app:secondary): coefficient validation, phase plane, mixture boundary
    tile(["fig_coeff_validation.png", "fig_phase.png", "fig_mixture.png"],
         "fig_secondary_gallery.png", ncols=3, labels=["a", "b", "c"], figw=12.0, rowh=3.4)
    # EDM-bridge robustness trio: hyperparameter robustness, data-scale, miscalibration floor
    tile(["fig_edm_robust.png", "fig_data_scale.png", "fig_edm_miscal.png"],
         "fig_edm_robust_gallery.png", ncols=3, labels=["a", "b", "c"], figw=12.0, rowh=3.4)
    # supplementary-law panels: superconvergence band width + initialisation/finite-N sensitivity
    tile(["fig_band.png", "fig_sensitivity.png"],
         "fig_supp_gallery.png", ncols=2, labels=["a", "b"], figw=10.0, rowh=3.6)
    # trained MNIST model: churn-quality sweep + training loss
    tile(["fig_edm_mnist.png", "fig_edm_loss.png"],
         "fig_mnist_gallery.png", ncols=2, labels=["a", "b"], figw=10.5, rowh=3.4)
    io.log("supp + mnist galleries", "figs.log")
