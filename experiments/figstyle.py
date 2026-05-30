"""Shared figure style for the whole paper: ONE color per concept, used in every plot.

Import this in every figure generator so the deterministic ODE, the reverse SDE, the superconvergent
churn lambda*, an intermediate churn, and the reference slopes are the SAME color in every figure. Data
variance s^2 is always mapped through the same colormap. This is what makes the paper read as one system."""

# canonical colors -- a concept keeps its color across every figure in the paper
COL = {
    "ode":   "#1769d6",   # lambda=0  deterministic probability-flow ODE   (blue)
    "mid":   "#7b3fb5",   # an intermediate churn 0<lambda<lambda*          (purple)
    "sde":   "#0f9d8f",   # lambda=1  reverse SDE                           (teal)
    "over":  "#e8772e",   # over-churn lambda>lambda*                       (orange)
    "lstar": "#d6273b",   # lambda*   superconvergent / cancellation churn  (crimson)
    "ref2":  "#8a8a8a",   # generic N^-2 reference                          (grey dotted)
    "ref4":  "#111111",   # exact-score N^-4 reference                      (near-black dashed)
    "floor": "#9aa7b2",   # a floor / real-vs-real baseline level           (slate)
}
CMAP_S2 = "viridis"       # data variance s^2 always mapped through viridis

# the standing baseline ladder drawn on convergence plots, in canonical colors
CHURN_BASELINES = [(0.0, "ode", r"$\lambda{=}0$ (ODE)"),
                   (1.0, "sde", r"$\lambda{=}1$ (SDE)"),
                   (2.0, "over", r"$\lambda{=}2$")]


def apply_rc():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.18, "lines.linewidth": 1.9})
