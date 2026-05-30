"""Standalone figure generator for E61 (sampler shootout). Separate from gen_new_artifacts.py
(edited by a parallel agent) to avoid cross-edits. Run: `python experiments/gen_shootout_fig.py`.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io_utils as io

FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
STYLE = {
    "ODE-Euler (det.)": dict(color="#8d99ae", marker="^", ls="--"),
    "ODE-Heun (det.)": dict(color="#457b9d", marker="s", ls="-."),
    "SDE-Heun (opt. churn)": dict(color="#e63946", marker="o", ls="-"),
}


def fig_shootout():
    r = io.load("e61_sampler_shootout")
    if not r:
        print("e61 not found"); return
    methods = r["methods"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, st in STYLE.items():
        m = methods[name]
        N = np.array(m["N"], float); mean = np.array(m["mean"], float); sd = np.array(m["std"], float)
        ax.fill_between(N, mean - sd, mean + sd, color=st["color"], alpha=0.18, lw=0)
        ax.plot(N, mean, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8,
                ms=5, label=name)
    if r.get("swd_floor"):
        ax.axhline(r["swd_floor"], color="0.55", ls=":", lw=1.1, label="real-vs-real floor")
    import matplotlib.ticker as mt
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xticks(r["config"]["N_grid"])
    ax.xaxis.set_major_formatter(mt.ScalarFormatter())
    ax.set_xticklabels(r["config"]["N_grid"])
    ax.set_xlabel(r"sampler steps $N$ (function evaluations)")
    ax.set_ylabel(r"sliced-Wasserstein to target")
    s2 = r["config"]["s2_eff"]
    ax.set_title(rf"Sampler shootout on a trained non-Gaussian score (ring, $s^2{{\approx}}{s2:.1f}$)",
                 fontsize=10.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(FIG, "fig_shootout.png")
    plt.savefig(out); plt.close()
    print("wrote", out)
    io.log("fig_shootout.png", "figs.log")


if __name__ == "__main__":
    fig_shootout()
