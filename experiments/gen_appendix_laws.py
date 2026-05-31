"""Three appendix laws drawn as graphs (replacing the corresponding numeric tables): the cancellation-churn
scaling lambda*/sqrt(s^2) -> kappa, the score-error floor KL ∝ delta^2, and the trained-network floor growing
with the residual score RMSE. All read from results/*.json; no manual data."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io_utils as io
from figstyle import COL, apply_rc


def main():
    apply_rc()
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 12})
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # (a) controlled score-error floor ∝ delta^2
    pa = io.load("e11_learned")["part_A"]
    ds, fl = [], []
    for _, v in pa.items():
        if v["delta"] > 0:
            ds.append(v["delta"]); fl.append(v["floor"])
    o = np.argsort(ds); ds = np.array(ds)[o]; fl = np.array(fl)[o]
    ax[0].loglog(ds, fl, "o", color=COL["sde"], ms=8, zorder=3)
    ax[0].loglog(ds, fl[0] * (ds / ds[0]) ** 2, "--", color="0.4", lw=1.5, label=r"$\propto\delta^2$")
    ax[0].set_xlabel(r"score error $\delta$"); ax[0].set_ylabel(r"terminal KL floor")
    ax[0].set_title(r"(a) controlled-miscalibration floor"); ax[0].legend(frameon=False, loc="upper left")

    # (b) trained-network floor vs residual score RMSE
    pb = io.load("e11_learned")["part_B"]
    rmse = np.array([r["residual_rmse"] for r in pb]); floor = np.array([r["floor_KL"] for r in pb])
    ax[1].loglog(rmse, floor, "o", color=COL["over"], ms=8, zorder=3, alpha=0.85)
    g = np.argsort(rmse)
    ax[1].loglog(rmse[g], floor[g][0] * (rmse[g] / rmse[g][0]) ** 2, "--", color="0.4", lw=1.3,
                 label=r"$\propto\mathrm{RMSE}^2$ guide")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax[1].set_xlim(0.0078, 0.0178); ax[1].set_xticks([0.008, 0.010, 0.012, 0.014, 0.016])
    ax[1].xaxis.set_major_formatter(ScalarFormatter()); ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_xlabel(r"residual score RMSE"); ax[1].set_ylabel(r"terminal KL floor")
    ax[1].set_title(r"(b) trained-network floor"); ax[1].legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out = os.path.join(io.FIG_DIR, "fig_floor_laws.png")
    fig.savefig(out); plt.close()
    io.log("fig_floor_laws.png", "figs.log")
    print("wrote", out)


if __name__ == "__main__":
    main()
