"""Standalone figure generator for E60 (the optimal-churn prediction law).

Kept separate from gen_new_artifacts.py on purpose (that file is edited by a parallel agent);
this avoids cross-edits. Run: `python experiments/gen_churn_law_fig.py`.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io_utils as io

FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def fig_churn_law():
    r = io.load("e60_churn_law")
    if not r:
        print("e60_churn_law not found"); return
    zoo = r["zoo"]; st = r.get("stats", {})
    pts = [(d["name"], d["predicted_opt"], d["measured_opt"]) for d in zoo
           if d["predicted_opt"] is not None and d["interior"]]
    if not pts:
        print("no interior+predicted points"); return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    xp = np.array([p[1] for p in pts]); ym = np.array([p[2] for p in pts])
    hi = max(xp.max(), ym.max()) * 1.12
    axA.plot([0, hi], [0, hi], ls="--", color="0.5", lw=1.2, label="identity (theory $=$ experiment)")
    cmap = plt.cm.viridis(np.linspace(0.08, 0.92, len(pts)))
    for (nm, x, y), c in zip(pts, cmap):
        axA.scatter(x, y, s=75, color=c, edgecolor="k", lw=0.5, zorder=3, label=nm)
    axA.set_xlabel(r"predicted optimal churn (Gaussian local theory, root of $v_N=s^2$)")
    axA.set_ylabel(r"measured optimal churn (trained score, sliced-$W_1$)")
    ttl = "Optimal EDM churn: theory vs trained non-Gaussian samplers"
    if st.get("R2_identity") is not None:
        ttl += rf"  ($R^2_{{\mathrm{{id}}}}={st['R2_identity']:.3f}$)"
    axA.set_title(ttl, fontsize=9.5)
    axA.set_xlim(0, hi); axA.set_ylim(0, hi)
    axA.legend(fontsize=7, loc="upper left", framealpha=0.9)

    nlad = [d for d in r["N_ladder"] if d["predicted_opt"] is not None]
    if nlad:
        Ns = [d["N"] for d in nlad]
        axB.plot(Ns, [d["predicted_opt"] for d in nlad], "o--", color="#4361ee", label="predicted")
        axB.plot(Ns, [d["measured_opt"] for d in nlad], "s-", color="#e76f51",
                 label="measured (trained)")
        axB.set_xlabel(r"sampler step count $N$")
        axB.set_ylabel(r"optimal churn $S_{\mathrm{churn}}$")
        axB.set_title(r"Churn--$N$ scaling, trained target (ring $r{=}2.8$)", fontsize=9.5)
        axB.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG, "fig_churn_law.png")
    plt.savefig(out); plt.close()
    print("wrote", out)
    io.log("fig_churn_law.png", "figs.log")


if __name__ == "__main__":
    fig_churn_law()
