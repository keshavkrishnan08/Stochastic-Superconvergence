"""E63 -- Superconvergence survives a score error: the few-step win on the VP-Euler sampler (the right sampler).

The trained EDM-Heun checks (E36/E61/E62/E39) are the wrong sampler to look for the effect: Heun is already
second order, so there is little O(1/N) discretisation bias for churn to cancel. The paper's mechanism is for
the first-order VP Euler-Maruyama sampler, and E32 already shows a trained VP score superconverging at
lambda* until its score floor. Here we make that win exact and sharp with a CONTROLLED score error: a score
miscalibrated by delta (s_hat = -x/(V+delta)), which the affine recursion handles exactly. For each delta we
compute the terminal KL versus step count N at the deterministic churn (lambda=0) and at the cancellation
churn lambda*. Both meet the same delta^2 floor, but lambda* descends as N^-4 while lambda=0 descends as
N^-2, so lambda* reaches the floor-accuracy in roughly the SQUARE-ROOT of the steps: a quadratic reduction in
sampler steps for the best accuracy a score of that quality allows. That is the practical payoff of the
mechanism under realistic, imperfect scores. Exact extended precision; reuses E31's miscalibrated recursion.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from learned_score import miscalibrated_recursion
from metrics import kl_gauss
from lambda_star import lambda_star_vp
import io_utils as io

NAME = "e63_score_error_win"


def run(dps=40, B=4.0, s2=2.0, T=5.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps); t0 = time.time()
    lam_star = float(lambda_star_vp(B, s2, T))
    Ns = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]
    deltas = [0.0, 1e-4, 1e-3, 1e-2]
    out = {}
    for delta in deltas:
        kl0, kls = [], []
        for N in Ns:
            kl0.append(float(kl_gauss(miscalibrated_recursion(N, T, B, 0.0, s2, delta), s2)))
            kls.append(float(kl_gauss(miscalibrated_recursion(N, T, B, lam_star, s2, delta), s2)))
        # steps each scheme needs to reach 2x the delta-floor (the floor = KL at the largest N for lambda=0)
        floor = max(kl0[-1], 1e-300)
        def steps_to(curve, target):
            import math
            xs = [(math.log(N), math.log(max(k, 1e-300))) for N, k in zip(Ns, curve)]
            for i in range(len(xs) - 1):
                if (xs[i][1] - math.log(target)) * (xs[i + 1][1] - math.log(target)) <= 0:
                    f = (math.log(target) - xs[i][1]) / (xs[i + 1][1] - xs[i][1])
                    return float(math.exp(xs[i][0] + f * (xs[i + 1][0] - xs[i][0])))
            return None
        tgt = 3.0 * floor
        n_det = steps_to(kl0, tgt); n_star = steps_to(kls, tgt)
        out[f"delta={delta}"] = {"delta": delta, "kl0": kl0, "klstar": kls, "floor": floor,
                                 "n_det_to_floor": n_det, "n_star_to_floor": n_star,
                                 "step_saving": (n_det / n_star) if (n_det and n_star) else None}
        sv = out[f"delta={delta}"]["step_saving"]
        io.log(f"  E63 delta={delta:.0e}: floor={floor:.2e}, steps to 3x floor "
               f"det={None if n_det is None else round(n_det)} vs lambda*={None if n_star is None else round(n_star)} "
               f"(x{'NA' if sv is None else round(sv,1)} fewer) ({time.time()-t0:.0f}s)")
    io.save(NAME, {"config": {"dps": dps, "B": B, "s2": s2, "T": T, "lambda_star": lam_star,
                              "Ns": Ns, "deltas": deltas}, "curves": out})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r: return
    Ns = np.array(r["config"]["Ns"], float); cur = r["curves"]
    shown = [d for d in ("delta=0.0001", "delta=0.001", "delta=0.01") if d in cur]
    cols = plt.cm.viridis(np.linspace(0.15, 0.78, len(shown)))
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for key, c in zip(shown, cols):
        d = cur[key]
        ax.loglog(Ns, np.clip(d["kl0"], 1e-300, None), "--", color=c, lw=1.5, alpha=0.9)
        ax.loglog(Ns, np.clip(d["klstar"], 1e-300, None), "o-", color=c, lw=2, ms=3.5,
                  label=rf"$\delta={d['delta']:g}$")
    N0 = Ns[0]; k0 = cur[shown[-1]]["kl0"][0]
    ax.loglog(Ns, k0 * (N0 / Ns) ** 2, ":", color=COL["ref2"], lw=1.2, alpha=0.7, label=r"$N^{-2}$")
    ax.loglog(Ns, k0 * (N0 / Ns) ** 4, ":", color=COL["ref4"], lw=1.2, alpha=0.6, label=r"$N^{-4}$")
    ax.plot([], [], "k--", label=r"deterministic $\lambda{=}0$"); ax.plot([], [], "ko-", ms=3.5, label=r"churn $\lambda^\star$")
    ax.set_xlabel(r"sampler steps $N$"); ax.set_ylabel(r"terminal KL")
    ax.set_title(r"with a score error $\delta$: $\lambda^\star$ ($N^{-4}$, solid) reaches the floor far before $\lambda{=}0$ ($N^{-2}$, dashed)")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9); ax.grid(True, which="both", alpha=0.18)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_score_error_win.png")); plt.close()
    io.log("fig_score_error_win.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
