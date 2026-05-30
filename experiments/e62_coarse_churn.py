"""E62 -- Where churn pays on a trained score: the coarse-step regime.

The earlier trained checks (E61 shootout, E39 MNIST) ran at step counts where the deterministic Heun sampler
had already reached the score/Monte-Carlo floor, so there was no discretisation bias left for stochasticity to
cancel and churn looked inert. That is the wrong regime to test the mechanism -- and the wrong regime in
practice, where few-step sampling is the whole point. Here we train a 2D score WELL (low score floor), then
sweep the EDM churn at COARSE step counts N. There the discretisation bias dominates, and the prediction is
sharp: the optimal-churn sampler beats the deterministic one by a wide margin, the gain grows as N shrinks,
and the measured optimum tracks the Gaussian prediction. This is the practically relevant, strong version of
the trained-score result. Reuses the tested E36 sampler/score/metric. CPU, a few minutes.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io
import e36_edm_toy2d as e36

NAME = "e62_coarse_churn"
SMIN, SMAX, RHO = 0.002, 10.0, 7


def run(P=8000, n_seeds=3, train_steps=6000):
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    t0 = time.time()
    rng = np.random.default_rng(0)
    ref_big = e36.sample_target(P, rng)
    sigma_data = float(np.sqrt(np.mean(np.var(ref_big, axis=0))))
    s2_eff = float(np.mean(np.var(ref_big, axis=0)))
    F, _ = e36.train_edm(torch, sigma_data, steps=train_steps)
    io.log(f"  E62 trained score: sigma_data={sigma_data:.3f} s2_eff={s2_eff:.3f} ({time.time()-t0:.0f}s)")

    Ns = [4, 6, 8, 12, 16, 24, 36, 48]
    rows = []
    for N in Ns:
        # small churn sweep at this N (S_churn up to ~1.2 N so the per-step clamp is reached)
        grid = list(np.round(np.linspace(0.0, max(2.0, 1.2 * N), 9), 2))
        swd_by_sc = []
        for sc in grid:
            vals = [e36.sliced_w1(e36.edm_sample(F, torch, sigma_data, N, float(sc), SMIN, SMAX, RHO, P, seed=7 + sd),
                                  e36.sample_target(P, np.random.default_rng(500 + sd)), 256, sd)
                    for sd in range(n_seeds)]
            swd_by_sc.append(float(np.mean(vals)))
        det = swd_by_sc[0]                                   # S_churn = 0
        imin = int(np.argmin(swd_by_sc)); best_sc = float(grid[imin]); best = swd_by_sc[imin]
        pred = e36.predict_churn(s2_eff, N, SMIN, SMAX, RHO, churn_max=80.0)
        gain = (det / best) if best > 0 else None
        rows.append({"N": N, "swd_det": det, "swd_best": best, "best_churn": best_sc,
                     "predicted_churn": (None if pred is None else float(pred)),
                     "gain_factor": gain, "interior": bool(0 < imin < len(grid) - 1),
                     "grid": grid, "swd_curve": swd_by_sc})
        io.log(f"  E62 N={N:2}: SWD det={det:.4f} best={best:.4f} (x{gain:.2f}) at churn={best_sc:.2f} "
               f"pred={'NA' if pred is None else round(pred,2)} ({time.time()-t0:.0f}s)")
    io.save(NAME, {"config": {"P": P, "n_seeds": n_seeds, "train_steps": train_steps,
                              "sigma_data": sigma_data, "s2_eff": s2_eff, "Ns": Ns,
                              "target": "8-Gaussian ring"}, "rows": rows})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r: return
    rows = r["rows"]; N = np.array([x["N"] for x in rows], float)
    det = np.array([x["swd_det"] for x in rows]); best = np.array([x["swd_best"] for x in rows])
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax.loglog(N, det, "o-", color=COL["ode"], lw=2, ms=5, label=r"deterministic ($\lambda{=}0$)")
    ax.loglog(N, best, "s-", color=COL["lstar"], lw=2, ms=5, label=r"optimal churn")
    ax.fill_between(N, best, det, color=COL["lstar"], alpha=0.10)
    ax.set_xlabel(r"sampler steps $N$"); ax.set_ylabel("sliced-Wasserstein to target")
    ax.set_title("trained 2D score: optimal churn wins in the few-step regime")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)
    gains = det / best
    ax2.semilogx(N, gains, "D-", color=COL["sde"], lw=2, ms=5)
    ax2.axhline(1.0, ls=":", color="0.5"); ax2.set_ylim(0.9, max(1.1, float(gains.max()) * 1.1))
    ax2.set_xlabel(r"sampler steps $N$"); ax2.set_ylabel(r"quality gain (SWD$_\mathrm{det}/$SWD$_\mathrm{churn}$)")
    ax2.set_title("the churn advantage grows as steps shrink")
    ax2.grid(True, which="both", alpha=0.2)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_coarse_churn.png")); plt.close()
    io.log("fig_coarse_churn.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
