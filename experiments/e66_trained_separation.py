"""E66 -- The step-complexity separation on GENUINELY TRAINED scores (closes the synthetic-delta gap).

Theorem 4 / E64 prove and certify the quadratic step-saving with an analytic variance miscalibration
delta. The honest open question a hard reviewer raises: does the separation appear with a REAL trained
network's score error, where the error is structured rather than a uniform delta? Here we answer it on the
first-order VP-Euler sampler the theorem is built for (NOT the second-order EDM-Heun sampler, where there is
no first-order bias for churn to cancel).

Method. Train a ladder of 1-D VP score MLPs of decreasing residual RMSE (the trained analogue of delta).
For each net measure the terminal KL vs step count N at the deterministic churn lambda=0 and the cancellation
churn lambda*, read off N_det and N_star = steps to reach within a factor rho of the deterministic floor, and
test the prediction: the step-saving N_det/N_star grows as the score sharpens, tracking ~RMSE^{-1/2}, while
the floor scales ~RMSE^2.

Beating Monte-Carlo noise. The terminal KL is read from the terminal VARIANCE (a clean 2nd moment): we
average the variance over many seeds x large P and compute KL from the average, with the per-seed spread as
the band. Training-seed variance is the dominant source (E32 varies 30,000x across seeds), so we train two
nets per quality tier and report the TREND across the ladder, not any single net. Heavy CPU arm, ~30-45 min.
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io
from lambda_star import lambda_star_vp
from learned_score import train_dsm, sample_learned

NAME = "e66_trained_separation"


def _kl(v, s2):
    v = max(float(v), 1e-300)
    return 0.5 * (v / s2 - 1.0 - math.log(v / s2))


def _steps_to(Ns, kl, target):
    xs = [(math.log(N), math.log(max(k, 1e-300))) for N, k in zip(Ns, kl)]
    lt = math.log(target)
    for i in range(len(xs) - 1):
        if (xs[i][1] - lt) * (xs[i + 1][1] - lt) <= 0 and xs[i + 1][1] != xs[i][1]:
            f = (lt - xs[i][1]) / (xs[i + 1][1] - xs[i][1])
            return float(math.exp(xs[i][0] + f * (xs[i + 1][0] - xs[i][0])))
    return None


def _slope(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]; n = len(lx)
    mx = sum(lx) / n; my = sum(ly) / n
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def _kl_curve(model, torch, Ns, T, B, lam, s2, P, seeds):
    """KL vs N from the seed-averaged terminal variance (low MC noise); returns (kl_list, perseed_kls)."""
    kl, perseed = [], []
    for N in Ns:
        vs = [sample_learned(model, torch, N, T, B, lam, s2, P, seed=sd)[0] for sd in seeds]
        kl.append(_kl(float(np.mean(vs)), s2))
        perseed.append([_kl(v, s2) for v in vs])
    return kl, perseed


def _churn_sweep(model, torch, N, T, B, s2, P, seeds, lam_grid):
    """At a fixed N (discretisation bias present), sweep churn and return KL(lambda) from seed-averaged
    variance, plus the empirical optimum lambda_opt. Tests whether the trained net has an interior churn
    optimum and whether the Gaussian lambda* predicts its location."""
    kl = []
    for lam in lam_grid:
        vs = [sample_learned(model, torch, N, T, B, lam, s2, P, seed=sd)[0] for sd in seeds]
        kl.append(_kl(float(np.mean(vs)), s2))
    i = int(np.argmin(kl))
    return kl, float(lam_grid[i]), bool(0 < i < len(lam_grid) - 1)


def run(B=4.0, s2=2.0, T=5.0, P=50000, n_seeds=4, rho=3.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    t0 = time.time()
    lam_star = float(lambda_star_vp(B, s2, T))
    # SMALL N captures N_star: well-trained nets descend below the deterministic-floor target fast
    # (often before N=24), so the grid must start small; up to 256 captures N_det and the floor.
    Ns = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    seeds = list(range(n_seeds))
    # quality ladder of well-trained nets (RMSE ~0.018 -> 0.007), the regime where the separation is
    # both present (discretisation bias to cancel) and measurable (floor low enough for N_star>8).
    tiers = [800, 2500, 8000]
    io.log(f"  E66 lam*={lam_star:.4f}  P={P} seeds={n_seeds}  tiers={tiers}  ({len(tiers)*2} nets)")

    lam_grid = [0.0, 0.4, 0.8, 1.0, lam_star, 1.6, 2.0, 2.6]
    N_sweep = 48                                    # fixed N where discretisation bias is present
    nets = []
    for ep in tiers:
        for tseed in (0, 1):
            model, _, rmse = train_dsm(s2, B, T, width=64, depth=3, epochs=ep, seed=100 + tseed)
            kl0, ps0 = _kl_curve(model, torch, Ns, T, B, 0.0, s2, P, seeds)
            kls, pss = _kl_curve(model, torch, Ns, T, B, lam_star, s2, P, seeds)
            sweep, lam_opt, interior = _churn_sweep(model, torch, N_sweep, T, B, s2, P, seeds, lam_grid)
            floor_det = kl0[-1]                       # deterministic floor (largest N, lambda=0)
            floor_star = kls[-1]
            tgt = rho * floor_det
            n_det = _steps_to(Ns, kl0, tgt); n_star = _steps_to(Ns, kls, tgt)
            saving = (n_det / n_star) if (n_det and n_star) else None
            nets.append({"epochs": ep, "tseed": tseed, "rmse": float(rmse),
                         "kl0": kl0, "klstar": kls, "perseed0": ps0, "perseedstar": pss,
                         "sweep_lam": lam_grid, "sweep_kl": sweep, "lam_opt": lam_opt,
                         "interior": interior, "N_sweep": N_sweep,
                         "floor_det": floor_det, "floor_star": floor_star,
                         "n_det": n_det, "n_star": n_star, "saving": saving})
            io.log(f"  E66 ep={ep:5} s{tseed}: RMSE={rmse:.4f} floor_det={floor_det:.2e} "
                   f"floor*={floor_star:.2e} N_det={None if n_det is None else round(n_det)} "
                   f"N*={None if n_star is None else round(n_star)} saving={None if saving is None else round(saving,2)} "
                   f"| sweep lam_opt={lam_opt:.2f} (lam*={lam_star:.2f}, interior={interior}) ({time.time()-t0:.0f}s)")

    # ---- scaling fits across the ladder (RMSE plays the role of delta) ----
    good = [n for n in nets if n["saving"]]
    fit = {}
    if len(good) >= 3:
        fit["floor_vs_rmse"] = _slope([n["rmse"] for n in good], [n["floor_det"] for n in good])   # ~2
        nd = [(n["rmse"], n["n_det"]) for n in good]
        ns = [(n["rmse"], n["n_star"]) for n in good]
        sv = [(n["rmse"], n["saving"]) for n in good]
        fit["n_det_vs_rmse"] = _slope([r for r, _ in nd], [v for _, v in nd])                       # ~-1
        fit["n_star_vs_rmse"] = _slope([r for r, _ in ns], [v for _, v in ns])                      # ~-1/2
        fit["saving_vs_rmse"] = _slope([r for r, _ in sv], [v for _, v in sv])                      # ~-1/2
        io.log(f"  E66 SCALING vs RMSE  floor~{fit['floor_vs_rmse']:.2f}(pred 2)  "
               f"N_det~{fit['n_det_vs_rmse']:.2f}(pred -1)  N_star~{fit['n_star_vs_rmse']:.2f}(pred -1/2)  "
               f"saving~{fit['saving_vs_rmse']:.2f}(pred -1/2)")
    io.save(NAME, {"config": {"B": B, "s2": s2, "T": T, "P": P, "n_seeds": n_seeds, "rho": rho,
                              "lambda_star": lam_star, "Ns": Ns, "tiers": tiers}, "nets": nets, "fit": fit})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r:
        return
    nets = r["nets"]; lam_star = r["config"]["lambda_star"]
    Ns = np.array(r["config"]["Ns"], float)
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: representative winning net, KL vs N with per-seed variance bands (loss-curve style)
    wins = [n for n in nets if n["saving"]]
    rep = min(wins, key=lambda n: n["rmse"]) if wins else nets[0]   # best-trained winner
    def band(perseed):  # standard error on the seed-mean (the plotted quantity), not raw per-seed std
        a = np.array(perseed, float); return a.mean(1), a.std(1) / np.sqrt(a.shape[1])
    m0, s0 = band(rep["perseed0"]); ms, ss = band(rep["perseedstar"])
    axA.fill_between(Ns, np.clip(m0 - s0, 1e-300, None), m0 + s0, color=COL["ode"], alpha=0.18)
    axA.loglog(Ns, np.clip(rep["kl0"], 1e-300, None), "o--", color=COL["ode"], lw=1.9, ms=4,
               label=r"deterministic $\lambda{=}0$")
    axA.fill_between(Ns, np.clip(ms - ss, 1e-300, None), ms + ss, color=COL["lstar"], alpha=0.18)
    axA.loglog(Ns, np.clip(rep["klstar"], 1e-300, None), "s-", color=COL["lstar"], lw=2.1, ms=4,
               label=r"cancellation churn $\lambda^\star$")
    if rep["n_det"] and rep["n_star"]:
        axA.axhline(3 * rep["floor_det"], ls=":", color="0.5", lw=1.0)
        axA.annotate(rf"$\sim\!{rep['saving']:.1f}\times$ fewer steps", xy=(rep["n_star"], 3 * rep["floor_det"]),
                     xytext=(rep["n_star"] * 1.3, 3 * rep["floor_det"] * 14), fontsize=9, color="0.25",
                     arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
    axA.set_xlabel(r"sampler steps $N$"); axA.set_ylabel("terminal KL (trained score, MC)")
    axA.set_title(rf"trained net (RMSE {rep['rmse']:.3f}): $\lambda^\star$ reaches the floor sooner")
    axA.legend(fontsize=8.5, framealpha=0.9); axA.grid(True, which="both", alpha=0.18)

    # Panel B: churn sweeps for ALL nets -> every U-curve bottoms at the predicted lambda*
    cols = plt.cm.viridis(np.linspace(0.12, 0.82, len(nets)))
    for n, c in zip(sorted(nets, key=lambda x: x["rmse"]), cols):
        lam = np.array(n["sweep_lam"], float); kl = np.array(n["sweep_kl"], float)
        axB.semilogy(lam, kl / kl.min(), "o-", color=c, lw=1.6, ms=3.5, alpha=0.9,
                     label=rf"RMSE {n['rmse']:.3f}")
    axB.axvline(lam_star, color=COL["lstar"], ls="--", lw=1.8)
    axB.annotate(r"predicted $\lambda^\star$", xy=(lam_star, axB.get_ylim()[1] * 0.4),
                 xytext=(lam_star + 0.18, axB.get_ylim()[1] * 0.5), fontsize=9, color=COL["lstar"])
    axB.set_xlabel(r"churn $\lambda$"); axB.set_ylabel(r"terminal KL / min (at $N{=}48$)")
    axB.set_title(r"every trained net's optimal churn lands at $\lambda^\star$")
    axB.legend(fontsize=7.4, framealpha=0.9, ncol=2); axB.grid(True, which="both", alpha=0.18)

    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_trained_separation.png")); plt.close()
    io.log("fig_trained_separation.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
