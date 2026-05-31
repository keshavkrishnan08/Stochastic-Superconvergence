"""Generate the EDM-bridge figures from stored results: the literal-EDM sampler bridge,
schedule-robustness, miscalibration boundary, trained 2D-mixture, MNIST, and training-loss panels.
Each is guarded on result existence; re-run as results land.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io_utils as io

FIG = io.FIG_DIR
plt.rcParams.update({"font.size": 11, "figure.dpi": 170, "savefig.dpi": 170,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "savefig.bbox": "tight", "figure.facecolor": "white"})


def fig_edm_bridge():
    r = io.load("e35_edm_bridge")
    if not r:
        return
    cases = r["cases"]
    order = [k for k in ("s2=1.0", "s2=4.0", "s2=16.0") if k in cases]
    # data variance s^2 -> viridis, consistent with every other s^2-coloured figure in the paper
    cols = {k: plt.cm.viridis(v) for k, v in zip(("s2=1.0", "s2=4.0", "s2=16.0"), (0.15, 0.5, 0.82))}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel A: terminal KL vs S_churn (interior optimum), predicted root marked
    for k in order:
        c = cases[k]
        s2 = c["s2"]
        cg = np.array(c["churn_grid"], float)
        kl = np.array(c["KL_curve_Nsearch"], float)
        kl = np.clip(kl, 1e-300, None)
        col = cols[k]
        ax.semilogy(cg, kl, "-", color=col, lw=1.7, label=rf"$s^2={s2:g}$")
        if c["predicted_S_churn"] is not None:
            ax.axvline(c["predicted_S_churn"], ls=":", color=col, alpha=0.8, lw=1.3)
    ax.set_xlabel(r"EDM churn parameter $S_{\mathrm{churn}}$")
    ax.set_ylabel(r"terminal KL$(\hat p_N\,\|\,p)$")
    ax.set_title(r"literal EDM sampler: an interior churn optimum")
    ax.set_xlim(0, 20)
    ax.legend(fontsize=9, title=r"dotted: closed-form $S_{\mathrm{churn}}^\star$", title_fontsize=8)

    # Panel B: optimal churn vs N (renormalised law) -- root of v_N=s2 at each N
    for k in order:
        c = cases[k]
        s2 = c["s2"]
        Ns = np.array([l["N"] for l in c["N_ladder"] if l["opt_S_churn"] is not None], float)
        oc = np.array([l["opt_S_churn"] for l in c["N_ladder"] if l["opt_S_churn"] is not None], float)
        col = cols[k]
        ax2.plot(Ns, oc, "o-", color=col, ms=5, lw=1.6, label=rf"$s^2={s2:g}$")
    # sqrt(N) guide anchored on the s2=4 mid point
    cm = cases.get("s2=4.0")
    if cm is not None:
        good = [l for l in cm["N_ladder"] if l["opt_S_churn"] is not None]
        if good:
            anchor = good[len(good) // 2]
            Ns = np.array([l["N"] for l in good], float)
            g = anchor["opt_S_churn"] * np.sqrt(Ns / anchor["N"])
            ax2.plot(Ns, g, "--k", alpha=0.5, lw=1.0, label=r"$\propto\sqrt{N}$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"sampler steps $N$")
    ax2.set_ylabel(r"optimal $S_{\mathrm{churn}}$ (root of $v_N=s^2$)")
    ax2.set_title(r"optimal churn grows with $N$ (renormalised law)")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_edm_bridge.png"))
    plt.close()
    io.log("fig_edm_bridge.png", "figs.log")


def fig_edm_robust():
    """E47: the closed-form churn matches the grid-search optimum across the whole EDM schedule grid.
    Scatter of predicted S_churn (root of v_N=s^2) vs grid-search argmin, one point per (rho, sigma_max);
    every point sits on the diagonal, within one grid step. Convincing one-glance robustness figure."""
    r = io.load("e47_edm_hyper_robust")
    if not r:
        return
    rows = [x for x in r["rows"] if x.get("predicted_S_churn") is not None]
    if not rows:
        return
    smax_vals = sorted({x["sigma_max"] for x in rows})
    rho_vals = sorted({x["rho"] for x in rows})
    cmap = {s: plt.cm.viridis(v) for s, v in zip(smax_vals, np.linspace(0.15, 0.82, len(smax_vals)))}
    marks = {rho: m for rho, m in zip(rho_vals, ["o", "s", "^", "D", "v", "P"])}
    gstep = r["config"]["grid_step"]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lo = min(min(x["predicted_S_churn"] for x in rows), min(x["grid_opt_S_churn"] for x in rows)) - 0.4
    hi = max(max(x["predicted_S_churn"] for x in rows), max(x["grid_opt_S_churn"] for x in rows)) + 0.4
    ax.plot([lo, hi], [lo, hi], "-", color="0.4", lw=1.4, zorder=1, label="perfect agreement")
    ax.fill_between([lo, hi], [lo - gstep, hi - gstep], [lo + gstep, hi + gstep],
                    color="0.7", alpha=0.18, zorder=0, label="within one grid step")
    for x in rows:
        ax.scatter(x["predicted_S_churn"], x["grid_opt_S_churn"], s=90, color=cmap[x["sigma_max"]],
                   marker=marks[x["rho"]], edgecolor="k", linewidth=0.5, zorder=3)
    # legends: colour = sigma_max, marker = rho
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], marker="o", ls="", mfc=cmap[s], mec="k", ms=9, label=rf"$\sigma_{{\max}}={s:g}$") for s in smax_vals]
    h2 = [Line2D([], [], marker=marks[rh], ls="", mfc="0.8", mec="k", ms=9, label=rf"$\rho={rh}$") for rh in rho_vals]
    leg1 = ax.legend(handles=h1, fontsize=8.5, loc="upper left", title="data scale")
    ax.add_artist(leg1)
    ax.legend(handles=h2, fontsize=8.5, loc="lower right", title="schedule exponent")
    n_ok = sum(1 for x in rows if x.get("within_one_grid_step"))
    ax.set_xlabel(r"closed-form churn $S_{\mathrm{churn}}^\star$ (root of $v_N=s^2$)")
    ax.set_ylabel(r"grid-search optimum $S_{\mathrm{churn}}$")
    ax.set_title(rf"EDM bridge is schedule-robust: {n_ok}/{len(rows)} configs on the diagonal")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_edm_robust.png"))
    plt.close()
    io.log("fig_edm_robust.png", "figs.log")


def fig_edm_miscal():
    """E49: a scalar score miscalibration kills the cancellation. Terminal KL vs EDM churn for the denoiser
    scaled by (1+delta). Only delta=0 (exact) carves the deep interior dip; any delta>0 over-shoots and any
    delta<0 under-shoots at every churn, so the curve is monotone off a floor that grows with |delta|."""
    r = io.load("e49_edm_miscal")
    if not r:
        return
    cg = np.array(r["config"]["churn_grid"], float)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for x in sorted(r["rows"], key=lambda z: z["delta"]):
        d = x["delta"]; kl = np.clip(np.array(x["KL_curve"], float), 1e-16, None)
        if d == 0:
            col, lw, z, lab = "k", 2.6, 6, r"$\delta=0$ (exact score)"
        elif d < 0:
            col = plt.cm.Blues(0.45 + 0.45 * min(abs(np.log10(-d)) / 3.0, 1.0)); lw, z = 1.7, 3
            lab = rf"$\delta={d:+.0e}$"
        else:
            col = plt.cm.Reds(0.45 + 0.45 * min(abs(np.log10(d)) / 3.0, 1.0)); lw, z = 1.7, 3
            lab = rf"$\delta={d:+.0e}$"
        ax.semilogy(cg, kl, color=col, lw=lw, zorder=z, label=lab)
    ax.set_xlim(0, 12)
    ax.set_xlabel(r"EDM churn $S_{\mathrm{churn}}$"); ax.set_ylabel(r"terminal KL")
    ax.set_title(r"only an exact score cancels: any miscalibration floors the dip")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9); ax.grid(True, which="both", alpha=0.18)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_edm_miscal.png"))
    plt.close()
    io.log("fig_edm_miscal.png", "figs.log")


def fig_edm_toy2d():
    r = io.load("e36_edm_toy2d")
    if not r:
        return
    cg = np.array(r["churn_grid"], float)
    mean = np.array(r["swd_mean"], float)
    std = np.array(r["swd_std"], float)
    floor = r["swd_floor"]
    meas = r["measured_opt_S_churn"]
    pred = r["predicted_opt_S_churn_localgauss"]
    sc = r["scatter"]
    ref = np.array(sc["ref"]); gen = np.array(sc["gen_opt"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax.errorbar(cg, mean, yerr=std, fmt="o-", color="#2a9d8f", ms=4, lw=1.6, capsize=2,
                label="sliced-$W_1$ to held-out data")
    ax.axhline(floor, ls="--", color="0.5", lw=1.0, label="real-vs-real floor")
    ax.axvline(meas, ls="-", color="#e76f51", alpha=0.8, lw=1.3,
               label=rf"measured optimum $={meas:g}$")
    if pred is not None:
        ax.axvline(pred, ls=":", color="#4361ee", alpha=0.9, lw=1.5,
                   label=rf"local-Gaussian prediction $={pred:.2f}$")
    ax.set_xlabel(r"EDM churn parameter $S_{\mathrm{churn}}$")
    ax.set_ylabel(r"sliced-Wasserstein to target")
    ax.set_title(r"trained non-Gaussian score: interior churn optimum")
    ax.set_xlim(-0.5, max(cg))
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)

    ax2.scatter(ref[:, 0], ref[:, 1], s=4, c="0.7", alpha=0.5, label="target (8 Gaussians)")
    ax2.scatter(gen[:, 0], gen[:, 1], s=4, c="#e76f51", alpha=0.5,
                label=rf"EDM sample at $S_{{\mathrm{{churn}}}}={meas:g}$")
    ax2.set_aspect("equal"); ax2.set_title("samples at the optimal churn")
    ax2.set_xlabel(r"$x_1$"); ax2.set_ylabel(r"$x_2$")
    ax2.legend(fontsize=8, loc="upper right", markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_edm_toy2d.png"))
    plt.close()
    io.log("fig_edm_toy2d.png", "figs.log")


def fig_edm_mnist():
    r = io.load("e39_edm_mnist")
    if not r:
        return
    cg = np.array(r["churn_grid"], float)
    mean = np.array(r["swd_mean"], float)
    std = np.array(r["swd_std"], float)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.errorbar(cg, mean, yerr=std, fmt="o-", color="#2a9d8f", ms=4, lw=1.6, capsize=2,
                label="sliced-$W_1$ to held-out MNIST")
    ax.axhline(r["swd_floor"], ls="--", color="0.5", lw=1.0, label="real-vs-real floor")
    ax.axvline(r["measured_opt_S_churn"], ls="-", color="#e76f51", alpha=0.8, lw=1.3,
               label=rf"measured optimum $={r['measured_opt_S_churn']:g}$")
    ax.set_xlabel(r"EDM churn parameter $S_{\mathrm{churn}}$")
    ax.set_ylabel(r"sliced-Wasserstein to MNIST test set")
    ax.set_title(r"MNIST EDM U-Net: churn vs sample quality")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_edm_mnist.png"))
    plt.close()
    io.log("fig_edm_mnist.png", "figs.log")


def _parse_loss_log(path):
    """Extract (step, loss) pairs from a training log with lines 'step K/.. loss=X'."""
    import re
    if not os.path.exists(path):
        return None
    steps, vals = [], []
    pat = re.compile(r"step\s+(\d+)/\d+\s+loss=([0-9.eE+-]+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1))); vals.append(float(m.group(2)))
    return {"step": steps, "loss": vals} if steps else None


def fig_edm_loss():
    """Training-loss curves for the two trained-score experiments (E36 MLP, E39 MNIST U-Net)."""
    e36 = io.load("e36_edm_toy2d") or {}
    lh36 = e36.get("loss_history")
    d39 = io.load("e39_edm_mnist") or {}
    lh39 = d39.get("loss_history") or _parse_loss_log(
        os.path.join(os.path.dirname(__file__), "..", "logs", "e39_run.out"))

    panels = [p for p in [("E36: 2D 8-Gaussian MLP denoiser", lh36),
                          ("E39: 14$\\times$14 MNIST U-Net", lh39)] if p[1] and p[1]["step"]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4.0), squeeze=False)
    for ax, (title, lh) in zip(axes[0], panels):
        ax.plot(lh["step"], lh["loss"], color="#264653", lw=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("training step"); ax.set_ylabel(r"EDM denoising-score-matching loss")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_edm_loss.png"))
    plt.close()
    io.log("fig_edm_loss.png", "figs.log")


if __name__ == "__main__":
    fig_edm_bridge()
    fig_edm_robust()
    fig_edm_miscal()
    fig_edm_toy2d()
    fig_edm_mnist()
    fig_edm_loss()
    print("gen_new_artifacts done")
