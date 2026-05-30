"""Generate all paper figures from results/*.json. Each figure is guarded on result existence,
so this runs incrementally as experiments complete. Outputs to figures/."""
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


def fig_headline():
    r = io.load("e01_headline")
    if not r: return
    from figstyle import COL
    Ns = r["Ns"]; C = r["curves"]
    plt.figure(figsize=(7, 4.6))
    # canonical churn colors, shared across every figure in the paper
    styles = {"0": (COL["ode"], r"$\lambda=0$ (ODE)"), "1": (COL["sde"], r"$\lambda=1$ (SDE)"),
              "2": (COL["over"], r"$\lambda=2$"), "lstar": (COL["lstar"], rf"$\lambda^\star={r['lambda_star']:.3f}$")}
    for tag, (col, lab) in styles.items():
        c = C[tag]
        plt.loglog(Ns, c["KL"], 'o-', color=col, ms=3.5, label=lab)
        if tag != "lstar":
            plt.loglog(Ns, c["theory_N2"], '--', color=col, alpha=0.5)
    kl0 = C["lstar"]["KL"][0]
    plt.loglog(Ns, [kl0 * (Ns[0] / N) ** 2 for N in Ns], ':', color=COL["ref2"], alpha=0.8, label=r"$N^{-2}$ ref")
    plt.loglog(Ns, [kl0 * (Ns[0] / N) ** 4 for N in Ns], '--', color=COL["ref4"], alpha=0.7, label=r"$N^{-4}$ ref")
    plt.xlabel("steps $N$"); plt.ylabel(r"terminal KL$(\,\hat p_N \,\|\, p\,)$")
    plt.title(r"Terminal KL versus number of steps $N$")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_headline.png")); plt.close()
    io.log("fig_headline.png", "figs.log")


def fig_resonance():
    """Multi-N: the order-vs-churn resonance, drawn at several step counts so the sharpening to a spike
    at lambda* is visible (a broad bump at small N tightening to a narrow order-4 spike at large N)."""
    from figstyle import COL
    r = io.load("e19_order_field")
    if r and "order" in r:
        lams = np.array(r["lams"]); P = np.clip(np.array(r["order"]), 1.8, 4.2)   # [lam][Npair]
        Nmid = np.array(r.get("Nmid", [(p[0] * p[1]) ** 0.5 for p in r["Npairs"]]))
        lstar = r["lambda_star"]
        idx = np.unique(np.linspace(0, len(Nmid) - 1, 4).round().astype(int))
        cols = plt.cm.plasma(np.linspace(0.12, 0.78, len(idx)))
        plt.figure(figsize=(7, 4.6))
        for j, c in zip(idx, cols):
            plt.plot(lams, P[:, j], "-", color=c, lw=1.9, label=rf"$N\approx{int(Nmid[j])}$")
        plt.axvline(lstar, ls=":", color=COL["lstar"], lw=1.8, label=rf"$\lambda^\star={lstar:.3f}$")
        plt.axhline(2, ls="--", color=COL["ref2"], alpha=0.7); plt.axhline(4, ls="--", color=COL["ref4"], alpha=0.5)
        plt.xlabel(r"churn $\lambda$"); plt.ylabel("measured KL order $p$")
        plt.title(r"the order resonance sharpens to four at $\lambda^\star$ as $N$ grows")
        plt.ylim(1.8, 4.25); plt.legend(fontsize=8.5, ncol=2, framealpha=0.9); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_resonance.png")); plt.close()
        io.log("fig_resonance.png", "figs.log")
        return
    r = io.load("e03_resonance")                                                  # fallback (single N)
    if not r: return
    plt.figure(figsize=(7, 4))
    plt.plot(r["lams"], r["orders"], "-", color=COL["lstar"])
    plt.axvline(r["lambda_star"], ls=":", color="k", alpha=0.6, label=rf"$\lambda^\star={r['lambda_star']:.3f}$")
    plt.axhline(2, ls="--", color=COL["ref2"], alpha=0.5); plt.axhline(4, ls="--", color=COL["ref4"], alpha=0.4)
    plt.xlabel(r"churn $\lambda$"); plt.ylabel("measured KL order $p$")
    plt.title(r"Measured KL convergence order versus churn $\lambda$")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_resonance.png")); plt.close()
    io.log("fig_resonance.png", "figs.log")


def fig_lambda_law():
    from figstyle import COL
    r = io.load("e04_lambda_star_law")
    if not r: return
    rows = [x for x in r["rows"] if x["lambda_star"]]
    s2 = [x["s2"] for x in rows]; ls = [x["lambda_star"] for x in rows]
    plt.figure(figsize=(7, 4))
    plt.loglog(s2, ls, 'o', ms=5, color=COL["lstar"], label=r"measured $\lambda^\star(s^2)$")
    kappa = r["kappa_tail_mean"]
    plt.loglog(s2, [kappa * np.sqrt(x) for x in s2], '--', color=COL["ref4"], alpha=0.7,
               label=rf"$\kappa\sqrt{{s^2}}$ law, $\kappa={kappa:.3f}$")
    plt.loglog(s2, [np.sqrt(x) for x in s2], ':', color=COL["ode"], alpha=0.7,
               label=r"$\sqrt{s^2}$ (slope reference)")
    plt.xlabel(r"data variance $s^2$"); plt.ylabel(r"$\lambda^\star$")
    plt.title(r"Superconvergence churn $\lambda^\star$ versus data variance $s^2$")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_lambda_law.png")); plt.close()
    io.log("fig_lambda_law.png", "figs.log")


def fig_goldilocks():
    r = io.load("e07_goldilocks")
    if not r: return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # left: U-shaped KL(lambda) for several eps, with lambda* marked
    for k, c in r["u_curves"].items():
        ax[0].plot(c["lams"], c["KL"], label=rf"$\varepsilon={c['eps']}$")
    ax[0].axvline(r["lambda_star"], ls=":", color="k", alpha=0.6, label=r"$\lambda^\star$")
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"churn $\lambda$"); ax[0].set_ylabel("KL")
    ax[0].set_title(r"Terminal KL versus churn $\lambda$ at several $\varepsilon$")
    ax[0].legend(fontsize=8)
    # right: predicted vs measured optimal churn
    rows = r["opt_rows"]
    pred = [x["pred_lambda_opt"] for x in rows]; meas = [x["meas_lambda_opt"] for x in rows]
    ax[1].plot(pred, meas, 'o', color="C3", ms=4, alpha=0.7)
    lo, hi = min(pred + meas), max(pred + meas)
    ax[1].plot([lo, hi], [lo, hi], '--k', alpha=0.5, label="$y=x$")
    ax[1].set_xlabel(r"predicted $\lambda_{\mathrm{opt}}$ (Thm B)")
    ax[1].set_ylabel(r"measured argmin-KL")
    ax[1].set_title(r"Predicted versus measured optimal churn")
    ax[1].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_goldilocks.png")); plt.close()
    io.log("fig_goldilocks.png", "figs.log")


def fig_mixture():
    r = io.load("e10_mixture")
    if not r or "by_lambda" not in r: return
    plt.figure(figsize=(7, 4.5))
    for k, d in r["by_lambda"].items():
        Ns = [row["N"] for row in d["rows"]]; w2 = [max(row["w2_mean"], 1e-9) for row in d["rows"]]
        plt.loglog(Ns, w2, 'o-', ms=3.5, label=rf"$\lambda={d['lam']}$ (order {d['tail_order']:.2f})")
    plt.loglog(Ns, [w2[0] * (Ns[0] / N) ** 2 for N in Ns], ':k', alpha=0.5, label=r"$N^{-2}$ ref")
    plt.xlabel("$N$"); plt.ylabel(r"debiased $W_2^2$ to target")
    plt.title(r"Debiased $W_2^2$ versus steps $N$ (Gaussian mixture)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_mixture.png")); plt.close()
    io.log("fig_mixture.png", "figs.log")


def fig_floor():
    from figstyle import COL
    r = io.load("e11_learned")
    if not r or "part_A" not in r: return
    plt.figure(figsize=(6.6, 4.4))
    items = list(r["part_A"].items())
    cols = plt.cm.plasma(np.linspace(0.1, 0.82, len(items)))    # sequential by score error delta
    N0 = None
    for (k, d), c in zip(items, cols):
        Ns = d["Ns"]; N0 = N0 or Ns[0]
        plt.loglog(Ns, [max(x, 1e-19) for x in d["KL"]], 'o-', ms=3, color=c, label=rf"$\delta={d['delta']}$")
    # canonical reference slopes, shared across the paper
    kl0 = max(r["part_A"][items[-1][0]]["KL"][0], 1e-12)
    plt.loglog([N0, N0 * 64], [kl0, kl0 * 64 ** -4], '--', color=COL["ref4"], alpha=0.7, label=r"$N^{-4}$ (exact)")
    plt.xlabel("steps $N$"); plt.ylabel(r"terminal KL at $\lambda^\star$")
    plt.title(r"the $N^{-4}$ descent floors at $\sim\delta^2$ for a miscalibrated score")
    plt.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_floor.png")); plt.close()
    io.log("fig_floor.png", "figs.log")


def fig_invariance():
    r = io.load("e05_invariance")
    if not r: return
    Bs = np.array(r["Bs"]); Ts = np.array(r["Ts"]); G = np.array(r["lambda_star_grid"], dtype=float)
    plt.figure(figsize=(6.5, 5))
    im = plt.imshow(G, aspect="auto", origin="lower", cmap="cividis",
                    extent=[Ts.min(), Ts.max(), Bs.min(), Bs.max()])
    plt.colorbar(im, label=r"$\lambda^\star$")
    plt.xlabel(r"horizon $T$"); plt.ylabel(r"rate $B$")
    plt.title(r"$\lambda^\star$ over $(B,\,T)$ at fixed $s^2$")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_invariance.png")); plt.close()
    io.log("fig_invariance.png", "figs.log")


def fig_sensitivity():
    r = io.load("e12_sensitivity")
    if not r: return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, d in r["init_offset"].items():
        ax[0].loglog(d["Ns"], [max(x, 1e-19) for x in d["KL"]], 'o-', ms=3, label=rf"$e_0={d['e0']}$")
    ax[0].set_xlabel("$N$"); ax[0].set_ylabel("KL")
    ax[0].set_title(r"Terminal KL versus $N$ at several initialisation errors $e_0$")
    ax[0].legend(fontsize=8)
    fn = r["finite_N_opt"]; Ns = [x["N"] for x in fn]; lo = [x["lambda_opt"] for x in fn]
    ax[1].semilogx(Ns, lo, 'o-', color="C3")
    ax[1].axhline(r["lambda_star"], ls=":", color="k", label=rf"$\lambda^\star={r['lambda_star']:.3f}$")
    ax[1].set_xlabel("$N$"); ax[1].set_ylabel(r"finite-$N$ optimal churn")
    ax[1].set_title(r"Finite-$N$ optimal churn versus $N$")
    ax[1].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_sensitivity.png")); plt.close()
    io.log("fig_sensitivity.png", "figs.log")


def fig_coeff_validation():
    r = io.load("e02_coefficient")
    if not r: return
    rows = r["rows"]
    cc = np.array([x["C_closed"] for x in rows]); cr = np.array([x["C_rich"] for x in rows])
    plt.figure(figsize=(6, 5.2))
    plt.plot(cc, cr, 'o', ms=2.5, alpha=0.5, color="C0")
    lo, hi = float(min(cc.min(), cr.min())), float(max(cc.max(), cr.max()))
    plt.plot([lo, hi], [lo, hi], '--k', alpha=0.6, label="$y=x$")
    plt.xlabel(r"closed-form $C(\lambda)$"); plt.ylabel(r"Richardson estimate of $C(\lambda)$")
    plt.title(r"Closed-form versus numerical coefficient")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_coeff_validation.png")); plt.close()
    io.log("fig_coeff_validation.png", "figs.log")


def fig_schedules():
    r = io.load("e08_universality")
    if not r: return
    sch = r["schedules"]; names = list(sch.keys())
    C0 = [sch[n]["C0"] for n in names]
    order = [sch[n].get("order_at_star", float("nan")) for n in names]
    labels = [n.replace("_", " ").replace("(", "\n(") for n in names]
    y = np.arange(len(names))[::-1]   # top-to-bottom
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    teal, coral, accent = "#2a9d8f", "#e76f51", "#4361ee"
    cols = [teal if c < 0 else coral for c in C0]
    b0 = ax[0].barh(y, C0, color=cols, edgecolor="white", height=0.66)
    ax[0].axvline(0, color="0.3", lw=1.0)
    for rect, c in zip(b0, C0):
        ax[0].text(c + (0.4 if c >= 0 else -0.4), rect.get_y()+rect.get_height()/2,
                   f"{c:+.2f}", va="center", ha="left" if c >= 0 else "right", fontsize=8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].set_xlabel(r"$C(0)$"); ax[0].set_title(r"sign of $C(0)$ by schedule")
    ax[0].margins(x=0.18); ax[0].grid(axis="x", alpha=0.25); ax[0].grid(axis="y", visible=False)
    oo = [o if o == o else 0 for o in order]
    b1 = ax[1].barh(y, oo, color=accent, edgecolor="white", height=0.66)
    ax[1].axvline(4, ls="--", color="0.3", lw=1.2, label=r"$N^{-4}$ target")
    ax[1].axvline(2, ls=":", color="0.5", lw=1.0, label=r"$N^{-2}$ generic")
    for rect, o in zip(b1, oo):
        if o > 0: ax[1].text(o-0.1, rect.get_y()+rect.get_height()/2, f"{o:.2f}", va="center", ha="right", fontsize=8, color="white")
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel(r"convergence order at $\lambda^\star$"); ax[1].set_title(r"order at $\lambda^\star$ by schedule")
    ax[1].grid(axis="x", alpha=0.25); ax[1].grid(axis="y", visible=False); ax[1].legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_schedules.png")); plt.close()
    io.log("fig_schedules.png", "figs.log")


ALL = [fig_headline, fig_resonance, fig_lambda_law, fig_goldilocks, fig_mixture, fig_floor,
       fig_invariance, fig_sensitivity, fig_coeff_validation, fig_schedules]

if __name__ == "__main__":
    for f in ALL:
        try:
            f()
        except Exception as e:
            io.log(f"FIG ERROR {f.__name__}: {e}", "figs.log")
    print("figures in", FIG)
