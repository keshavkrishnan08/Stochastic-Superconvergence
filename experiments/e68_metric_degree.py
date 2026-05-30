"""E68 -- The metric-degree rule, comprehensively (exact). Bulletproofs the umbrella theorem's claim that
EVERY smooth divergence inherits the order jump by its DEGREE in the terminal variance error: degree-1
distances (total variation, W1) improve 1 -> 2, degree-2 divergences (KL, reverse-KL, W2^2, Hellinger^2,
chi^2) improve 2 -> 4. All computed from the exact recursion at extended precision (no sampling), at lambda=0
(generic) and at the cancellation churn lambda*.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
import io_utils as io
from diffusion import set_dps, vp_const
from recursion import v_terminal
from lambda_star import lambda_star_vp

NAME = "e68_metric_degree"


def divergences(v, s2):
    v = mp.mpf(v); s2 = mp.mpf(s2)
    sv, ss = mp.sqrt(v), mp.sqrt(s2)
    KL = (v / s2 - 1 - mp.log(v / s2)) / 2
    rKL = (s2 / v - 1 - mp.log(s2 / v)) / 2
    W2sq = (sv - ss) ** 2
    H2 = 1 - mp.sqrt(2 * sv * ss / (v + s2))                       # Hellinger^2
    chi2 = s2 / mp.sqrt(v * (2 * s2 - v)) - 1 if 2 * s2 > v else mp.nan   # chi^2(p||q)
    W1 = mp.sqrt(2 / mp.pi) * abs(sv - ss)
    # total variation between N(0,v) and N(0,s2): 2(Phi(x0/min)-Phi(x0/max)) closed form via crossover
    if abs(v - s2) < mp.mpf(10) ** (-(mp.mp.dps - 5)):
        TV = mp.mpf(0)
    else:
        x0 = mp.sqrt(abs(v * s2 * mp.log(s2 / v) / (s2 - v)))      # densities cross at +-x0
        cdf = lambda z: (1 + mp.erf(z / mp.sqrt(2))) / 2
        TV = abs((2 * cdf(x0 / ss) - 1) - (2 * cdf(x0 / sv) - 1))
    return {"var_err": abs(v - s2), "TV": TV, "W1": W1,                      # degree 1
            "KL": KL, "rKL": rKL, "W2sq": W2sq, "H2": H2, "chi2": chi2}       # degree 2


def _order(Ns, ys):
    lx = [math.log(n) for n in Ns]; ly = [math.log(max(float(y), 1e-300)) for y in ys]; n = len(lx)
    mx = sum(lx) / n; my = sum(ly) / n
    return -sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


DEG = {"var_err": 1, "TV": 1, "W1": 1, "KL": 2, "rKL": 2, "W2sq": 2, "H2": 2, "chi2": 2}
PRED = {1: (1, 2), 2: (2, 4)}   # (order at lambda=0, order at lambda*)


def run(dps=70):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps)
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    configs = [(4.0, 2.0, 5.0), (2.0, 4.0, 8.0), (8.0, 1.5, 4.0)]
    out = []
    for B, s2, T in configs:
        sched = vp_const(B, s2, T); lam = lambda_star_vp(B, s2, T); us = lam ** 2
        vs0 = {N: v_terminal(sched, N, mp.mpf(0)) for N in Ns}
        vss = {N: v_terminal(sched, N, us) for N in Ns}
        d0 = {N: divergences(vs0[N], s2) for N in Ns}
        ds = {N: divergences(vss[N], s2) for N in Ns}
        orders = {}
        for k in DEG:
            o0 = _order(Ns, [d0[N][k] for N in Ns])
            ostar = _order(Ns, [ds[N][k] for N in Ns])
            orders[k] = {"deg": DEG[k], "order_lam0": o0, "order_lamstar": ostar,
                         "pred_lam0": PRED[DEG[k]][0], "pred_lamstar": PRED[DEG[k]][1]}
        out.append({"B": B, "s2": s2, "T": T, "lambda_star": float(lam), "orders": orders})
        io.log(f"  E68 B={B} s2={s2} T={T} (lam*={float(lam):.3f}):")
        for k in DEG:
            o = orders[k]
            io.log(f"     {k:8} deg{o['deg']}: order@lam0={o['order_lam0']:.3f}(pred {o['pred_lam0']})  "
                   f"order@lam*={o['order_lamstar']:.3f}(pred {o['pred_lamstar']})")
    io.save(NAME, {"config": {"dps": dps, "Ns": Ns}, "results": out})
    io.log(f"{NAME} DONE")


def figure():
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r:
        return
    res = r["results"][0]["orders"]                       # canonical config for the bars
    keys = ["var_err", "TV", "W1", "KL", "rKL", "W2sq", "H2", "chi2"]
    labels = [r"$|v_N{-}s^2|$", "TV", r"$W_1$", "KL", "rev-KL", r"$W_2^2$", r"$H^2$", r"$\chi^2$"]
    o0 = [res[k]["order_lam0"] for k in keys]; ostar = [res[k]["order_lamstar"] for k in keys]
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 175, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w / 2, o0, w, color=COL["ode"], label=r"generic churn $\lambda{=}0$")
    ax.bar(x + w / 2, ostar, w, color=COL["lstar"], label=r"cancellation churn $\lambda^\star$")
    for xi, v in zip(x - w / 2, o0): ax.text(xi, v + 0.07, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, ostar): ax.text(xi, v + 0.07, f"{v:.2f}", ha="center", fontsize=7.5)
    for y in (1, 2, 4): ax.axhline(y, color="0.7", ls=":", lw=0.8)
    ax.axvline(2.5, color="0.5", ls="-", lw=0.8)
    ax.text(1.0, 2.95, "degree 1\n(linear)", ha="center", fontsize=9.5, color="0.35")
    ax.text(5.2, 2.6, "degree 2 (quadratic)", ha="center", fontsize=9.5, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("measured convergence order")
    ax.set_ylim(0, 5.0)
    ax.set_title(r"every divergence jumps by its degree at $\lambda^\star$: $1\!\to\!2$ or $2\!\to\!4$")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95); ax.grid(False)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_metric_degree.png")); plt.close()
    io.log("fig_metric_degree.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
