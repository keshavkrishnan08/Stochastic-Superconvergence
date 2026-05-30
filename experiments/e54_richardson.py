"""E54 -- Superconvergence is not Richardson extrapolation (a hard reviewer's question), and it stacks.

A natural objection: a generic sampler can already reach order four by Richardson extrapolation, combining an
N-step and a 2N-step run to cancel the leading 1/N error. So what does the cancellation churn buy?

We answer exactly, on the VP Gaussian sampler, by comparing terminal KL against the *true cost* (total Euler
steps), for four schemes:
  (a) generic churn lambda=0, single run            -> KL ~ N^-2
  (b) lambda=0 with Richardson on (N, 2N)           -> KL ~ N^-4, but at 3x the steps
  (c) cancellation churn lambda*, single run        -> KL ~ N^-4, at 1x the steps
  (d) lambda* with Richardson on (N, 2N)            -> KL ~ N^-6 (the cancellations stack)

Two messages a reviewer needs: (i) lambda* matches Richardson's order at a THIRD of the cost (a single run,
no second solve, no tuning of an extrapolation table); and (ii) it is orthogonal to Richardson -- applying
both gives order six. So superconvergence is a distinct, free, and composable mechanism. Extended precision.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps, vp_const
from recursion import v_terminal
from lambda_star import lambda_star_vp
from metrics import kl_gauss
import io_utils as io

NAME = "e54_richardson"


def kl_order(series):
    """series: list of (cost, KL). Order in N (=cost up to constant) from the last two points."""
    (c0, k0), (c1, k1) = series[-2], series[-1]
    if k0 > 0 and k1 > 0 and c1 > c0:
        return float(mp.log(mp.mpf(k0) / k1) / mp.log(mp.mpf(c1) / c0))
    return None


def run(dps=50, B=4.0, s2=4.0, T=5.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps); t0 = time.time()
    sched = vp_const(B, s2, T)
    lam = lambda_star_vp(B, s2, T)
    lstar = lam ** 2
    Ns = [64, 128, 256, 512, 1024, 2048]

    def vN(u, N):
        return v_terminal(sched, N, u)

    out = {"generic": [], "generic_rich": [], "lstar": [], "lstar_rich": []}
    for N in Ns:
        # (a) generic lambda=0, cost N
        out["generic"].append((N, float(kl_gauss(vN(0.0, N), s2))))
        # (c) lambda*, cost N
        out["lstar"].append((N, float(kl_gauss(vN(lstar, N), s2))))
        # (b) Richardson on lambda=0: cancel 1/N term with (N,2N) -> 2 v_2N - v_N; cost N+2N=3N
        vR = 2 * vN(0.0, 2 * N) - vN(0.0, N)
        out["generic_rich"].append((3 * N, float(kl_gauss(vR, s2))))
        # (d) Richardson on lambda*: cancel 1/N^2 term -> (4 v_2N - v_N)/3; cost 3N
        vR2 = (4 * vN(lstar, 2 * N) - vN(lstar, N)) / 3
        out["lstar_rich"].append((3 * N, float(kl_gauss(vR2, s2))))

    orders = {k: kl_order(v) for k, v in out.items()}
    # cost to reach a fixed accuracy target: KL <= 1e-10, by log-log interpolation on (cost, KL)
    def cost_for(series, target=1e-10):
        import math
        xs = [(math.log(c), math.log(max(k, 1e-300))) for c, k in series]
        for i in range(len(xs) - 1):
            (lc0, lk0), (lc1, lk1) = xs[i], xs[i + 1]
            if (lk0 - math.log(target)) * (lk1 - math.log(target)) <= 0:
                f = (math.log(target) - lk0) / (lk1 - lk0)
                return float(math.exp(lc0 + f * (lc1 - lc0)))
        return None
    costs = {k: cost_for(v) for k, v in out.items()}

    io.save(NAME, {"config": {"dps": dps, "B": B, "s2": s2, "T": T, "Ns": Ns, "lambda_star": float(lam)},
                   "curves": {k: [[c, kk] for c, kk in v] for k, v in out.items()},
                   "kl_orders": orders, "cost_to_1e-10": costs})
    io.log(f"  E54 KL-orders vs cost: generic={orders['generic']:.2f} "
           f"generic+Rich={orders['generic_rich']:.2f} lambda*={orders['lstar']:.2f} "
           f"lambda*+Rich={orders['lstar_rich']:.2f}")
    if all(costs[k] for k in ("generic_rich", "lstar")):
        io.log(f"  E54 cost to KL=1e-10: lambda*={costs['lstar']:.0f} vs Richardson(lam=0)={costs['generic_rich']:.0f} "
               f"steps ({costs['generic_rich']/costs['lstar']:.1f}x cheaper)")
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r: return
    c = r["curves"]; o = r["kl_orders"]
    style = {"generic": (COL["ode"], "o-", rf"$\lambda{{=}}0$ (order {o['generic']:.1f})"),
             "generic_rich": (COL["sde"], "s--", rf"$\lambda{{=}}0$ + Richardson (order {o['generic_rich']:.1f})"),
             "lstar": (COL["lstar"], "o-", rf"$\lambda^\star$ (order {o['lstar']:.1f})"),
             "lstar_rich": ("#7b3fb5", "D--", rf"$\lambda^\star$ + Richardson (order {o['lstar_rich']:.1f})")}
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for k in ("generic", "generic_rich", "lstar", "lstar_rich"):
        arr = np.array(c[k]); col, ls, lab = style[k]
        ax.loglog(arr[:, 0], np.clip(arr[:, 1], 1e-300, None), ls, color=col, ms=4.5, lw=1.8, label=lab)
    ax.set_xlabel(r"total cost (Euler steps)"); ax.set_ylabel(r"terminal KL")
    ax.set_title(r"$\lambda^\star$ matches Richardson at a third of the cost, and stacks with it")
    ax.legend(fontsize=8.8, loc="lower left", framealpha=0.95); ax.grid(True, which="both", alpha=0.18)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_richardson.png")); plt.close()
    io.log("fig_richardson.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
