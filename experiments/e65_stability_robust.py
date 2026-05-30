"""E65 -- Robustness of the score-error stability theorem (Thm 4) beyond the constant-delta model.

Two checks a hard reviewer demands, both exact:

 (A) FLOOR-DROP RANGE. The floor-lowering factor (beta(0)/beta(lam*))^2 is quoted at one (s2,B,T)
     (~280x). Closed-form beta(lam) shows how it varies with data scale: it grows with s2 and ->1 as
     s2->1 (where lam*->0, so beta(lam*)->beta(0)). So the floor drop is the large-s2 end of a RANGE,
     not a universal constant -- and it vanishes continuously at the s2=1 boundary, consistent with the
     rest of the paper's boundary story.

 (B) TIME- AND SIGN-VARYING SCORE BIAS. The constant, variance-only miscalibration s_hat=-x/(V+delta)
     is the special case that keeps the recursion affine. We keep affinity/exactness but drop the
     'constant' part: a time-VARYING delta(t) -- a linear ramp, and a sign-CHANGING oscillation. If the
     steep N^-4 approach to the floor at lam* survives a time-varying and even sign-varying score bias,
     the extra order is not an artifact of the constant-delta model. (A genuinely NONLINEAR score error
     breaks Gaussianity of the iterate and is out of scope by construction; we say so in the paper.)

Exact extended precision; reuses the VP integrating factor and E64's closed-form beta. Minutes on CPU.
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from metrics import kl_gauss
from lambda_star import lambda_star_vp
import io_utils as io
from e64_stability_law import beta_closed

NAME = "e65_stability_robust"


def _recursion_dt(N, T, B, lam, s2, delta_fn):
    """Exact affine VP-Euler variance recursion with a (possibly time-varying) score bias delta_fn(t):
    A(t) = -B/2 + (1+u)/2 * B/(V(t)+delta_fn(t)). Stays affine for any deterministic delta_fn -> exact."""
    N = int(N); dt = mp.mpf(T) / N; B = mp.mpf(B); lam = mp.mpf(lam); s2 = mp.mpf(s2)
    T = mp.mpf(T); one = mp.mpf(1); u = lam ** 2
    V = lambda t: one + (s2 - 1) * mp.e ** (-B * t)
    v = V(T)
    for k in range(N):
        tk = T - k * dt
        A = -B / 2 + (one + u) / 2 * B / (V(tk) + delta_fn(tk))
        v = (1 - A * dt) ** 2 * v + u * B * dt
    return v


def _tail_order(Ns, kl, floor):
    """Least-squares log-log order over the CLEAN descent band 8*floor < KL < 3000*floor: excludes the
    near-floor flattening (small slope) and the smallest-N pre-asymptotic head, both of which bias a
    2-point estimate. This is the genuine convergence order of the pre-floor descent."""
    floor = max(floor, 1e-300)
    pts = [(N, k) for N, k in zip(Ns, kl) if 8 * floor < k < 3000 * floor]
    if len(pts) < 2:
        pts = list(zip(Ns, kl))[:3]
    lx = [math.log(N) for N, _ in pts]; ly = [math.log(k) for _, k in pts]; n = len(lx)
    mx = sum(lx) / n; my = sum(ly) / n
    return -sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def run(dps=40, B=4.0, s2=2.0, T=5.0, delta0=1e-3):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps); t0 = time.time()
    one = mp.mpf(1)
    lam_star = float(lambda_star_vp(B, s2, T))

    # ---- (A) floor-drop range vs s2 ----
    s2_grid = [1.05, 1.25, 1.5, 2.0, 4.0, 8.0, 16.0, 64.0]
    ratioA = []
    for s2v in s2_grid:
        lam = lambda_star_vp(B, s2v, T)
        if lam is None:
            ratioA.append({"s2": s2v, "ratio": None, "floor_drop": None}); continue
        b0 = float(beta_closed(B, s2v, T, 0.0)); bs = float(beta_closed(B, s2v, T, float(lam)))
        ratioA.append({"s2": s2v, "beta0": b0, "beta_star": bs, "ratio": b0 / bs,
                       "floor_drop": (b0 / bs) ** 2, "lam_star": float(lam)})
    io.log(f"  E65(A) floor-drop range: s2=1.05 -> {ratioA[0]['floor_drop']:.1f}x ... "
           f"s2=64 -> {ratioA[-1]['floor_drop']:.0f}x  (->1 as s2->1)")

    # ---- (B) order survives time-varying & sign-varying delta(t) ----
    Tm = mp.mpf(T); d0 = mp.mpf(delta0)
    profiles = {
        "constant":   lambda t: d0,                                   # baseline (Thm 4 model)
        "ramp":       lambda t: d0 * (2 * t / Tm),                    # 0 -> 2*delta0, mean delta0
        "decay":      lambda t: d0 * 2 * (one - t / Tm),              # 2*delta0 -> 0
        "sign-osc":   lambda t: d0 * mp.sin(4 * mp.pi * t / Tm),      # sign-CHANGING, amplitude delta0
    }
    Ns = [12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    # reference floor (constant profile) to locate the pre-floor window
    profB = {}
    for pname, dfn in profiles.items():
        kl0 = [float(kl_gauss(_recursion_dt(N, T, B, 0.0, s2, dfn), s2)) for N in Ns]
        kls = [float(kl_gauss(_recursion_dt(N, T, B, lam_star, s2, dfn), s2)) for N in Ns]
        floor0 = kl0[-1]; floor_s = kls[-1]
        ord0 = _tail_order(Ns, kl0, floor0); ords = _tail_order(Ns, kls, floor_s)
        profB[pname] = {"Ns": Ns, "kl0": kl0, "klstar": kls, "floor0": floor0, "floor_star": floor_s,
                        "order_det": ord0, "order_star": ords}
        io.log(f"  E65(B) delta(t)={pname:9}: order@lam=0 ~ {ord0:.2f} (pred 2)   "
               f"order@lam* ~ {ords:.2f} (pred 4)   floor*={floor_s:.2e}  ({time.time()-t0:.0f}s)")

    io.save(NAME, {"config": {"dps": dps, "B": B, "s2": s2, "T": T, "delta0": delta0,
                              "lambda_star": lam_star, "Ns": Ns},
                   "floor_range": ratioA, "profiles": profB})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r:
        return
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: floor-drop range vs s2
    fr = [x for x in r["floor_range"] if x["floor_drop"]]
    s2s = np.array([x["s2"] for x in fr]); drop = np.array([x["floor_drop"] for x in fr])
    axA.loglog(s2s, drop, "o-", color=COL["lstar"], lw=2, ms=6)
    axA.axhline(1.0, ls=":", color="0.5", lw=1.2)
    axA.annotate(r"$\to 1$ as $s^2\to1$", xy=(s2s[0], drop[0]), xytext=(s2s[0]*1.1, drop[0]*2.2),
                 fontsize=9, color="0.35")
    axA.set_xlabel(r"data scale $s^2$"); axA.set_ylabel(r"KL floor-drop $(\beta(0)/\beta(\lambda^\star))^2$")
    axA.set_title(r"the floor drop is a range, vanishing at $s^2{=}1$")
    axA.grid(True, which="both", alpha=0.18)

    # Panel B: order survives time-varying delta(t)
    prof = r["profiles"]; Nl = prof["constant"]["Ns"]; Ns = np.array(Nl, float)
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(prof)))
    for (pname, d), c in zip(prof.items(), cols):
        ostar = _tail_order(Nl, d["klstar"], d["floor_star"])   # windowed (clean) order
        axB.loglog(Ns, np.clip(d["klstar"], 1e-300, None), "o-", color=c, lw=1.8, ms=3.5,
                   label=rf"$\lambda^\star$, $\delta(t)$={pname} (ord {ostar:.1f})")
    # deterministic constant reference (N^-2)
    od = _tail_order(Nl, prof["constant"]["kl0"], prof["constant"]["floor0"])
    axB.loglog(Ns, np.clip(prof["constant"]["kl0"], 1e-300, None), "k--", lw=1.4, alpha=0.7,
               label=rf"$\lambda{{=}}0$ ($N^{{-2}}$, ord {od:.1f})")
    k0 = prof["constant"]["klstar"][0]
    axB.loglog(Ns, k0 * (Ns[0] / Ns) ** 4, ":", color=COL["ref4"], lw=1.1, alpha=0.6, label=r"$N^{-4}$")
    axB.set_xlabel(r"sampler steps $N$"); axB.set_ylabel(r"terminal KL")
    axB.set_title(r"$N^{-4}$ at $\lambda^\star$ survives time- and sign-varying $\delta(t)$")
    axB.legend(fontsize=7.6, framealpha=0.9); axB.grid(True, which="both", alpha=0.18)

    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_stability_robust.png")); plt.close()
    io.log("fig_stability_robust.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
