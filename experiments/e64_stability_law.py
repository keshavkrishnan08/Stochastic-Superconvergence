"""E64 -- The score-error stability law: a step-complexity SEPARATION, predicted and measured.

E63 showed that a churn-lambda* sampler reaches the deterministic sampler's best accuracy in far
fewer steps under a controlled score error delta. This experiment turns that win into a verified
SCALING LAW behind a new theorem (Thm: score-error stability). Three predictions, all from the
master decomposition:

  (i)  FLOOR.   The miscalibrated score s_hat = -x/(V+delta) keeps the recursion exactly affine, so
       the terminal variance has an N-independent floor bias e_delta(lambda) = beta(lambda) delta +
       O(delta^2), with beta a closed-form integral against the SAME integrating factor Phi. Hence the
       achievable KL floor is Theta(delta^2).  -> floor slope 2 in log-log; beta certified vs the
       continuous first-order ODE.
  (ii) APPROACH. Below the floor crossover the KL approaches its floor at the Thm-super order:
       N^-2 at lambda=0, N^-4 at lambda*.
  (iii)STEP COMPLEXITY. To reach within a fixed factor rho of the DETERMINISTIC floor, the
       deterministic sampler needs N_det = Theta(delta^-1) steps while the superconvergent sampler
       needs N_star = Theta(delta^-1/2). The step-saving N_det/N_star = Theta(delta^-1/2) grows without
       bound as the score improves: a quadratic reduction N_star ~ sqrt(N_det).

The true asymptotic floor is computed from the CONTINUOUS limit (mpmath odefun), not a finite-N proxy
(which is discretisation-contaminated at small delta). Exact extended precision; reuses E31's
miscalibrated recursion and the VP integrating factor. A few minutes on CPU.
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from learned_score import miscalibrated_recursion
from metrics import kl_gauss
from lambda_star import lambda_star_vp
import io_utils as io

NAME = "e64_stability_law"


def _Vfun(B, s2):
    one = mp.mpf(1)
    return lambda t: one + (s2 - 1) * mp.e ** (-B * t)


def floor_variance(B, s2, T, lam, delta):
    """True N->infinity terminal variance of the miscalibrated VP-Euler sampler: solve the
    continuous limit dv/dtau = -2 A_delta(T-tau) v + u B, v(0)=V(T), to tau=T.  (EM recursion
    v_{k+1}=(1-A dt)^2 v_k + uB dt has this as its dt->0 limit.)"""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); lam = mp.mpf(lam); delta = mp.mpf(delta)
    u = lam ** 2; one = mp.mpf(1); V = _Vfun(B, s2)
    A = lambda t: -B / 2 + (one + u) / 2 * B / (V(t) + delta)
    f = lambda tau, v: -2 * A(T - tau) * v + u * B
    sol = mp.odefun(f, mp.mpf(0), V(T), tol=mp.mpf(10) ** (-(mp.mp.dps - 8)))
    return sol(T)


def beta_closed(B, s2, T, lam):
    """Closed-form floor sensitivity beta(lambda) = d/d(delta) v_inf |_{delta=0}, from the
    first-order variation ODE  dw/dtau = -2 A0(T-tau) w + (1+u) B / V(T-tau),  w(0)=0,  beta=w(T).
    (This is the integral of (1+u)B/V against the homogeneous propagator -- same Phi as C and D.)"""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); lam = mp.mpf(lam)
    u = lam ** 2; one = mp.mpf(1); V = _Vfun(B, s2)
    A0 = lambda t: -B / 2 + (one + u) / 2 * B / V(t)
    f = lambda tau, w: -2 * A0(T - tau) * w + (one + u) * B / V(T - tau)
    sol = mp.odefun(f, mp.mpf(0), mp.mpf(0), tol=mp.mpf(10) ** (-(mp.mp.dps - 8)))
    return sol(T)


def _steps_to(Ns, curve, target):
    """log-log interpolate the N at which a decreasing KL curve first crosses `target`."""
    xs = [(math.log(N), math.log(max(float(k), 1e-300))) for N, k in zip(Ns, curve)]
    lt = math.log(target)
    for i in range(len(xs) - 1):
        if (xs[i][1] - lt) * (xs[i + 1][1] - lt) <= 0 and xs[i + 1][1] != xs[i][1]:
            f = (lt - xs[i][1]) / (xs[i + 1][1] - xs[i][1])
            return float(math.exp(xs[i][0] + f * (xs[i + 1][0] - xs[i][0])))
    return None


def _slope(xs, ys):
    """least-squares slope of log(ys) vs log(xs)."""
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    n = len(lx); mx = sum(lx) / n; my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def run(dps=36, B=4.0, s2=2.0, T=5.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps); t0 = time.time()
    lam_star = float(lambda_star_vp(B, s2, T))
    deltas = [3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2]
    Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
          2048, 3072, 4096, 6144, 8192, 12288, 16384]
    rhos = [2.0, 3.0, 5.0]

    # closed-form floor sensitivity (certified once; delta-independent)
    beta0 = float(beta_closed(B, s2, T, 0.0))
    beta_star = float(beta_closed(B, s2, T, lam_star))
    io.log(f"  E64 beta(0)={beta0:.5f}  beta(lam*)={beta_star:.5f}  lam*={lam_star:.5f}")

    rows = []
    for delta in deltas:
        v_inf0 = floor_variance(B, s2, T, 0.0, delta)         # deterministic asymptotic floor
        v_infs = floor_variance(B, s2, T, lam_star, delta)    # lambda* asymptotic floor
        kl_floor0 = float(kl_gauss(v_inf0, s2))
        kl_floors = float(kl_gauss(v_infs, s2))
        e0 = float(v_inf0 - s2); estar = float(v_infs - s2)
        # measured floor sensitivity (should match beta as delta->0)
        b0_meas = e0 / delta; bstar_meas = estar / delta
        # finite-N KL curves
        kl0 = [float(kl_gauss(miscalibrated_recursion(N, T, B, 0.0, s2, delta), s2)) for N in Ns]
        kls = [float(kl_gauss(miscalibrated_recursion(N, T, B, lam_star, s2, delta), s2)) for N in Ns]
        # step complexity to reach rho * deterministic-floor, for each rho (the band)
        nd = {}; nstar = {}
        for rho in rhos:
            tgt = rho * kl_floor0
            nd[rho] = _steps_to(Ns, kl0, tgt)
            nstar[rho] = _steps_to(Ns, kls, tgt)
        sv = (nd[3.0] / nstar[3.0]) if (nd[3.0] and nstar[3.0]) else None
        rows.append({"delta": delta, "kl_floor0": kl_floor0, "kl_floor_star": kl_floors,
                     "e0": e0, "e_star": estar, "beta0_meas": b0_meas, "beta_star_meas": bstar_meas,
                     "n_det": nd, "n_star": nstar, "saving": sv, "kl0": kl0, "klstar": kls})
        io.log(f"  E64 delta={delta:.0e}: floor0={kl_floor0:.2e} floor*={kl_floors:.2e} "
               f"(*{kl_floor0/max(kl_floors,1e-300):.0f} lower)  n_det={None if nd[3.0] is None else round(nd[3.0])} "
               f"n*={None if nstar[3.0] is None else round(nstar[3.0])} saving={None if sv is None else round(sv,2)} "
               f"({time.time()-t0:.0f}s)")

    # ---- fit the scaling exponents (the theorem's predictions) ----
    ds = [r["delta"] for r in rows]
    fit = {}
    fit["floor0_slope"] = _slope(ds, [r["kl_floor0"] for r in rows])          # predict 2
    fit["floor_star_slope"] = _slope(ds, [r["kl_floor_star"] for r in rows])  # predict 2
    nd3 = [(r["delta"], r["n_det"][3.0]) for r in rows if r["n_det"][3.0]]
    ns3 = [(r["delta"], r["n_star"][3.0]) for r in rows if r["n_star"][3.0]]
    sv3 = [(r["delta"], r["saving"]) for r in rows if r["saving"]]
    fit["n_det_slope"] = _slope([d for d, _ in nd3], [v for _, v in nd3])      # predict -1
    fit["n_star_slope"] = _slope([d for d, _ in ns3], [v for _, v in ns3])     # predict -0.5
    fit["saving_slope"] = _slope([d for d, _ in sv3], [v for _, v in sv3])     # predict -0.5
    # beta certification: measured beta at smallest delta vs closed form
    fit["beta0_closed"] = beta0; fit["beta_star_closed"] = beta_star
    fit["beta0_meas_small"] = rows[0]["beta0_meas"]; fit["beta_star_meas_small"] = rows[0]["beta_star_meas"]
    fit["beta0_relerr"] = abs(rows[0]["beta0_meas"] - beta0) / abs(beta0)
    fit["beta_star_relerr"] = abs(rows[0]["beta_star_meas"] - beta_star) / abs(beta_star)

    io.log(f"  E64 SLOPES  floor0={fit['floor0_slope']:.3f} (pred 2)  n_det={fit['n_det_slope']:.3f} (pred -1)  "
           f"n_star={fit['n_star_slope']:.3f} (pred -1/2)  saving={fit['saving_slope']:.3f} (pred -1/2)")
    io.log(f"  E64 BETA cert  beta(0): closed={beta0:.5f} meas={rows[0]['beta0_meas']:.5f} "
           f"(rel {fit['beta0_relerr']:.1e})   beta(lam*): closed={beta_star:.5f} meas={rows[0]['beta_star_meas']:.5f} "
           f"(rel {fit['beta_star_relerr']:.1e})")
    io.save(NAME, {"config": {"dps": dps, "B": B, "s2": s2, "T": T, "lambda_star": lam_star,
                              "deltas": deltas, "Ns": Ns, "rhos": rhos},
                   "rows": rows, "fit": fit})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r:
        return
    rows = r["rows"]; fit = r["fit"]
    ds = np.array([x["delta"] for x in rows], float)
    f0 = np.array([x["kl_floor0"] for x in rows], float)
    fs = np.array([x["kl_floor_star"] for x in rows], float)
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 170, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- Panel A: the delta^2 floor law ----
    axA.loglog(ds, f0, "o", color=COL["ode"], ms=6, label=r"deterministic floor $\lambda{=}0$")
    axA.loglog(ds, fs, "s", color=COL["lstar"], ms=6, label=r"churn floor $\lambda^\star$")
    ref = f0[0] * (ds / ds[0]) ** 2
    axA.loglog(ds, ref, ":", color="0.45", lw=1.4, label=r"$\propto\delta^{2}$ reference")
    axA.set_xlabel(r"score error $\delta$"); axA.set_ylabel(r"achievable KL floor")
    axA.set_title(rf"floor $=\Theta(\delta^2)$  (slope ${fit['floor0_slope']:.2f}$, ${fit['floor_star_slope']:.2f}$)")
    axA.legend(fontsize=8.5, framealpha=0.9); axA.grid(True, which="both", alpha=0.18)

    # ---- Panel B: the step-complexity separation (with rho-band) ----
    def band(key):
        def g(x, rho):  # JSON round-trips the rho dict keys to strings
            d = x[key]; v = d.get(str(rho), d.get(rho))
            return v if v else np.nan
        lo = np.array([g(x, 2.0) for x in rows], float)
        mid = np.array([g(x, 3.0) for x in rows], float)
        hi = np.array([g(x, 5.0) for x in rows], float)
        return lo, mid, hi
    nd_lo, nd_mid, nd_hi = band("n_det"); ns_lo, ns_mid, ns_hi = band("n_star")
    m = np.isfinite(nd_mid)
    axB.fill_between(ds[m], np.minimum(nd_lo[m], nd_hi[m]), np.maximum(nd_lo[m], nd_hi[m]),
                     color=COL["ode"], alpha=0.15)
    axB.loglog(ds[m], nd_mid[m], "o-", color=COL["ode"], lw=2, ms=5,
               label=rf"deterministic $N_{{\det}}\propto\delta^{{{fit['n_det_slope']:.2f}}}$")
    ms_ = np.isfinite(ns_mid)
    axB.fill_between(ds[ms_], np.minimum(ns_lo[ms_], ns_hi[ms_]), np.maximum(ns_lo[ms_], ns_hi[ms_]),
                     color=COL["lstar"], alpha=0.15)
    axB.loglog(ds[ms_], ns_mid[ms_], "s-", color=COL["lstar"], lw=2, ms=5,
               label=rf"churn $N_\star\propto\delta^{{{fit['n_star_slope']:.2f}}}$")
    # reference slopes anchored at the smallest delta
    axB.loglog(ds, nd_mid[0] * (ds / ds[0]) ** (-1.0), ":", color="0.45", lw=1.2, alpha=0.8)
    axB.loglog(ds, ns_mid[0] * (ds / ds[0]) ** (-0.5), ":", color="0.45", lw=1.2, alpha=0.8)
    axB.set_xlabel(r"score error $\delta$"); axB.set_ylabel(r"steps to reach the deterministic floor")
    axB.set_title(r"quadratic step-saving: $N_\star\sim\sqrt{N_{\det}}$ (band: $\rho\in[2,5]$)")
    axB.legend(fontsize=8.5, framealpha=0.9); axB.grid(True, which="both", alpha=0.18)

    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_stability_law.png")); plt.close()
    io.log("fig_stability_law.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
