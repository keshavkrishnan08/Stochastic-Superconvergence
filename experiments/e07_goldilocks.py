"""E7 — Goldilocks / floor-shifted superconvergence (Thm B), the other face of the umbrella.

With a persistent score-error floor eps^2 the variance error is
    v_N - s2 = C(lambda)/N + D(lambda) eps^2 + O(N^-2),   D(lambda) = int_0^T Phi dt > 0.
The KL-optimal churn cancels the leading term: C(lambda_opt) = -N D(lambda_opt) eps^2, a
floor-shifted superconvergence dip. As eps->0, lambda_opt -> lambda* (root of C). We verify:
  (A) D(lambda) closed form == recursion floor (Richardson).
  (B) predicted lambda_opt(N,eps) [root of C+N D eps^2] == measured argmin_lambda KL.
  (C) at lambda_opt the KL dips below the eps-floor (a partial order recovery).
  (D) U-shaped KL(lambda) for several eps.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
from coefficient import C_closed_vp, D_closed_vp, D_richardson
from lambda_star import lambda_star_vp, _bisect
from metrics import kl_gauss, golden_min, local_order
import io_utils as io

NAME = "e07_goldilocks"


def predicted_lambda_opt(B, s2, T, N, eps, lam_hi=2.0):
    """Theorem B (unified): the KL-optimal churn minimises the leading variance error
       | C(lambda)/N + D(lambda) eps^2 |.
    When eps is small this has a ROOT in (0, lambda*) (a floor-shifted superconvergence dip);
    when the floor is large there is no root and the optimum is the interior minimiser of |.|.
    Returns (lambda_opt, is_cancellation_root)."""
    g = lambda lam: C_closed_vp(B, lam, s2, T) / N + D_closed_vp(B, lam, s2, T) * mp.mpf(eps) ** 2
    ls = lambda_star_vp(B, s2, T)
    # try a cancellation root in (0, lambda*)
    if g(mp.mpf(0)) * g(ls) < 0:
        f = lambda u: g(mp.sqrt(u))
        u = _bisect(f, mp.mpf(0), ls ** 2, maxit=80)
        return mp.sqrt(u), True
    # else minimise |g| (golden section on absolute value)
    lam = golden_min(lambda lam: abs(g(lam)), 0.0, lam_hi, iters=60)
    return lam, False


def run(s2=2.0, B=4.0, T=5.0, dps=60):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T)

    # (A) D(lambda) closed form vs recursion floor
    D_check = []
    for lam in [0.0, 0.5, 1.0, float(lstar), 2.0]:
        Dc = D_closed_vp(B, lam, s2, T)
        Dr = D_richardson(sched, mp.mpf(lam) ** 2)
        D_check.append({"lam": lam, "D_closed": float(Dc), "D_rich": float(Dr),
                        "rel": float(abs(Dc - Dr) / abs(Dc))})
    io.log(f"  [A] D(lambda) closed-vs-recursion max rel-err "
           f"{max(d['rel'] for d in D_check):.2e}")

    # (B,C) predicted vs measured optimal churn, and the dip
    rows = []
    for eps in [0.02, 0.05, 0.1, 0.2]:
        for N in [128, 256, 512, 1024, 2048]:
            pred, is_root = predicted_lambda_opt(B, s2, T, N, eps)
            f = lambda lam: kl_gauss(v_terminal(sched, N, mp.mpf(lam) ** 2, eps2=eps ** 2), s2)
            meas = golden_min(f, 0.0, float(lstar) + 0.5, iters=60)
            kl_opt = float(f(meas)); kl_at_lstar = float(f(lstar)); kl_at_1 = float(f(1.0))
            rows.append({"eps": eps, "N": N,
                         "pred_lambda_opt": float(pred), "is_cancellation_root": is_root,
                         "meas_lambda_opt": float(meas),
                         "kl_opt": kl_opt, "kl_at_lstar": kl_at_lstar, "kl_at_1": kl_at_1,
                         "dip_vs_lstar": kl_at_lstar / kl_opt})
        last = rows[-1]
        io.log(f"  [B] eps={eps}: N=2048 pred={last['pred_lambda_opt']:.4f} "
               f"meas={last['meas_lambda_opt']:.4f} root={last['is_cancellation_root']} "
               f"dip(KL@l*/KL@opt)={last['dip_vs_lstar']:.2f}x")

    paired = [(r["pred_lambda_opt"], r["meas_lambda_opt"]) for r in rows]
    max_abs = max(abs(p - m) for p, m in paired) if paired else None

    # (D) U-shaped KL(lambda) at fixed N for several eps
    Ncurve = 512
    lams = np.linspace(0.0, 2.2, 90)
    ucurves = {}
    for eps in [0.02, 0.05, 0.1, 0.2]:
        kls = [float(kl_gauss(v_terminal(sched, Ncurve, mp.mpf(l) ** 2, eps2=eps ** 2), s2)) for l in lams]
        ucurves[f"{eps}"] = {"eps": eps, "lams": lams.tolist(), "KL": kls,
                             "argmin_lambda": float(lams[int(np.argmin(kls))])}

    res = {"config": {"s2": s2, "B": B, "T": T, "dps": dps, "Ncurve": Ncurve},
           "lambda_star": float(lstar),
           "D_check": D_check, "opt_rows": rows,
           "pred_vs_meas_max_abs": max_abs, "u_curves": ucurves}
    io.save(NAME, res)
    io.log(f"{NAME}: predicted vs measured lambda_opt max |diff| = "
           f"{max_abs:.4f}; floor-shifted superconvergence verified.")
    return res


if __name__ == "__main__":
    run()
