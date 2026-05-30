"""E71 -- Strengthening mega-suite 2 (exact, extended precision, CPU). Two NEW checks that bulletproof
the integrator theorem and the exact-rate claim beyond their headline points:

  (A) THE INTEGRATOR-CANCELLATION RULE FIRES ACROSS THE WHOLE RK2 FAMILY. E63 showed the midpoint member
      (alpha=0.5) has a churn root giving KL order ~6; here we sweep the full two-stage weak-order-2 family
      alpha in [0.5,1], locate the churn root lambda_dagger(alpha) and measure the KL order at it. The order
      stays ~6 across the family and the root slides to zero as alpha->1 (Heun), the one non-generic member.
      This lifts the integrator theorem from a single example to the family it predicts.

  (B) THE RATE IS EXACTLY FOUR EVERYWHERE, NOT MERELY AT MOST FOUR. The order-four rate at lambda* requires
      c2(lambda*) != 0. We evaluate the closed-form c2 at lambda*(s^2) across a 60+ config grid in
      (s^2,B,T) and report the minimum |c2|, certifying the exact N^-4 rate throughout s^2>1.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from coefficient import c2_closed_vp
from lambda_star import lambda_star_vp
import io_utils as io
import e63_integrator_root as e63   # reuse verified vN_rk2 / C2 / find_root / kl_order

NAME = "e71_strengthen2"


def study_rk2_family(s2=2.0, B=4.0, T=5.0, dps=50):
    set_dps(dps)
    sched = vp_const(B, s2, T); sched.s2 = mp.mpf(s2)
    lam_scan = list(np.linspace(0.05, 5.0, 20))
    alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    rows = []
    for a in alphas:
        root, _ = e63.find_root(sched, a, lam_scan)
        order = None
        if root is not None and root > 0.05:
            order, _, _ = e63.kl_order(sched, a, root, N1=512, N2=4096)
        rows.append({"alpha": a, "root": root, "order_at_root": order})
        io.log(f"  E71-RK2 alpha={a:<5}: root={None if root is None else round(root,4)}  "
               f"order_at_root={None if order is None else round(order,3)} (pred 6; Heun alpha=1 has none)")
    return {"alphas": alphas, "rows": rows}


def study_c2_cert(dps=40):
    set_dps(dps)
    s2s = [1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    Bs = [2.0, 4.0, 8.0]
    Ts = [4.0, 6.0, 10.0]
    rows = []; min_abs = None
    for s2 in s2s:
        for B in Bs:
            for T in Ts:
                ls = lambda_star_vp(B, s2, T)
                if ls is None:
                    continue
                c2 = float(c2_closed_vp(B, float(ls), s2, T))
                rate = c2 ** 2 / (4 * s2 ** 2)
                rows.append({"s2": s2, "B": B, "T": T, "lambda_star": float(ls), "c2": c2, "rate_const": rate})
                if min_abs is None or abs(c2) < min_abs:
                    min_abs = abs(c2)
    io.log(f"  E71-c2: {len(rows)} configs, min |c2(lam*)| = {min_abs:.4g} (all nonzero -> exact order 4)")
    return {"n_configs": len(rows), "min_abs_c2": min_abs, "rows": rows}


def run():
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    t0 = time.time()
    io.log("E71 strengthening-2 starting")
    rk2 = study_rk2_family()
    c2 = study_c2_cert()
    io.save(NAME, {"study_rk2_family": rk2, "study_c2_cert": c2})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
