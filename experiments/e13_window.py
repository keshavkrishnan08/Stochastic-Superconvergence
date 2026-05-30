"""E13 — Superconvergence window width (resonance sharpness / robustness of lambda*).
At finite N, churn within a band around lambda* still gives near-order-4. We quantify the
half-width Delta(N): the band where KL(lambda) <= 10x KL(lambda*). Prediction: since
KL ~ C'(lambda*)^2 (lambda-lambda*)^2/(4s^4) N^-2 near the root vs ~N^-4 at it, the half-width
shrinks as Delta(N) ~ sqrt(10)*sqrt(KL*(N))*2s^2/|C'| ~ N^-1. We measure the N-scaling of the band."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
from coefficient import C_closed_vp
from lambda_star import lambda_star_vp
from metrics import kl_gauss
import io_utils as io

NAME = "e13_window"


def run(s2=2.0, B=4.0, T=5.0, dps=80):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T)
    # C'(lambda*) via finite difference of closed form
    h = mp.mpf('1e-6')
    Cp = (C_closed_vp(B, lstar + h, s2, T) - C_closed_vp(B, lstar - h, s2, T)) / (2 * h)
    rows = []
    for N in [256, 1024, 4096]:
        kl_star = float(kl_gauss(v_terminal(sched, N, lstar ** 2), s2))
        thresh = 10 * kl_star
        # find band [lstar-dl, lstar+dr] where KL <= thresh, by scanning
        def kl_at(lam): return float(kl_gauss(v_terminal(sched, N, mp.mpf(lam) ** 2), s2))
        # search outward
        def edge(direction):
            lo, hi = float(lstar), float(lstar) + direction * 1.0
            # ensure hi is outside band
            for _ in range(60):
                mid = (lo + hi) / 2
                if kl_at(mid) <= thresh: lo = mid
                else: hi = mid
            return abs((lo + hi) / 2 - float(lstar))
        dr = edge(+1); dl = edge(-1)
        half = (dr + dl) / 2
        rows.append({"N": N, "kl_star": kl_star, "half_width": half, "dr": dr, "dl": dl})
        io.log(f"  N={N}: KL*={kl_star:.2e} half-width={half:.4e}")
    # fit half-width ~ N^p
    Ns = [r["N"] for r in rows]; hw = [r["half_width"] for r in rows]
    slope = float(np.polyfit(np.log(Ns), np.log(hw), 1)[0])
    res = {"config": {"s2": s2, "B": B, "T": T}, "lambda_star": float(lstar),
           "Cprime_at_star": float(Cp), "rows": rows, "halfwidth_slope": slope}
    io.save(NAME, res)
    io.log(f"{NAME}: half-width ~ N^{slope:.3f} (theory ~ N^-1); C'(lam*)={float(Cp):.3f}")
    return res


if __name__ == "__main__":
    run()
