"""E3 — The resonance curve (Thm A). Measured local order p(lambda) across a fine churn grid;
it sits at ~2 generically and spikes to 4 exactly at lambda*. Deterministic, high precision."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e03_resonance"


def run(s2=2.0, B=4.0, T=5.0, dps=70, n_lam=120, N1=512, N2=8192):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T)
    lams = np.linspace(0.0, 2.4, n_lam)
    orders, kl1s, kl2s = [], [], []
    for lam in lams:
        k1 = kl_gauss(v_terminal(sched, N1, mp.mpf(lam) ** 2), s2)
        k2 = kl_gauss(v_terminal(sched, N2, mp.mpf(lam) ** 2), s2)
        orders.append(local_order(k1, k2, N1, N2))
        kl1s.append(float(k1)); kl2s.append(float(k2))
    peak_i = int(np.argmax(orders))
    res = {"config": {"s2": s2, "B": B, "T": T, "dps": dps, "N1": N1, "N2": N2},
           "lambda_star": float(lstar), "lams": lams.tolist(), "orders": orders,
           "KL_N1": kl1s, "KL_N2": kl2s,
           "peak_lambda": float(lams[peak_i]), "peak_order": float(orders[peak_i])}
    io.save(NAME, res)
    io.log(f"{NAME}: lambda*={float(lstar):.5f}  peak order {orders[peak_i]:.3f} at lam={lams[peak_i]:.4f} "
           f"(grid step {lams[1]-lams[0]:.4f})")
    return res


if __name__ == "__main__":
    run()
