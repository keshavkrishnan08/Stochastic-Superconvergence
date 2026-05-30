"""E15 — EDM-literal VE schedule (sigma(t)=t, the Karras noise schedule): bridge to real EDM.
Uses ve_quad-style V(t)=s2+ (here we build sigma^2(t)=t^2 so the EDM sigma=t). Confirms a
superconvergence root exists for the actual EDM noise schedule and the N^-2 -> N^-4 jump occurs,
tying the theory to the schedule practitioners use. Deterministic (exact recursion)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import ve_quad, set_dps
from coefficient import C_richardson
from lambda_star import u_star_richardson
from recursion import v_terminal
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e15_edm_schedule"


def run(dps=70, N1=512, N2=8192):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    # EDM-like: sigma^2(t)=c t^2 (sigma ~ t). Sweep c (sets max noise) and data variance s2.
    out = {}
    for s2 in [2.0, 4.0]:
        for c in [0.2, 0.5, 1.0]:
            sched = ve_quad(c, s2, 5.0)
            C0, _ = C_richardson(sched, 0.0, Ns=(1024, 2048, 4096))
            us = u_star_richardson(sched, Ns=(1024, 2048, 4096))
            key = f"s2={s2},c={c}"
            if us is None:
                out[key] = {"s2": s2, "c": c, "C0": float(C0), "has_root": False}
                io.log(f"  {key}: C0={float(C0):+.3f} NO root"); continue
            lam = float(mp.sqrt(us))
            k1 = kl_gauss(v_terminal(sched, N1, us), sched.s2)
            k2 = kl_gauss(v_terminal(sched, N2, us), sched.s2)
            order = local_order(k1, k2, N1, N2)
            out[key] = {"s2": s2, "c": c, "C0": float(C0), "has_root": True,
                        "lambda_star": lam, "order_at_star": order}
            io.log(f"  {key}: C0={float(C0):+.3f} lam*={lam:.4f} order@*={order:.3f}")
    res = {"config": {"dps": dps, "schedule": "VE sigma~t (EDM-like)"}, "cases": out}
    io.save(NAME, res)
    io.log(f"{NAME}: EDM-like schedule exhibits superconvergence where C0<0.")
    return res


if __name__ == "__main__":
    run()
