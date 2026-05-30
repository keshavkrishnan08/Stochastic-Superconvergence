"""E8 — Schedule / diffusion universality (Thm D). For VE-linear, VE-quadratic and a
time-varying VP ramp: confirm C(0)<0, find the super-convergence root via Richardson,
and measure KL order at the root (->4) vs at lambda=1 (->2)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import ve_linear, ve_quad, vp_ramp, vp_const, set_dps
from coefficient import C_richardson
from lambda_star import u_star_richardson
from recursion import v_terminal
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e08_universality"


def run(dps=70, N1=512, N2=8192):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    scheds = {
        "VP_const(B=4)":      vp_const(4.0, 2.0, 5.0),
        "VP_ramp(B0=2,B1=1)": vp_ramp(2.0, 1.0, 2.0, 5.0),
        "VE_linear(c=1.0)":   ve_linear(1.0, 2.0, 5.0),
        "VE_quad(c=0.3)":     ve_quad(0.3, 2.0, 5.0),
    }
    out = {}
    Ns = (1024, 2048, 4096)
    for name, sched in scheds.items():
        C0, _ = C_richardson(sched, 0.0, Ns=Ns)
        us = u_star_richardson(sched, Ns=Ns)
        entry = {"C0": float(C0), "has_root": us is not None}
        if us is not None:
            lam = mp.sqrt(us)
            k1 = kl_gauss(v_terminal(sched, N1, us), sched.s2)
            k2 = kl_gauss(v_terminal(sched, N2, us), sched.s2)
            order_star = local_order(k1, k2, N1, N2)
            k1b = kl_gauss(v_terminal(sched, N1, mp.mpf(1) ** 2), sched.s2)
            k2b = kl_gauss(v_terminal(sched, N2, mp.mpf(1) ** 2), sched.s2)
            order_1 = local_order(k1b, k2b, N1, N2)
            entry.update({"lambda_star": float(lam), "order_at_star": order_star,
                          "order_at_1": order_1})
            io.log(f"  {name:22s} C0={float(C0):+.4f} lam*={float(lam):.4f} "
                   f"order@*={order_star:.3f} order@1={order_1:.3f}")
        else:
            io.log(f"  {name:22s} C0={float(C0):+.4f}  NO super-convergence root")
        out[name] = entry
    res = {"config": {"dps": dps, "N1": N1, "N2": N2}, "schedules": out}
    io.save(NAME, res)
    io.log(f"{NAME} done.")
    return res


if __name__ == "__main__":
    run()
