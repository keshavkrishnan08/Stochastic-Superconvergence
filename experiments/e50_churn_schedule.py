"""E50 -- The cancellation ladder: a time-varying churn cancels the NEXT coefficient too (N^-4 -> N^-6).

A constant churn lambda* zeroes the leading discretisation coefficient C(lambda), lifting the variance-error
order from 1 to 2 (KL: 2 -> 4). A churn that varies over the trajectory adds a second functional degree of
freedom, so it can zero BOTH the leading coefficient C and the second coefficient c2 at once -- a double
cancellation that lifts the variance-error order to 3 (KL: 4 -> 6). We parametrise u(t)=lambda(t)^2 as
u0 + u1*(1 - t/T), find along the leading-cancellation curve u0(u1) the schedule slope u1 that also kills c2,
and measure the resulting order. If it reaches 3 (variance) / 6 (KL), superconvergence is the first rung of a
ladder: each extra schedule parameter buys one more order. Exact Gaussian recursion at extended precision.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from metrics import kl_gauss
import io_utils as io

NAME = "e50_churn_schedule"


def vN_tv(u0, u1, N, B, s2, T):
    """Exact terminal variance of the VP Euler--Maruyama sampler with time-varying churn
    u(t)=u0+u1*(1-t/T), evaluated on the uniform grid t_k=T-k*dt. Gaussian => exact affine recursion."""
    u0 = mp.mpf(u0); u1 = mp.mpf(u1); B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T)
    dt = T / N
    v = 1 + (s2 - 1) * mp.e ** (-B * T)                       # v_0 = V(T)
    for k in range(N):
        t = T - k * dt
        V = 1 + (s2 - 1) * mp.e ** (-B * t)
        u = u0 + u1 * (1 - t / T)                             # churn varies along the trajectory
        A = -B / 2 + (1 + u) / 2 * B / V
        v = (1 - A * dt) ** 2 * v + u * B * dt
    return v


def order_of(u0, u1, Ns, B, s2, T):
    errs = [abs(vN_tv(u0, u1, N, B, s2, T) - s2) for N in Ns]
    # order between consecutive Ns (variance-error order)
    ords = [float(mp.log(errs[i] / errs[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i]))
            for i in range(len(Ns) - 1) if errs[i] > 0 and errs[i + 1] > 0]
    return (sum(ords) / len(ords)) if ords else None, [float(e) for e in errs]


def root_u0(u1, Nref, B, s2, T, lo=0.2, hi=6.0):
    """Find u0 on the leading-cancellation curve: v_N(u0,u1)=s2 at the reference N."""
    f = lambda u0: vN_tv(u0, u1, Nref, B, s2, T) - mp.mpf(s2)
    a, b = mp.mpf(lo), mp.mpf(hi); fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    for _ in range(80):
        m = (a + b) / 2; fm = f(m)
        if fa * fm <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return (a + b) / 2


def run(dps=45, B=4.0, s2=4.0, T=5.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps); t0 = time.time()
    Nref = 2048
    Ns = [512, 1024, 2048, 4096]

    # baseline: constant churn (u1=0) at the leading-cancellation root -> expect variance order ~2
    u0_const = root_u0(0.0, Nref, B, s2, T)
    ord_const, _ = order_of(u0_const, 0.0, Ns, B, s2, T)
    io.log(f"  E50 constant churn: u0={float(u0_const):.4f} (lambda*={float(mp.sqrt(u0_const)):.4f}) "
           f"variance-order={ord_const:.3f}")

    # sweep the schedule slope u1; along each, sit on the leading-cancellation curve u0(u1) and measure order.
    # the order peaks where the second coefficient c2 also vanishes (double cancellation).
    best = {"u1": 0.0, "u0": float(u0_const), "order": ord_const or 0.0}
    scan = []
    u1 = mp.mpf("-3.0")
    while u1 <= mp.mpf("3.01"):
        u0 = root_u0(u1, Nref, B, s2, T)
        if u0 is not None:
            o, errs = order_of(u0, u1, Ns, B, s2, T)
            if o is not None:
                scan.append({"u1": float(u1), "u0": float(u0), "order": o})
                if o > best["order"]:
                    best = {"u1": float(u1), "u0": float(u0), "order": o}
        u1 += mp.mpf("0.25")

    # refine around the best u1
    if scan:
        c = mp.mpf(best["u1"]); step = mp.mpf("0.05"); u1 = c - mp.mpf("0.25")
        while u1 <= c + mp.mpf("0.2501"):
            u0 = root_u0(u1, Nref, B, s2, T)
            if u0 is not None:
                o, errs = order_of(u0, u1, Ns, B, s2, T)
                if o is not None and o > best["order"]:
                    best = {"u1": float(u1), "u0": float(u0), "order": o}
            u1 += step

    # KL at the best double-cancellation schedule, across N, to confirm KL order ~ 2x variance order
    kls = []
    for N in [256, 512, 1024, 2048, 4096]:
        v = vN_tv(best["u0"], best["u1"], N, B, s2, T)
        kls.append({"N": N, "KL": float(kl_gauss(v, s2)), "var_err": float(abs(v - s2))})
    kl_order = None
    if kls[-1]["KL"] > 0 and kls[0]["KL"] > 0:
        kl_order = float(mp.log(mp.mpf(kls[1]["KL"]) / kls[-1]["KL"]) / mp.log(mp.mpf(kls[-1]["N"]) / kls[1]["N"]))

    res = {"config": {"dps": dps, "B": B, "s2": s2, "T": T, "Nref": Nref, "Ns": Ns,
                      "schedule": "u(t)=u0+u1*(1-t/T)"},
           "constant": {"u0": float(u0_const), "lambda_star": float(mp.sqrt(u0_const)),
                        "variance_order": ord_const},
           "best_schedule": best, "variance_order_gain": best["order"] - (ord_const or 0.0),
           "kl_curve": kls, "kl_order_best": kl_order, "scan": scan}
    io.save(NAME, res)
    io.log(f"  E50 best schedule: u0={best['u0']:.4f} u1={best['u1']:.4f} "
           f"variance-order={best['order']:.3f} (vs {ord_const:.3f} constant); KL-order~{kl_order:.2f}")
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")
    return res


if __name__ == "__main__":
    run()
