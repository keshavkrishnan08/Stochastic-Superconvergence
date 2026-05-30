"""E45 - superconvergence survives a non-uniform step grid (as practical EDM samplers use).

Real samplers do not step uniformly in t; they cluster steps where the score varies fastest (near t=0),
typically a power-law grid t_k = T (1-k/N)^rho. The schedule theorem predicts the leading coefficient
becomes C_rho(lambda) = T * int Phi sigma / rho_density dt, still a smooth function of lambda with a sign
change, so a (shifted) cancellation churn lambda*_rho should still exist and still give order four. We
test this directly: build the non-uniform recursion, locate its cancellation churn by Richardson, and
measure the terminal-KL order there and beside it. Exact, deterministic, no sampling.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
import io_utils as io
from diffusion import vp_const, set_dps
from metrics import kl_gauss


def grid_times(N, T, rho):
    # t_0 = T (prior) down to t_N = 0 (data); rho>1 clusters steps near t=0.
    return [mp.mpf(T) * (1 - mp.mpf(k) / N) ** rho for k in range(N + 1)]


def v_nonuniform(sched, N, u, rho):
    """Terminal variance on a non-uniform grid via the exact affine EM map with variable dt_k."""
    u = mp.mpf(u); T = sched.T
    ts = grid_times(N, T, rho)
    v = sched.V(T)
    for k in range(N):
        tk = ts[k]; dt = ts[k] - ts[k + 1]
        A = sched.A(tk, u)
        v = (1 - A * dt) ** 2 * v + u * sched.b2(tk) * dt
    return v


def C_richardson_nu(sched, u, rho, Ns=(512, 1024, 2048)):
    g = [mp.mpf(N) * (v_nonuniform(sched, N, u, rho) - sched.s2) for N in Ns]
    xs = [1 / mp.mpf(N) for N in Ns]; k = len(Ns); V = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            V[i, j] = xs[i] ** j
    return mp.lu_solve(V, mp.matrix(g))[0]


def u_star_nu(sched, rho):
    f = lambda u: C_richardson_nu(sched, u, rho)
    if f(mp.mpf(0)) >= 0:
        return None
    lo, hi = mp.mpf(0), mp.mpf(0.1)
    while f(hi) < 0 and hi < 50:
        hi *= 1.7
    if f(hi) < 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def run(B=4.0, s2=2.0, T=5.0, dps=55):
    set_dps(dps)
    sched = vp_const(B, s2, T)
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    out = {}
    for rho in [1.5, 2.0, 3.0]:
        us = u_star_nu(sched, rho)
        if us is None:
            out[f"rho={rho}"] = {"u_star": None}; io.log(f"  e45 rho={rho}: no root"); continue
        ls = mp.sqrt(us)
        def order(u):
            kls = [kl_gauss(v_nonuniform(sched, N, u, rho), s2) for N in Ns]
            return [float(mp.log(kls[i] / kls[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i])) for i in range(len(Ns) - 1)]
        o_star = order(us); o_off = order((ls * mp.mpf("1.3")) ** 2)
        out[f"rho={rho}"] = {"lambda_star_nu": mp.nstr(ls, 8), "order_at_star_tail": o_star[-1],
                             "order_off_tail": o_off[-1], "order_at_star": o_star}
        io.log(f"  e45 rho={rho}: lam*_nu={mp.nstr(ls,6)} order*={o_star[-1]:.3f} off={o_off[-1]:.3f}")
    io.save("e45_nonuniform", {"config": {"B": B, "s2": s2, "T": T, "Ns": Ns, "dps": dps}, "grids": out})
    io.log("e45_nonuniform DONE")


if __name__ == "__main__":
    run()
