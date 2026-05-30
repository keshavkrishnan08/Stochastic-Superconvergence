"""E63 -- The integrator-cancellation rule FIRES: a weak-order-2 integrator with a churn root -> KL N^-6.

Theorem (higher-order integrators) says a weak-order-p integrator has generic KL order 2p, and a churn
root of its OWN leading coefficient lifts the order to 2(p+1). E9 confirmed the generic part for Heun
(p=2 -> KL ~ N^-4) but found NO churn root for Heun, so the positive branch of the rule was never
exhibited. This experiment closes that gap exactly.

Heun is the alpha=1 member of the one-parameter family of two-stage, weak-order-2 integrators
    k1 = g(t_k, v),   k2 = g(t_k - alpha*dt, v + alpha*dt*k1),
    v_{k+1} = v + dt*[(1 - 1/(2 alpha)) k1 + (1/(2 alpha)) k2]
applied to the EXACT Gaussian variance ODE  dv/dtau = g(t,v) = -2 A(t,u) v + u b2(t),  t = T - tau.
Every member is weak order 2 (generic KL ~ N^-4); they differ only in the order-2 leading coefficient
C2(lambda; alpha). We scan alpha, locate a member whose C2(.;alpha) changes sign in lambda (a churn
root lambda_dagger), and measure the KL order there. At the root the order-2 variance error is
annihilated and KL is sixth order; off the root and for Heun it is fourth. Exact Gaussian recursion at
extended precision, no sampling, no fitted parameters.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e63_integrator_root"


def vN_rk2(sched, N, u, alpha):
    """Exact terminal variance of the alpha-family 2-stage RK2 integrator on the Gaussian variance ODE.
    alpha=1 reproduces the deterministic Heun used in recursion.py. mpmath-exact at current precision."""
    N = int(N); u = mp.mpf(u); alpha = mp.mpf(alpha); T = sched.T
    dt = T / N
    w1 = 1 - 1 / (2 * alpha)
    w2 = 1 / (2 * alpha)

    def g(t, vv):
        return -2 * sched.A(t, u) * vv + u * sched.b2(t)

    v = sched.V(T)
    for k in range(N):
        tk = T - k * dt
        ts = tk - alpha * dt                      # stage time (tau advances by alpha*dt => t decreases)
        k1 = g(tk, v)
        k2 = g(ts, v + alpha * dt * k1)
        v = v + dt * (w1 * k1 + w2 * k2)
    return v


def C2(sched, alpha, lam, Ns=(512, 1024, 2048)):
    """Leading order-2 coefficient C2 = lim_{N->inf} N^2 (v_N - s2) via Richardson in 1/N."""
    s2 = sched.s2
    g = [mp.mpf(N) ** 2 * (vN_rk2(sched, N, mp.mpf(lam) ** 2, alpha) - s2) for N in Ns]
    xs = [1 / mp.mpf(N) for N in Ns]
    Vm = mp.matrix(len(Ns), len(Ns))
    for i in range(len(Ns)):
        for j in range(len(Ns)):
            Vm[i, j] = xs[i] ** j
    return mp.lu_solve(Vm, mp.matrix(g))[0]


def find_root(sched, alpha, lams):
    """Return a churn root lambda_dagger of C2(.;alpha) bracketed on `lams`, or None."""
    vals = [C2(sched, alpha, float(l)) for l in lams]
    for i in range(len(lams) - 1):
        if vals[i] * vals[i + 1] < 0:
            lo, hi = mp.mpf(lams[i]), mp.mpf(lams[i + 1])
            flo = C2(sched, alpha, float(lo))
            for _ in range(40):
                md = (lo + hi) / 2
                if flo * C2(sched, alpha, float(md)) <= 0:
                    hi = md
                else:
                    lo = md; flo = C2(sched, alpha, float(lo))
            return float((lo + hi) / 2), [float(v) for v in vals]
    return None, [float(v) for v in vals]


def kl_order(sched, alpha, lam, N1=512, N2=8192):
    s2 = sched.s2
    k1 = kl_gauss(vN_rk2(sched, N1, mp.mpf(lam) ** 2, alpha), s2)
    k2 = kl_gauss(vN_rk2(sched, N2, mp.mpf(lam) ** 2, alpha), s2)
    return float(local_order(k1, k2, N1, N2)), float(k1), float(k2)


def run(s2=2.0, B=4.0, T=5.0, dps=80):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T)
    sched.s2 = mp.mpf(s2)
    t0 = time.time()

    lam_scan = list(np.linspace(0.05, 5.0, 30))

    # 1) Heun (alpha=1) recovers the E9 null: scan its C2 for a root (expect none).
    heun_root, heun_C2 = find_root(sched, 1.0, lam_scan)
    io.log(f"{NAME}: Heun(alpha=1) churn root = {heun_root} (E9 null check)")

    # 2) Scan the alpha family for members whose C2 has a churn root. Headline the canonical
    #    midpoint method (alpha=0.5); Heun (alpha=1) is the no-root member.
    alpha_grid = [0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
    family = []
    for a in alpha_grid:
        root, _ = find_root(sched, a, lam_scan)
        family.append({"alpha": a, "root": root})
        io.log(f"{NAME}:   alpha={a:<4} churn root = {root}")
    # headline the canonical midpoint method (alpha=0.5) if it fires, else the first member that does
    with_root = [(f["alpha"], f["root"]) for f in family if f["root"] is not None]
    fired = next((x for x in with_root if abs(x[0] - 0.5) < 1e-9), with_root[0] if with_root else None)

    result = {"config": {"s2": s2, "B": B, "T": T, "dps": dps},
              "lam_scan": lam_scan, "heun_C2_scan": heun_C2, "heun_root": heun_root,
              "alpha_family": family}

    if fired is not None:
        a, root = fired
        # generic order (lambda=1) vs order at the cancellation churn, for the firing integrator
        ord_generic, k1g, k2g = kl_order(sched, a, 1.0)
        ord_root, k1r, k2r = kl_order(sched, a, root)
        # also a fine KL-vs-N curve for the figure
        Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        curve_root = [float(kl_gauss(vN_rk2(sched, N, mp.mpf(root) ** 2, a), s2)) for N in Ns]
        curve_gen = [float(kl_gauss(vN_rk2(sched, N, mp.mpf(1.0) ** 2, a), s2)) for N in Ns]
        curve_heun = [float(kl_gauss(vN_rk2(sched, N, mp.mpf(1.0) ** 2, 1.0), s2)) for N in Ns]
        C2_curve_fire = [float(C2(sched, a, float(l))) for l in lam_scan]
        result.update({
            "fired_alpha": a, "fired_root": root,
            "order_generic_lam1": ord_generic, "order_at_root": ord_root,
            "Ns": Ns, "curve_root": curve_root, "curve_generic": curve_gen, "curve_heun": curve_heun,
            "C2_curve_fired": C2_curve_fire,
        })
        io.log(f"{NAME}: FIRED alpha={a} root lambda_dagger={root:.5f} | "
               f"KL order generic(lam=1)={ord_generic:.3f} (=2p=4) -> at root={ord_root:.3f} (->2(p+1)=6)")
    else:
        io.log(f"{NAME}: no firing member in alpha grid -> order-2 coefficient sign-definite across family")

    result["elapsed_s"] = time.time() - t0
    io.save(NAME, result)
    io.log(f"{NAME}: done in {result['elapsed_s']:.0f}s")
    return result


if __name__ == "__main__":
    run()
