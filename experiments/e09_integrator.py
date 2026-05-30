"""E9 — Integrator-order stacking (Thm E, exploratory). EM (weak order 1 -> KL N^-2) vs Heun
(order 2 -> KL N^-4). Measure orders at lambda in {0,1,lambda*_EM} and scan for a Heun churn
root that lifts the order further. Also a tiny consistency check: the exact per-step map
reproduces the target (KL ~ 0)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal, v_continuous_endpoint
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e09_integrator"


def run(s2=2.0, B=4.0, T=5.0, dps=80):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T)

    # consistency (Lemma 2): exact continuous reverse variance endpoint == s2 (KL -> 0)
    v_exact = v_continuous_endpoint(sched, 1.0)
    consistency_kl = float(kl_gauss(v_exact, s2))

    N1, N2 = 512, 8192
    def order(integr, lam):
        k1 = kl_gauss(v_terminal(sched, N1, mp.mpf(lam) ** 2, integrator=integr), s2)
        k2 = kl_gauss(v_terminal(sched, N2, mp.mpf(lam) ** 2, integrator=integr), s2)
        return local_order(k1, k2, N1, N2)

    em = {f"{lam}": order("EM", lam) for lam in [0.0, 1.0, float(lstar)]}
    heun = {f"{lam}": order("heun", lam) for lam in [0.0, 1.0, float(lstar)]}

    # Heun order-2 coefficient C2(lam) = lim N^2 (v_N - s2); find its root (-> N^-6).
    def C2_heun(lam, Ns=(512, 1024, 2048)):
        g = [mp.mpf(N) ** 2 * (v_terminal(sched, N, mp.mpf(lam) ** 2, integrator="heun") - s2) for N in Ns]
        xs = [1 / mp.mpf(N) for N in Ns]
        Vm = mp.matrix(len(Ns), len(Ns))
        for i in range(len(Ns)):
            for j in range(len(Ns)):
                Vm[i, j] = xs[i] ** j
        return mp.lu_solve(Vm, mp.matrix(g))[0]

    lams = np.linspace(0.0, 5.0, 41)
    C2_vals = [float(C2_heun(l)) for l in lams]
    # detect a sign change -> root -> N^-6
    heun_root = None
    for i in range(len(lams) - 1):
        if C2_vals[i] * C2_vals[i + 1] < 0:
            lo, hi = mp.mpf(lams[i]), mp.mpf(lams[i + 1])
            for _ in range(50):
                md = (lo + hi) / 2
                if C2_heun(float(lo)) * C2_heun(float(md)) <= 0: hi = md
                else: lo = md
            heun_root = float((lo + hi) / 2); break
    heun_root_order = order("heun", heun_root) if heun_root else None

    res = {"config": {"s2": s2, "B": B, "T": T, "dps": dps, "N1": N1, "N2": N2},
           "lambda_star_EM": float(lstar),
           "consistency_continuous_KL": consistency_kl,
           "EM_orders": em, "Heun_orders": heun,
           "Heun_C2_scan": {"lams": lams.tolist(), "C2": C2_vals},
           "heun_root": heun_root, "heun_root_order": heun_root_order}
    io.save(NAME, res)
    io.log(f"{NAME}: consistency(Lemma2) KL={consistency_kl:.2e} (->0); "
           f"EM order@lam*={em[str(float(lstar))]:.3f}; Heun generic order@1={heun['1.0']:.3f} (=2p,p=2); "
           f"Heun C2 root={heun_root} order@root={heun_root_order}")
    return res


if __name__ == "__main__":
    run()
