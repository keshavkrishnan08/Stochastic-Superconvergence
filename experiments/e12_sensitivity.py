"""E12 — Sensitivity / ablations.
(a) Initialization offset e0: super-convergence needs e0=0; for e0>0 the transient term
    survives and KL plateaus at ~ (Psi(T) e0)^2/(4 s^4) (the practical boundary, links Thm B).
(b) Finite-N optimal churn: argmin_lambda KL(N) -> lambda* as N->inf (the O(h) correction).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order, golden_min
import io_utils as io

NAME = "e12_sensitivity"


def Psi_T_vp(B, s2, T, u):
    """Transient multiplier Psi(T)=Phi(T,u)=s^{2(1+u)} e^{-uBT}/V(T)^{1+u}."""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); u = mp.mpf(u); one = mp.mpf(1)
    V_T = one + (s2 - 1) * mp.e ** (-B * T)
    return s2 ** (one + u) * mp.e ** (-u * B * T) / V_T ** (one + u)


def run(s2=2.0, B=4.0, T=5.0, dps=90):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T); ustar = lstar ** 2

    # (a) init-offset plateau at lambda*
    Ns = [2 ** k for k in range(5, 15)]   # 32 .. 16384
    Psi = Psi_T_vp(B, s2, T, ustar)
    init_curves = {}
    for e0 in [0.0, 1e-4, 1e-2, 1e-1]:
        kls = [float(kl_gauss(v_terminal(sched, N, ustar, e0=e0), s2)) for N in Ns]
        pred_plateau = float(kl_gauss(s2 + Psi * mp.mpf(e0), s2)) if e0 > 0 else 0.0
        tail_order = local_order(kls[-3], kls[-1], Ns[-3], Ns[-1])
        init_curves[f"{e0}"] = {"e0": e0, "Ns": Ns, "KL": kls,
                                "pred_plateau": pred_plateau, "tail_order": tail_order}

    # (b) finite-N optimal churn -> lambda*
    finiteN = []
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        f = lambda lam: kl_gauss(v_terminal(sched, N, mp.mpf(lam) ** 2), s2)
        lopt = float(golden_min(f, 0.5, 2.5, iters=60))
        finiteN.append({"N": N, "lambda_opt": lopt, "minus_lstar": lopt - float(lstar)})

    res = {"config": {"s2": s2, "B": B, "T": T, "dps": dps},
           "lambda_star": float(lstar), "Psi_T": float(Psi),
           "init_offset": init_curves, "finite_N_opt": finiteN}
    io.save(NAME, res)
    io.log(f"{NAME}: Psi(T)={float(Psi):.3e}; init e0=0 tail order="
           f"{init_curves['0.0']['tail_order']:.3f} (->4), e0=0.1 tail order="
           f"{init_curves['0.1']['tail_order']:.3f} (->0 plateau); "
           f"finite-N lam_opt(N=64)={finiteN[0]['lambda_opt']:.4f} -> "
           f"lam_opt(N=4096)={finiteN[-1]['lambda_opt']:.4f} (lam*={float(lstar):.4f})")
    return res


if __name__ == "__main__":
    run()
