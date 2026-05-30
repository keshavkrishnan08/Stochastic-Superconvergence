"""E1 — Headline super-convergence (Thm A).
KL vs N for lambda in {0, 1, 2, lambda*}; measured local order -> {2,2,2,4}.
Deterministic exact recursion at high precision. Writes results/e01_headline.json.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
from coefficient import C_closed_vp
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e01_headline"


def run(s2=2.0, B=4.0, T=5.0, dps=120, Ns=None):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    Ns = Ns or [2 ** k for k in range(4, 17)]   # 16 .. 65536
    sched = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T)
    io.log(f"{NAME}: lambda*={mp.nstr(lstar,10)}  Ns={Ns[0]}..{Ns[-1]} dps={dps}")
    lams = {"0": mp.mpf(0), "1": mp.mpf(1), "2": mp.mpf(2), "lstar": lstar}
    curves = {}
    for tag, lam in lams.items():
        kls = []
        for N in Ns:
            v = v_terminal(sched, N, lam ** 2, integrator="EM")
            kls.append(kl_gauss(v, s2))
        orders = [local_order(kls[i], kls[i + 1], Ns[i], Ns[i + 1]) for i in range(len(Ns) - 1)]
        C = C_closed_vp(B, lam, s2, T)
        theory = [float(C ** 2 / (4 * mp.mpf(s2) ** 2) / mp.mpf(N) ** 2) for N in Ns]
        curves[tag] = {"lam": float(lam), "C": float(C), "Ns": Ns,
                       "KL": [float(k) for k in kls], "theory_N2": theory,
                       "orders": orders, "order_tail": orders[-1]}
        io.log(f"  lam={tag:5s} ({float(lam):.4f})  C={float(C):+.5f}  tail order={orders[-1]:.4f}  "
               f"KL[N={Ns[-1]}]={float(kls[-1]):.3e}")
    res = {"config": {"s2": s2, "B": B, "T": T, "dps": dps},
           "lambda_star": float(lstar), "Ns": Ns, "curves": curves}
    io.save(NAME, res)
    io.log(f"{NAME} done.")
    return res


if __name__ == "__main__":
    run()
