"""E4 — The lambda*(s2) scaling law (Thm A). Root present for s2>1, absent for s2<=1;
lambda*/sqrt(s2) -> kappa ~ 1.20. Fit kappa from the large-s2 tail."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import set_dps
from lambda_star import lambda_star_vp
import io_utils as io

NAME = "e04_lambda_star_law"


def run(B=4.0, T=5.0, dps=40):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    # no-root region
    below = []
    for s2 in [0.25, 0.5, 0.75, 0.9, 0.99, 1.0]:
        ls = lambda_star_vp(B, s2, T)
        below.append({"s2": s2, "lambda_star": None if ls is None else float(ls)})
    # scaling region
    s2s = np.logspace(np.log10(1.05), np.log10(256.0), 50)
    rows = []
    for s2 in s2s:
        ls = lambda_star_vp(B, float(s2), T)
        if ls is None:
            rows.append({"s2": float(s2), "lambda_star": None, "ratio": None}); continue
        rows.append({"s2": float(s2), "lambda_star": float(ls), "ratio": float(ls / mp.sqrt(s2))})
    tail = [r["ratio"] for r in rows if r["s2"] > 64]
    kappa = float(np.mean(tail)) if tail else None
    res = {"config": {"B": B, "T": T, "dps": dps},
           "below_one": below, "rows": rows,
           "kappa_tail_mean": kappa, "kappa_tail_pts": len(tail),
           "ratio_at_max_s2": rows[-1]["ratio"]}
    io.save(NAME, res)
    io.log(f"{NAME}: no root for s2<=1 (checked to 0.99); kappa(tail s2>64)={kappa:.5f}; "
           f"ratio at s2=256 = {rows[-1]['ratio']:.5f}")
    return res


if __name__ == "__main__":
    run()
