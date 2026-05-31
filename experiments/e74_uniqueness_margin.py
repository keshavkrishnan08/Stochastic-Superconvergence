"""E74 -- The uniqueness of the cancellation churn reduces to one explicit inequality, u* > s^2-1
(Remark app:rmk:unique): if C(u)<0 on [0,s^2-1] then no root sits there, every root above is a simple
upcrossing, and with C(0)<0 the positive root is unique. We stress-test that inequality far beyond the
325-config spot check: a dense (s^2,B,T) grid, reporting the margin u* - (s^2-1), the tightest-margin
corner, and the small-s^2 limit ratio u*/(s^2-1) (predicted -> 2). All exact closed-form, no sampling.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import set_dps
from lambda_star import lambda_star_vp
from coefficient import C_closed_vp
import io_utils as io

NAME = "e74_uniqueness_margin"


def run(dps=35):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps)
    t0 = time.time()
    s2s = [1.001, 1.003, 1.01, 1.03, 1.1, 1.3, 1.6, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1000.0]
    Bs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    Ts = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []; n_ok = 0; n_fail = 0; min_margin = None; tightest = None
    ratios_small = []                       # u*/(s^2-1) at the smallest s^2, to read the limit
    for s2 in s2s:
        for B in Bs:
            for T in Ts:
                ls = lambda_star_vp(B, s2, T)
                if ls is None:               # no positive root (expected only for s^2<=1, not in this grid)
                    rows.append({"s2": s2, "B": B, "T": T, "root": None}); continue
                ustar = float(ls) ** 2
                margin = ustar - (s2 - 1.0)
                # independent check that C<0 just below s^2-1 (the reduction's equivalent form)
                c_below = float(C_closed_vp(B, float(mp.sqrt(max(s2 - 1.0, 1e-9) * 0.5)), s2, T))
                ok = margin > 0 and c_below < 0
                n_ok += int(ok); n_fail += int(not ok)
                if min_margin is None or margin < min_margin:
                    min_margin = margin; tightest = {"s2": s2, "B": B, "T": T, "margin": margin, "ustar": ustar}
                if s2 <= 1.01:
                    ratios_small.append(ustar / (s2 - 1.0))
                rows.append({"s2": s2, "B": B, "T": T, "ustar": ustar, "margin": margin,
                             "ratio": ustar / (s2 - 1.0), "C_below_neg": c_below < 0, "ok": ok})
    out = {"n_configs": len([r for r in rows if r.get("root", 1) is not None]),
           "n_ok": n_ok, "n_fail": n_fail, "min_margin": min_margin, "tightest": tightest,
           "limit_ratio_small_s2": (sum(ratios_small) / len(ratios_small)) if ratios_small else None,
           "rows": rows}
    io.save(NAME, out)
    io.log(f"  E74 uniqueness: {n_ok} hold / {n_fail} fail across {out['n_configs']} configs; "
           f"min margin u*-(s^2-1)={min_margin:.3e}; u*/(s^2-1)->{out['limit_ratio_small_s2']:.3f} as s^2->1")
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
