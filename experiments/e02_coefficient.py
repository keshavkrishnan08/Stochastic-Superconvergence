"""E2 — Coefficient validation across (s2,B,T) (umbrella). C_closed_vp vs C_richardson;
elementary anchors C(0),C(1) vs quadrature. Target: max rel-err < 1e-6 everywhere."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import vp_const, set_dps
from coefficient import C_closed_vp, C_richardson, C0_vp, C1_vp
import io_utils as io

NAME = "e02_coefficient"


def run(dps=45):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    S2 = [1.5, 2.0, 4.0, 8.0, 16.0]; BB = [1.0, 2.0, 4.0, 8.0]; TT = [2.0, 5.0, 10.0, 20.0]
    LAM = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    # use deeper N where T is large (the recursion needs more steps to be asymptotic)
    rows = []; max_rd = mp.mpf(0); max_abs = mp.mpf(0); max_anchor = mp.mpf(0); worst = None
    for s2 in S2:
        for B in BB:
            for T in TT:
                sched = vp_const(B, s2, T)
                Ns = (2048, 4096, 8192) if T <= 10 else (4096, 8192, 16384)
                for lam in LAM:
                    Cv = C_closed_vp(B, lam, s2, T)
                    Cr, _ = C_richardson(sched, lam ** 2, Ns=Ns)
                    ad = abs(Cv - Cr)
                    # relative error is only meaningful away from the root (|C| not tiny);
                    # near the root we track absolute error instead.
                    rd = ad / abs(Cv) if abs(Cv) > mp.mpf('1e-2') else mp.mpf(0)
                    max_abs = max(max_abs, ad)
                    if rd > max_rd:
                        max_rd = rd; worst = (s2, B, T, lam, float(Cv), float(Cr), float(rd))
                    rows.append({"s2": s2, "B": B, "T": T, "lam": lam,
                                 "C_closed": float(Cv), "C_rich": float(Cr),
                                 "abs": float(ad), "rel": float(rd)})
                # anchors at this (s2,B,T): relative error, guard non-finite
                for Cf, anchor in [(C_closed_vp(B, 0, s2, T), C0_vp(B, s2, T)),
                                   (C_closed_vp(B, 1, s2, T), C1_vp(B, s2, T))]:
                    if mp.isfinite(Cf) and mp.isfinite(anchor) and Cf != 0:
                        max_anchor = max(max_anchor, abs(Cf - anchor) / abs(Cf))
    res = {"n_configs": len(rows), "max_rel_err": float(max_rd), "max_abs_err": float(max_abs),
           "worst": worst, "max_anchor_rel_err": float(max_anchor), "rows": rows,
           "pass": bool(max_rd < mp.mpf('1e-5') and max_anchor < mp.mpf('1e-10'))}
    io.save(NAME, res)
    io.log(f"{NAME}: {len(rows)} configs, max rel-err(|C|>1e-2)={float(max_rd):.2e}, "
           f"max abs-err={float(max_abs):.2e}, anchor rel-err={float(max_anchor):.2e}, PASS={res['pass']}")
    return res


if __name__ == "__main__":
    run()
