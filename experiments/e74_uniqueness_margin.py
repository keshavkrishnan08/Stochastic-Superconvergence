"""E74 -- dense confirmation of the uniqueness inequality proved in Lemma app:lemma:location:
C(u)<0 on (0, s^2-1], so the cancellation root satisfies u* > s^2-1 (hence is unique and simple).
The binding test is at the boundary u=s^2-1: one coefficient evaluation per config (no root-find),
checking C(s^2-1)<0, plus the closed-form reduction G(s^2-1)>0. A handful of configs also report the
actual margin u*-(s^2-1) from the full root. All exact, no sampling.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from coefficient import C_closed_vp
from lambda_star import lambda_star_vp
import io_utils as io

NAME = "e74_uniqueness_margin"


def G_boundary(s2, kmax=4000):
    """The reduction G(u) of Lemma app:lemma:location at the boundary u=s^2-1, where (1+u)rho-u=0 so
    G = sum_{k>=1} rho^k/(s^2-1+k). It is a sum of positive terms (hence >0 analytically); we evaluate a
    truncated sum only to report its value. Plain-float to stay fast near rho->1."""
    rho = (s2 - 1.0) / s2; u = s2 - 1.0
    s = 0.0; term = rho
    for k in range(1, kmax + 1):
        s += term / (u + k); term *= rho
        if term < 1e-15:
            break
    return s


def run(dps=30):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps)
    t0 = time.time()
    s2s = [1.001, 1.003, 1.01, 1.03, 1.1, 1.3, 1.6, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1000.0]
    Bs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    Ts = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []; n_ok = 0; n_tot = 0; tightest_C = None; min_G = None
    for s2 in s2s:
        for B in Bs:
            for T in Ts:
                C_bd = float(C_closed_vp(B, float(mp.sqrt(s2 - 1.0)), s2, T))   # C at u=s^2-1; proof: <0
                G_bd = G_boundary(s2)                                           # reduction at boundary; proof: >0
                ok = (C_bd < 0) and (G_bd > 0)
                n_ok += int(ok); n_tot += 1
                if tightest_C is None or C_bd > tightest_C:                     # least-negative = tightest
                    tightest_C = C_bd
                if min_G is None or G_bd < min_G:
                    min_G = G_bd
                rows.append({"s2": s2, "B": B, "T": T, "C_at_boundary": C_bd, "G_at_boundary": G_bd, "holds": ok})
    # actual margins u*-(s^2-1) from the full root, for a few representative configs
    margins = []
    for s2, B, T in [(1.01, 4.0, 5.0), (2.0, 4.0, 5.0), (8.0, 4.0, 5.0), (64.0, 4.0, 5.0), (256.0, 4.0, 50.0)]:
        ls = lambda_star_vp(B, s2, T)
        if ls is not None:
            margins.append({"s2": s2, "B": B, "T": T, "margin": float(ls) ** 2 - (s2 - 1.0)})
    out = {"n_configs": n_tot, "n_holding": n_ok, "tightest_C_at_boundary": tightest_C,
           "min_G_at_boundary": min_G, "representative_margins": margins, "rows": rows}
    io.save(NAME, out)
    io.log(f"  E74: inequality C(s^2-1)<0 holds for {n_ok}/{n_tot} configs (s^2 in [1.001,1000], B in [0.5,32], "
           f"T in [1,50]); tightest C@boundary={tightest_C:.3e}; min G@boundary={min_G:.3e}")
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
