"""E34 -- Uniqueness / sign-change / nondegeneracy certification of the cancellation churn.

Backs the appendix sentence (proof of Thm super): "dense scans over s^2 in [1.01,256],
B in [1,16], T in [2,40] find one positive sign change, no sign change for the tested
s^2<=1 cases, and c2(lambda*) bounded away from zero on the robustness grid."

Two independent certifications, matching the two clauses of that sentence:
  (1) SIGN-CHANGE SCAN over the full named grid s^2 in [1.01,256], B in [1,16], T in [2,40]:
      tabulate the closed-form C(lambda) on a fine u=lambda^2 grid and COUNT positive sign
      changes -- certifies exactly one positive root for s^2>1, and zero for s^2<=1.
  (2) NONDEGENERACY on the robustness grid: evaluate the exact second-order coefficient
      c2(lambda*) at the located root -- certifies |c2(lambda*)| bounded away from zero
      (the exact N^-4 rate).
Plus the large-s^2 kappa sequence (reproduces 1.2054, 1.2069, 1.2072 -> kappa).

Deterministic high precision (mpmath). The C integral uses peak-aware subdivision so the
large-s^2 / large-u integrand (sharply peaked near t=0) stays fast and accurate.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import time
import mpmath as mp
from diffusion import set_dps
from coefficient import c2_closed_vp
from lambda_star import u_star_vp
import io_utils as io

NAME = "e34_uniqueness_cert"


def C_fast(B, u, s2, T):
    """C(lambda) = T * int_0^T Phi(t) sig(t) dt for VP-const, with subdivision near t=0
    where the integrand e^{-uBt}/V^{1+u} concentrates (scale 1/(uB)). Stable at large s2,u."""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); u = mp.mpf(u); one = mp.mpf(1)
    V = lambda t: s2 * mp.e ** (-B * t) + (one - mp.e ** (-B * t))
    Phi = lambda t: s2 ** (one + u) * mp.e ** (-u * B * t) / V(t) ** (one + u)
    sig = lambda t: (B ** 2 / 4) * (-V(t) + (one + u) ** 2 / V(t) - 2 * u)
    f = lambda t: Phi(t) * sig(t)
    a = T / (u * B + 1)                       # peak width near 0
    pts = [mp.mpf(0)]
    for m in (one, 4):
        p = m * a
        if 0 < p < T:
            pts.append(p)
    pts.append(T)
    pts = sorted(set(pts))
    return T * mp.quad(f, pts)


def positive_sign_changes(B, s2, T, u_max, n_grid):
    """Count sign changes of C(u) for u in (0, u_max] on a log-spaced grid (plus u=0)."""
    umin = mp.mpf(u_max) * mp.mpf(10) ** (-6)
    us = [mp.mpf(0)] + [umin * (mp.mpf(u_max) / umin) ** (mp.mpf(i) / (n_grid - 1))
                        for i in range(n_grid)]
    Cs = [C_fast(B, u, s2, T) for u in us]
    changes = 0
    for i in range(len(us) - 1):
        if Cs[i] != 0 and Cs[i] * Cs[i + 1] < 0:
            changes += 1
    return changes, Cs[0]


def run(dps=20,
        s2_pos=(1.01, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0),
        s2_neg=(0.5, 0.9, 1.0),
        Bs=(1.0, 2.0, 4.0, 8.0, 16.0),
        Ts=(2.0, 5.0, 12.0, 40.0),
        n_grid=36,
        c2_Bs=(2.0, 4.0, 8.0), c2_s2=(1.5, 2.0, 4.0, 8.0, 16.0), c2_T=6.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    t0 = time.time()

    # (1) sign-change scan: exactly one positive root for s2>1, zero for s2<=1
    pos_nodes, pos_bad = [], []
    for s2 in s2_pos:
        u_max = max(mp.mpf(5), 3 * mp.mpf(s2))
        for B in Bs:
            for T in Ts:
                nch, C0 = positive_sign_changes(B, s2, T, u_max, n_grid)
                node = {"s2": float(s2), "B": float(B), "T": float(T),
                        "C0": float(C0), "n_sign_changes": nch}
                pos_nodes.append(node)
                if nch != 1:
                    pos_bad.append(node)
        io.log(f"  [sign] s2={s2}: {len(Bs)*len(Ts)} nodes  ({time.time()-t0:.0f}s)")

    neg_nodes, neg_bad = [], []
    for s2 in s2_neg:
        u_max = mp.mpf(5)
        for B in Bs:
            for T in Ts:
                nch, C0 = positive_sign_changes(B, s2, T, u_max, n_grid)
                node = {"s2": float(s2), "B": float(B), "T": float(T),
                        "C0": float(C0), "n_sign_changes": nch}
                neg_nodes.append(node)
                if nch != 0 or C0 < 0:
                    neg_bad.append(node)
        io.log(f"  [sign] (s2<=1) s2={s2}: {len(Bs)*len(Ts)} nodes")

    # (2) nondegeneracy: c2(lambda*) bounded away from zero on the robustness grid
    set_dps(30)
    c2_nodes = []
    c2_abs_min = None
    for B in c2_Bs:
        for s2 in c2_s2:
            us = u_star_vp(B, s2, c2_T)
            if us is None:
                continue
            lam = mp.sqrt(us)
            c2 = c2_closed_vp(B, lam, s2, c2_T)
            c2_nodes.append({"B": float(B), "s2": float(s2), "T": float(c2_T),
                             "lambda_star": float(lam), "c2_at_star": float(c2)})
            ac2 = abs(c2)
            c2_abs_min = ac2 if c2_abs_min is None else min(c2_abs_min, ac2)
        io.log(f"  [c2] B={B}: done  ({time.time()-t0:.0f}s)")

    # large-s2 kappa sequence
    set_dps(50)
    kappa_seq = {}
    for s2 in (256.0, 1024.0, 4096.0):
        us = u_star_vp(4.0, s2, 5.0)
        kappa_seq[str(int(s2))] = None if us is None else float(mp.sqrt(us) / mp.sqrt(s2))

    summary = {
        "n_pos_nodes": len(pos_nodes),
        "all_pos_unique_root": len(pos_bad) == 0,
        "n_pos_violations": len(pos_bad),
        "n_neg_nodes": len(neg_nodes),
        "all_neg_no_root": len(neg_bad) == 0,
        "n_neg_violations": len(neg_bad),
        "n_c2_nodes": len(c2_nodes),
        "c2_abs_min_over_grid": None if c2_abs_min is None else float(c2_abs_min),
        "kappa_sequence_B4_T5": kappa_seq,
    }
    res = {"config": {"dps": dps, "s2_pos": list(s2_pos), "s2_neg": list(s2_neg),
                      "Bs": list(Bs), "Ts": list(Ts), "n_grid": n_grid,
                      "c2_grid": {"Bs": list(c2_Bs), "s2": list(c2_s2), "T": c2_T}},
           "summary": summary, "violations_pos": pos_bad, "violations_neg": neg_bad,
           "pos_nodes": pos_nodes, "neg_nodes": neg_nodes, "c2_nodes": c2_nodes}
    io.save(NAME, res)
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s | "
           f"unique-root@all={summary['all_pos_unique_root']} "
           f"(viol {summary['n_pos_violations']}/{summary['n_pos_nodes']}); "
           f"no-root s2<=1={summary['all_neg_no_root']} "
           f"(viol {summary['n_neg_violations']}/{summary['n_neg_nodes']}); "
           f"min|c2(lam*)|={summary['c2_abs_min_over_grid']:.4g} over {summary['n_c2_nodes']} nodes; "
           f"kappa_seq={kappa_seq}")
    return res


if __name__ == "__main__":
    run()
