"""E31 - the order climbs to four as the score error vanishes (exact, deterministic).

The criticism of an idealised Gaussian result is "real scores are imperfect, so superconvergence
never shows." We answer it exactly. Take a LINEAR but miscalibrated score s_hat=-x/(V+delta): the
variance map stays affine, so the terminal KL is exact at any precision, and delta is an exactly
controllable score-error knob. As delta -> 0:
  (i) the best convergence order attained at lambda* climbs continuously from 2 toward 4;
  (ii) the pre-floor window (the N-range over which the N^-4 descent is visible) widens as ~1/delta,
       matching the crossover law N* ~ sqrt(c2/D)/(c*delta) of E25.
So the idealised order-four result is the delta->0 limit of an imperfect score, reached continuously,
not a knife-edge. Pure mpmath; no training, no sampling.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import vp_const, set_dps
from learned_score import miscalibrated_recursion
from lambda_star import lambda_star_vp
from metrics import kl_gauss
import io_utils as io


def local_orders(Ns, kls):
    out = []
    for i in range(len(Ns) - 1):
        lo, hi = kls[i], kls[i + 1]
        if lo <= 0 or hi <= 0:
            out.append(None); continue
        out.append(float(mp.log(lo / hi) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i])))
    return out


def run(s2=2.0, B=4.0, T=5.0, dps=80):
    set_dps(dps)
    ls = float(lambda_star_vp(B, s2, T))
    # fine geometric-ish N ladder from coarse to very fine
    Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
          3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    deltas = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0]
    rows = []
    for delta in deltas:
        t0 = time.time()
        kls = [kl_gauss(miscalibrated_recursion(N, T, B, ls, s2, delta), s2) for N in Ns]
        ords = local_orders(Ns, kls)
        valid = [o for o in ords if o is not None]
        order_max = max(valid) if valid else None
        # pre-floor window: largest N at which the local order is still >= 3 (still in the N^-4 descent)
        Nwin = 0
        for i, o in enumerate(ords):
            if o is not None and o >= 3.0:
                Nwin = Ns[i + 1]
        floor = float(kls[-1])
        rows.append({"delta": delta, "order_max": order_max, "N_window": Nwin,
                     "floor_KL": floor, "KL": [float(k) for k in kls]})
        io.log(f"  e31 delta={delta:7}: order_max={order_max:.3f}  N_window={Nwin}  floor={floor:.2e}"
               f"  ({time.time()-t0:.1f}s)")
    io.save("e31_floor_order", {"config": {"s2": s2, "B": B, "T": T, "dps": dps, "lambda_star": ls},
                                "Ns": Ns, "rows": rows})
    io.log("e31_floor_order DONE")


if __name__ == "__main__":
    run()
