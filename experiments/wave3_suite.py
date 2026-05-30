"""Wave-3: reviewer-coverage experiments, chained after wave-2.

E38 (exact, mpmath): the order doubling is really the VARIANCE-ERROR order improving from 1 to 2; each
     divergence inherits it by its degree. We measure the order at lambda* and off it in four quantities,
     the raw variance error |v_N-s^2|, KL, squared-W2, and total variation, and show variance error and TV
     (both linear/degree-1 in v_N-s^2 to leading order, except |Dv| itself which is degree 1) behave as
     1->2 while KL and W2^2 (degree 2) behave as 2->4. This preempts the "what about TV?" question and
     pins down the mechanism: tuning lambda halves the variance-error order, and the metric squares it.
E42 (torch, heavy): a high-capacity, long-trained score pushed to low residual error, sampled at small N
     with many samples, to exhibit a TRAINED-score convergence order clearly above two over the pre-floor
     window (closing the gap that Monte-Carlo noise leaves in the lighter E32 nets).
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import mpmath as mp
import io_utils as io
from diffusion import vp_const, set_dps
from recursion import v_terminal_closed
from lambda_star import lambda_star_vp
from metrics import kl_gauss, w2sq_gauss
from learned_score import train_dsm, sample_learned


def tv_gauss(v, s2):
    """Total variation between N(0,v) and N(0,s2) (centred 1D Gaussians), exact via the crossing points."""
    v = mp.mpf(v); s2 = mp.mpf(s2)
    if v == s2:
        return mp.mpf(0)
    a2, b2 = (v, s2) if v < s2 else (s2, v)            # a<b std
    a, b = mp.sqrt(a2), mp.sqrt(b2)
    # densities cross at +/- c, c^2 = 2 a^2 b^2/(b^2-a^2) * ln(b/a)
    c = mp.sqrt(2 * a2 * b2 / (b2 - a2) * mp.log(b / a))
    Phi = lambda z: (1 + mp.erf(z / mp.sqrt(2))) / 2
    # TV = P_a(|x|<c) - P_b(|x|<c)  (narrower minus wider mass inside the crossing band)
    Pa = 2 * Phi(c / a) - 1; Pb = 2 * Phi(c / b) - 1
    return abs(Pa - Pb)


def e38_metric_orders():
    set_dps(60)
    B, s2, T = 4.0, 2.0, 5.0
    sched = vp_const(B, s2, T); ls = lambda_star_vp(B, s2, T)
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    def order(seq):
        return float(mp.log(seq[-3] / seq[-1]) / mp.log(mp.mpf(Ns[-1]) / Ns[-3]))
    out = {}
    for tag, u in [("at_star", ls ** 2), ("off", (ls * mp.mpf("1.3")) ** 2)]:
        vs = [v_terminal_closed(sched, N, u) for N in Ns]
        dv = [abs(v - s2) for v in vs]
        kl = [kl_gauss(v, s2) for v in vs]; w2 = [w2sq_gauss(v, s2) for v in vs]; tv = [tv_gauss(v, s2) for v in vs]
        out[tag] = {"var_err_order": order(dv), "KL_order": order(kl),
                    "W2sq_order": order(w2), "TV_order": order(tv)}
        io.log(f"  e38 {tag}: |dv|={out[tag]['var_err_order']:.3f} KL={out[tag]['KL_order']:.3f} "
               f"W2^2={out[tag]['W2sq_order']:.3f} TV={out[tag]['TV_order']:.3f}")
    io.save("e38_metric_orders", {"config": {"B": B, "s2": s2, "T": T}, "lambda_star": mp.nstr(ls, 8), "out": out})
    io.log("e38_metric_orders DONE")


def e42_trained_order(s2=2.0, B=4.0, T=5.0):
    """High-capacity, long-trained score; sample at small N with many samples to expose a pre-floor order
    above two for a genuinely trained score."""
    set_dps(40)
    ls = float(lambda_star_vp(B, s2, T))
    Ns = [16, 24, 32, 48, 64, 96]                      # pre-floor window (small N -> cheap to oversample)
    P = 1_000_000
    configs = [(384, 5, 80000, 5e-4, 0), (384, 5, 80000, 5e-4, 1)]
    cur = io.load("e42_trained_order") or {"config": {"s2": s2, "B": B, "T": T, "lambda_star": ls,
                                                       "Ns": Ns, "P": P}, "runs": []}
    have = {(r["width"], r["depth"], r["epochs"], r["seed"]) for r in cur["runs"]}
    for (w, dep, ep, lr, seed) in configs:
        if (w, dep, ep, seed) in have:
            io.log(f"  e42 ({w},{dep},{ep},{seed}) cached"); continue
        t0 = time.time()
        try:
            model, torch, rmse = train_dsm(s2, B, T, width=w, depth=dep, epochs=ep, lr=lr, seed=seed)
        except Exception as ex:
            io.log(f"  e42 train ERR {ex}"); continue
        if not (rmse == rmse):
            io.log(f"  e42 NaN rmse"); continue
        kls = [float(kl_gauss(mp.mpf(sample_learned(model, torch, N, T, B, ls, s2, P, seed=11)[0]), s2)) for N in Ns]
        # robust pre-floor slope: least-squares of log KL vs log N over the descending prefix
        lx = np.log(np.array(Ns, float)); ly = np.log(np.array(kls))
        k = max(3, int(np.argmin(ly)) + 1)             # use points down to the minimum (pre-floor)
        slope = float(np.polyfit(lx[:k], ly[:k], 1)[0])
        cur["runs"].append({"width": w, "depth": dep, "epochs": ep, "seed": seed,
                            "residual_rmse": float(rmse), "Ns": Ns, "KL": kls, "prefloor_order": -slope})
        io.save("e42_trained_order", cur)
        io.log(f"  e42 w={w} d={dep} ep={ep} seed={seed}: rmse={rmse:.4f} prefloor_order={-slope:.2f} "
               f"({time.time()-t0:.0f}s)")
    io.log("e42_trained_order DONE")


if __name__ == "__main__":
    import traceback
    suite = [e38_metric_orders, e42_trained_order]
    pick = [a.lower() for a in sys.argv[1:]]
    if pick:
        suite = [fn for fn in suite if any(p in fn.__name__.lower() for p in pick)]
    io.log(f"wave3_suite START: {[fn.__name__ for fn in suite]}")
    t_all = time.time()
    for fn in suite:
        t = time.time()
        try:
            fn(); io.log(f"[{fn.__name__}] OK {time.time()-t:.0f}s")
        except Exception as ex:
            io.log(f"[{fn.__name__}] ERR {ex}\n{traceback.format_exc()}")
    io.log(f"wave3_suite DONE all in {time.time()-t_all:.0f}s")
