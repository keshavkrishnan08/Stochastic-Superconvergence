"""Mega-suite: the heavy, time-using experiment wave.

E33 (exact ablations, mpmath, fast): the order-four effect is independent of the divergence used,
     a nonzero initialisation error masks it exactly like the score floor, and the anisotropic no-go
     holds as the dimension grows.
E32 (trained-score, torch-CPU, heavy): train denoising-score-matching MLPs across a quality ladder
     (width x depth x epochs x seeds), then RUN the learned sampler at lambda* across a step ladder and
     measure the Monte-Carlo terminal KL. Produces (i) realistic noisy convergence curves to sit beside
     the exact ones in the main text, and (ii) the empirical floor-vs-residual-error law (the noisy
     counterpart of the exact E31 delta^2 floor). Resumable: each trained config checkpoints to JSON.
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import mpmath as mp
import io_utils as io
from diffusion import vp_const, set_dps
from recursion import v_terminal, v_terminal_closed
from lambda_star import lambda_star_vp
from metrics import kl_gauss, w2sq_gauss
from learned_score import train_dsm, sample_learned


# ============================ E33: exact ablations =============================
def e33_divergence_equivalence():
    """At lambda* the order is four in BOTH KL and W2^2 (both quadratic in the variance error);
    off lambda* it is two in both. Confirms the order is a property of the sampler, not the metric."""
    set_dps(60)
    B, s2, T = 4.0, 2.0, 5.0
    sched = vp_const(B, s2, T); ls = lambda_star_vp(B, s2, T)
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    def order(seq):
        return [float(mp.log(seq[i] / seq[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i])) for i in range(len(Ns) - 1)]
    out = {}
    for tag, u in [("at_star", ls ** 2), ("off", (ls * mp.mpf("1.3")) ** 2)]:
        vs = [v_terminal_closed(sched, N, u) for N in Ns]
        kl = [kl_gauss(v, s2) for v in vs]; w2 = [w2sq_gauss(v, s2) for v in vs]
        out[tag] = {"KL_order_tail": order(kl)[-1], "W2_order_tail": order(w2)[-1]}
        io.log(f"  e33a {tag}: KL_order={out[tag]['KL_order_tail']:.3f}  W2_order={out[tag]['W2_order_tail']:.3f}")
    io.save("e33a_divergence_equiv", {"config": {"B": B, "s2": s2, "T": T}, "lambda_star": mp.nstr(ls, 8), "out": out})
    io.log("e33a_divergence_equiv DONE")


def e33_init_sensitivity():
    """A nonzero initialisation error e0 turns on the transient Psi(lambda*) e0, which is N-independent and
    masks superconvergence beyond a crossover, exactly as the score floor does. Order at lambda* vs N for an
    e0 ladder."""
    set_dps(70)
    B, s2, T = 4.0, 2.0, 5.0
    sched = vp_const(B, s2, T); ls = lambda_star_vp(B, s2, T); u = ls ** 2
    Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    rows = []
    for e0 in [0.0, 1e-8, 1e-6, 1e-4, 1e-2]:
        kls = [float(kl_gauss(v_terminal(sched, N, u, e0=e0), s2)) for N in Ns]
        # tail order over the last two points (floor-dominated -> ~0 when e0>0 and N large)
        tail = math.log(kls[-3] / kls[-1]) / math.log(Ns[-1] / Ns[-3]) if kls[-1] > 0 else None
        rows.append({"e0": e0, "Ns": Ns, "KL": kls, "tail_order": tail})
        io.log(f"  e33b e0={e0:.0e}: tail_order={tail:.3f}  floor(KL,16384)={kls[-1]:.2e}")
    io.save("e33b_init_sensitivity", {"config": {"B": B, "s2": s2, "T": T}, "lambda_star": mp.nstr(ls, 8), "rows": rows})
    io.log("e33b_init_sensitivity DONE")


def e33_highdim_aniso():
    """The anisotropic no-go as the dimension grows: random spectra of dimension d=2..16; the aggregate
    KL order at the best single churn stays at two (no global superconvergence) unless degenerate."""
    set_dps(45)
    B, T = 4.0, 5.0
    Ns = [512, 1024, 2048, 4096]
    rng = np.random.RandomState(0)
    out = {}
    for d in [2, 4, 8, 16]:
        spec = sorted(float(x) for x in np.exp(rng.uniform(np.log(1.3), np.log(8.0), d)))
        roots = [lambda_star_vp(B, s, T) for s in spec]
        rf = [float(r) for r in roots if r is not None]
        # aggregate KL at the per-mode-root average (a reasonable global churn)
        lam = mp.mpf(sum(rf) / len(rf))
        def agg(N):
            return sum(kl_gauss(v_terminal_closed(vp_const(B, s, T), N, lam ** 2), s) for s in spec)
        kls = [agg(N) for N in Ns]
        order = [float(mp.log(kls[i] / kls[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i])) for i in range(len(Ns) - 1)]
        spread = (max(rf) - min(rf)) / (sum(rf) / len(rf))
        out[f"d={d}"] = {"spectrum": spec, "root_rel_spread": float(spread), "aggregate_order_tail": order[-1]}
        io.log(f"  e33c d={d}: agg_order_tail={order[-1]:.3f}  root_spread={float(spread):.3f}")
    io.save("e33c_highdim_aniso", {"config": {"B": B, "T": T, "Ns": Ns}, "dims": out})
    io.log("e33c_highdim_aniso DONE")


# ============================ E32: trained-score wave =========================
QUALITY_LADDER = [
    # (width, depth, epochs) from coarse/short to large/long; each trained at 2 seeds
    (32, 2, 3000),
    (64, 3, 8000),
    (128, 3, 16000),
    (256, 4, 30000),
    (256, 4, 60000),
]
SEEDS = [0, 1]


def _valid(x):
    try:
        return x is not None and x == x  # not NaN
    except Exception:
        return False


def e32_trained_score(s2=2.0, B=4.0, T=5.0):
    """Train a quality ladder of DSM scores and run the learned sampler at lambda* over a step ladder."""
    set_dps(40)
    ls = float(lambda_star_vp(B, s2, T))
    Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
    P = 120_000
    cur = io.load("e32_trained_curves") or {"config": {"s2": s2, "B": B, "T": T, "lambda_star": ls,
                                                        "Ns": Ns, "P": P}, "runs": []}
    have = {(r["width"], r["depth"], r["epochs"], r["seed"]) for r in cur["runs"] if _valid(r.get("residual_rmse"))}
    for (w, dep, ep) in QUALITY_LADDER:
        for seed in SEEDS:
            key = (w, dep, ep, seed)
            if key in have:
                io.log(f"  e32 {key} cached, skip"); continue
            t0 = time.time()
            try:
                model, torch, rmse = train_dsm(s2, B, T, width=w, depth=dep, epochs=ep, seed=seed)
            except Exception as ex:
                io.log(f"  e32 {key} train ERR {ex}"); continue
            if not _valid(rmse):
                io.log(f"  e32 {key} NaN rmse, skip"); continue
            kls = []
            for N in Ns:
                v, _ = sample_learned(model, torch, N, T, B, ls, s2, P, seed=7)
                kls.append(float(kl_gauss(mp.mpf(v), s2)))
            floor = min(kls[-3:])  # floor ~ min KL at large N
            cur["runs"].append({"width": w, "depth": dep, "epochs": ep, "seed": seed,
                                "residual_rmse": float(rmse), "Ns": Ns, "KL": kls, "floor": floor})
            io.save("e32_trained_curves", cur)
            io.log(f"  e32 w={w} d={dep} ep={ep} seed={seed}: rmse={rmse:.4f} floor={floor:.2e} "
                   f"({time.time()-t0:.0f}s)")
    io.log("e32_trained_score DONE")


if __name__ == "__main__":
    import traceback
    suite = [e33_divergence_equivalence, e33_init_sensitivity, e33_highdim_aniso, e32_trained_score]
    pick = [a.lower() for a in sys.argv[1:]]
    if pick:
        suite = [fn for fn in suite if any(p in fn.__name__.lower() for p in pick)]
    io.log(f"mega_suite START: {[fn.__name__ for fn in suite]}")
    t_all = time.time()
    for fn in suite:
        t = time.time()
        try:
            fn(); io.log(f"[{fn.__name__}] OK {time.time()-t:.0f}s")
        except Exception as ex:
            io.log(f"[{fn.__name__}] ERR {ex}\n{traceback.format_exc()}")
    io.log(f"mega_suite DONE all in {time.time()-t_all:.0f}s")
