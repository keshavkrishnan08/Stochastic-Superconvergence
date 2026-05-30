"""Wave-2 experiments (launched after the mega-suite to keep compute saturated toward the full budget).

E35 (exact, mpmath): higher-order integrator compounding at extended precision. A weak-order-2 Heun step
     gives generic KL order 4; we verify it across configs and confirm the EM->Heun lift is the same
     leading-coefficient cancellation seen for Euler-Maruyama.
E34 (trained, torch): a 2D anisotropic learned-score sampler. Train a score on N(0, diag(s1^2,s2^2)) and
     scan a single global churn; the empirical aggregate order stays at two (no single churn superconverges
     a non-degenerate spectrum), the learned-score counterpart of the exact anisotropic no-go.
E36 (trained, torch): denser architecture/epoch sweep to pin the empirical floor ~ RMSE^2 law with more
     points, extending E32.
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import mpmath as mp
import io_utils as io
from diffusion import vp_const, set_dps
from recursion import v_terminal, v_terminal_closed
from lambda_star import lambda_star_vp, u_star_richardson
from metrics import kl_gauss
from learned_score import train_dsm, sample_learned


def e35_heun_compounding():
    """Heun (weak order 2) gives generic KL order ~4 across configs; report EM vs Heun generic orders."""
    set_dps(60)
    Ns = [128, 256, 512, 1024, 2048, 4096]
    configs = [(4.0, 2.0, 5.0), (2.0, 3.0, 6.0), (8.0, 1.5, 4.0)]
    rows = []
    for (B, s2, T) in configs:
        sched = vp_const(B, s2, T)
        def order(u, integ):
            kls = [kl_gauss(v_terminal(sched, N, u, integrator=integ), s2) for N in Ns]
            return float(mp.log(kls[-3] / kls[-1]) / mp.log(mp.mpf(Ns[-1]) / Ns[-3]))
        u_gen = mp.mpf(1.0)
        em = order(u_gen, "EM"); heun = order(u_gen, "heun")
        rows.append({"B": B, "s2": s2, "T": T, "EM_generic": em, "Heun_generic": heun})
        io.log(f"  e35 B={B} s2={s2} T={T}: EM={em:.3f} Heun={heun:.3f}")
    io.save("e35_heun_compounding", {"config": {"Ns": Ns}, "rows": rows})
    io.log("e35_heun_compounding DONE")


def e34_aniso_trained(s1=1.5, s2=4.0, B=4.0, T=5.0):
    """2D diagonal target; train a shared-coordinate DSM score; scan a single global churn and measure the
    aggregate order. Expect order ~2 (no global superconvergence) since the per-mode roots differ."""
    set_dps(40)
    # per-mode exact roots (for reference)
    r1 = lambda_star_vp(B, s1, T); r2 = lambda_star_vp(B, s2, T)
    # train two single-coordinate scores (the modes decouple), reuse train_dsm per variance
    Ns = [32, 64, 128, 256, 512, 1024]; P = 120_000
    out = {"per_mode_roots": [float(r1), float(r2)], "Ns": Ns}
    try:
        m1, torch, rm1 = train_dsm(s1, B, T, width=128, depth=3, epochs=16000, seed=0)
        m2, _, rm2 = train_dsm(s2, B, T, width=128, depth=3, epochs=16000, seed=0)
    except Exception as ex:
        io.log(f"  e34 train ERR {ex}"); return
    lam_global = float((float(r1) + float(r2)) / 2)  # a single global churn between the two roots
    kls = []
    for N in Ns:
        v1, _ = sample_learned(m1, torch, N, T, B, lam_global, s1, P, seed=7)
        v2, _ = sample_learned(m2, torch, N, T, B, lam_global, s2, P, seed=7)
        kls.append(float(kl_gauss(mp.mpf(v1), s1) + kl_gauss(mp.mpf(v2), s2)))
    order = float(mp.log(mp.mpf(kls[-3]) / kls[-1]) / mp.log(mp.mpf(Ns[-1]) / Ns[-3]))
    out.update({"lam_global": lam_global, "KL": kls, "aggregate_order": order,
                "rmse": [float(rm1), float(rm2)]})
    io.save("e34_aniso_trained", out)
    io.log(f"e34_aniso_trained DONE: aggregate_order={order:.3f} (expect ~2)")


def e36_floor_law_dense(s2=2.0, B=4.0, T=5.0):
    """Denser architecture/epoch sweep to pin the empirical floor ~ rmse^2 law (extends E32)."""
    set_dps(40)
    ls = float(lambda_star_vp(B, s2, T))
    grid = [(48, 2, 4000), (96, 3, 6000), (160, 3, 12000), (224, 4, 24000), (320, 4, 45000)]
    Ns = [32, 128, 512]; P = 100_000
    cur = io.load("e36_floor_dense") or {"config": {"s2": s2, "B": B, "T": T, "lambda_star": ls}, "runs": []}
    have = {(r["width"], r["depth"], r["epochs"]) for r in cur["runs"]}
    for (w, dep, ep) in grid:
        if (w, dep, ep) in have:
            io.log(f"  e36 ({w},{dep},{ep}) cached"); continue
        t0 = time.time()
        try:
            model, torch, rmse = train_dsm(s2, B, T, width=w, depth=dep, epochs=ep, seed=0)
        except Exception as ex:
            io.log(f"  e36 ({w},{dep},{ep}) ERR {ex}"); continue
        if not (rmse == rmse):
            io.log(f"  e36 ({w},{dep},{ep}) NaN"); continue
        kls = [float(kl_gauss(mp.mpf(sample_learned(model, torch, N, T, B, ls, s2, P, seed=7)[0]), s2)) for N in Ns]
        cur["runs"].append({"width": w, "depth": dep, "epochs": ep, "residual_rmse": float(rmse),
                            "floor": min(kls)})
        io.save("e36_floor_dense", cur)
        io.log(f"  e36 w={w} d={dep} ep={ep}: rmse={rmse:.4f} floor={min(kls):.2e} ({time.time()-t0:.0f}s)")
    io.log("e36_floor_law_dense DONE")


if __name__ == "__main__":
    import traceback
    suite = [e35_heun_compounding, e34_aniso_trained, e36_floor_law_dense]
    pick = [a.lower() for a in sys.argv[1:]]
    if pick:
        suite = [fn for fn in suite if any(p in fn.__name__.lower() for p in pick)]
    io.log(f"wave2_suite START: {[fn.__name__ for fn in suite]}")
    t_all = time.time()
    for fn in suite:
        t = time.time()
        try:
            fn(); io.log(f"[{fn.__name__}] OK {time.time()-t:.0f}s")
        except Exception as ex:
            io.log(f"[{fn.__name__}] ERR {ex}\n{traceback.format_exc()}")
    io.log(f"wave2_suite DONE all in {time.time()-t_all:.0f}s")
