"""E70 -- Strengthening mega-suite (exact, extended precision, CPU-fast). Two NEW generality checks
that bulletproof the main theorems beyond their canonical operating points:

  (A) STEP-SEPARATION ACROSS DATA SCALES. E64 certified N_det=Theta(delta^-1) vs N_star=Theta(delta^-1/2)
      at s^2=2. Here we re-run the full sweep at s^2 in {1.5,2,4,8,16} and confirm the quadratic
      step-saving holds at every scale, and that the floor-drop (beta(0)/beta(lam*))^2 grows with s^2
      (the range the Discussion quotes). Reuses E64's exact continuous-limit floor + closed-form beta.

  (B) THE PER-EIGENMODE FIX RESTORES ORDER 4 ACROSS DIMENSIONS. The anisotropic no-go says a single
      global churn gives aggregate KL order 2 for a non-degenerate spectrum; the fix is a per-eigenmode
      churn Lambda=diag(lambda*(s_i^2)). We verify, for random spectra of growing dimension d in
      {2,4,8,16,32,64}, that the best GLOBAL churn pins to order ~2 while the per-eigenmode churn restores
      order ~4 -- and that the gap is stable as d grows. Exact VP-Euler per mode, no sampling.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
import numpy as np
from diffusion import set_dps
from learned_score import miscalibrated_recursion
from metrics import kl_gauss, kl_gauss_aniso, local_order
from lambda_star import lambda_star_vp
import io_utils as io
import e64_stability_law as e64   # reuse floor_variance, beta_closed, _steps_to, _slope

NAME = "e70_strengthen"


def study_A(B, T, dps):
    """Step-separation across data scales."""
    set_dps(dps)
    s2s = [1.5, 2.0, 4.0, 8.0, 16.0]
    deltas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    Ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    rho = 3.0
    out = []
    for s2 in s2s:
        lam_star = float(lambda_star_vp(B, s2, T))
        beta0 = float(e64.beta_closed(B, s2, T, 0.0))
        beta_star = float(e64.beta_closed(B, s2, T, lam_star))
        floor_drop = (beta0 / beta_star) ** 2
        nd, ns = [], []
        for delta in deltas:
            kf0 = float(kl_gauss(e64.floor_variance(B, s2, T, 0.0, delta), s2))
            k0 = [float(kl_gauss(miscalibrated_recursion(N, T, B, 0.0, s2, delta), s2)) for N in Ns]
            ks = [float(kl_gauss(miscalibrated_recursion(N, T, B, lam_star, s2, delta), s2)) for N in Ns]
            nd.append(e64._steps_to(Ns, k0, rho * kf0))
            ns.append(e64._steps_to(Ns, ks, rho * kf0))
        dd = [(d, n) for d, n in zip(deltas, nd) if n]
        ds = [(d, n) for d, n in zip(deltas, ns) if n]
        nd_slope = e64._slope([d for d, _ in dd], [n for _, n in dd]) if len(dd) >= 2 else None
        ns_slope = e64._slope([d for d, _ in ds], [n for _, n in ds]) if len(ds) >= 2 else None
        out.append({"s2": s2, "lambda_star": lam_star, "beta0": beta0, "beta_star": beta_star,
                    "floor_drop": floor_drop, "n_det_slope": nd_slope, "n_star_slope": ns_slope})
        io.log(f"  E70-A s2={s2:5.1f}: lam*={lam_star:.4f}  floor-drop={floor_drop:.3g}x  "
               f"n_det~d^{nd_slope:+.3f} (pred -1)  n_star~d^{ns_slope:+.3f} (pred -0.5)")
    return {"s2s": s2s, "deltas": deltas, "Ns": Ns, "rho": rho, "rows": out}


def _aniso_orders(B, T, spectrum, N1, N2):
    """Aggregate KL order at the best global churn vs the per-eigenmode churn, two step counts."""
    per = [float(lambda_star_vp(B, s2, T)) for s2 in spectrum]
    # best single global churn: grid-min aggregate KL at N1
    grid = np.linspace(0.5 * min(per), 1.3 * max(per), 60)
    def agg(lams, N):
        vs = [miscalibrated_recursion(N, T, B, lams[i], spectrum[i], 0) for i in range(len(spectrum))]
        return float(kl_gauss_aniso(vs, spectrum))
    kls_grid = [agg([g] * len(spectrum), N1) for g in grid]
    g_best = float(grid[int(np.argmin(kls_grid))])
    # order under global-best churn
    klg1, klg2 = agg([g_best] * len(spectrum), N1), agg([g_best] * len(spectrum), N2)
    ord_global = local_order(klg1, klg2, N1, N2)
    # order under per-eigenmode churn
    klp1, klp2 = agg(per, N1), agg(per, N2)
    ord_permode = local_order(klp1, klp2, N1, N2)
    return g_best, ord_global, ord_permode


def study_B(B, T, dps):
    """Per-eigenmode fix restoring order 4 across dimensions."""
    set_dps(dps)
    dims = [2, 4, 8, 16, 32, 64]
    N1, N2 = 256, 1024
    rng = np.random.default_rng(0)
    out = []
    for d in dims:
        og, op = [], []
        for rep in range(3):                       # average over 3 random spectra
            spec = sorted(np.exp(rng.uniform(np.log(1.3), np.log(8.0), size=d)).tolist())
            _, o_g, o_p = _aniso_orders(B, T, spec, N1, N2)
            og.append(o_g); op.append(o_p)
        out.append({"d": d, "order_global_mean": float(np.mean(og)), "order_global_std": float(np.std(og)),
                    "order_permode_mean": float(np.mean(op)), "order_permode_std": float(np.std(op))})
        io.log(f"  E70-B d={d:3d}: global-churn order={np.mean(og):.3f}+/-{np.std(og):.3f} (pred 2)  "
               f"per-eigenmode order={np.mean(op):.3f}+/-{np.std(op):.3f} (pred 4)")
    return {"dims": dims, "N1": N1, "N2": N2, "n_spectra": 3, "rows": out}


def run(dps=34, B=4.0, T=5.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    t0 = time.time()
    io.log(f"E70 strengthening mega-suite starting (dps={dps})")
    A = study_A(B, T, dps)
    B_ = study_B(B, T, dps)
    io.save(NAME, {"config": {"dps": dps, "B": B, "T": T}, "study_A_scales": A, "study_B_aniso_dim": B_})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
