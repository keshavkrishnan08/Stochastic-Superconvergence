"""Wave-4 suite: two exact ablations that bulletproof the EDM bridge (E35) for reviewers.

E47  EDM hyperparameter robustness. The bridge in E35 used one schedule (rho=7, sigma_max=80).
     Here we re-run the literal EDM sampler's exact Gaussian variance propagation across a grid of
     (rho, sigma_max) and confirm that (a) the churn still has an interior optimum and (b) the
     grid-search optimum still coincides with the closed-form root of v_N=s^2 to within one grid step.
     If it holds across the schedule grid, the bridge is not an artefact of one hyperparameter choice.

E49  The bridge under a miscalibrated score. E35 used the exact denoiser; trained scores are not exact.
     We scale the linear Gaussian denoiser by (1+delta) -- a controllable, N-independent miscalibration --
     and re-run the exact EDM propagation. The prediction (Theorem 'goldilocks') is that the interior
     optimum survives, the terminal KL flattens onto a delta-dependent floor, and the optimal churn
     shifts off the zero-floor root as the floor grows. This connects the exact bridge (E35) to the
     trained-score regime (E36/E39) on the literal sampler, in closed form.

Both are exact (Gaussian linear denoiser keeps the iterate Gaussian); no Monte-Carlo, no training.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from metrics import kl_gauss
import io_utils as io
from e35_edm_bridge import edm_sigmas, SQRT2_MINUS_1


def edm_terminal_var_delta(s2, N, S_churn, sigma_min, sigma_max, rho=7, delta=0.0):
    """Exact terminal variance of the EDM Alg.-2 sampler with denoiser scaled by (1+delta).

    Denoiser D(x;sigma)=a(sigma)(1+delta) x, a=s2/(s2+sigma^2). The ODE slope is
    d(sigma)=(1-a(1+delta))/sigma = (sigma^2 - s2*delta)/((s2+sigma^2) sigma); delta=0 is the exact case.
    Iterate stays Gaussian, so variance propagates exactly (Heun predictor/corrector, S_churn injection)."""
    s2 = mp.mpf(s2); dl = mp.mpf(delta)
    sig = edm_sigmas(N, sigma_min, sigma_max, rho)
    v = mp.mpf(sigma_max) ** 2
    Schurn = mp.mpf(S_churn); gcap = SQRT2_MINUS_1

    def dco(sg):                                   # slope coefficient d(sigma)/x
        return (sg ** 2 - s2 * dl) / ((s2 + sg ** 2) * sg)

    for i in range(N):
        si = sig[i]; si1 = sig[i + 1]
        gamma = min(Schurn / N, gcap)
        shat = si * (1 + gamma)
        vhat = v + (shat ** 2 - si ** 2)
        d1 = dco(shat)
        m1 = 1 + (si1 - shat) * d1
        if si1 == 0:
            M = m1
        else:
            d2 = dco(si1) * m1
            M = 1 + (si1 - shat) * (d1 / 2 + d2 / 2)
        v = M ** 2 * vhat
    return v


def _bisect(f, lo, hi, maxit=70):
    lo = mp.mpf(lo); hi = mp.mpf(hi); flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(maxit):
        mid = (lo + hi) / 2; fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def e47_edm_hyper_robust(dps=30, s2=4.0, N_search=1024, sigma_min=0.002):
    NAME = "e47_edm_hyper_robust"
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps); t0 = time.time()
    step = mp.mpf(1) / 2
    grid = [step * k for k in range(0, 41)]                 # S_churn in [0,20] step 0.5
    gstep = float(step)
    rows = []; n_ok = 0
    for rho in (3, 5, 7, 9):
        for sM in (40.0, 80.0, 160.0):
            kls = [kl_gauss(edm_terminal_var_delta(s2, N_search, sc, sigma_min, sM, rho), s2) for sc in grid]
            kmin = min(kls); imin = kls.index(kmin)
            interior = bool(0 < imin < len(grid) - 1)
            f = lambda sc: edm_terminal_var_delta(s2, N_search, sc, sigma_min, sM, rho) - mp.mpf(s2)
            root = _bisect(f, grid[0], grid[-1])
            gap = None if root is None else float(abs(root - grid[imin]))
            within = bool(gap is not None and gap <= gstep)
            n_ok += int(interior and within)
            rows.append({"rho": rho, "sigma_max": sM, "interior_optimum": interior,
                         "grid_opt_S_churn": float(grid[imin]),
                         "predicted_S_churn": None if root is None else float(root),
                         "abs_gap": gap, "within_one_grid_step": within,
                         "KL_at0": float(kls[0]), "KL_at_opt": float(kmin)})
            io.log(f"  e47 rho={rho} sMax={sM}: interior={interior} grid*={float(grid[imin]):.2f} "
                   f"pred*={'NA' if root is None else round(float(root),3)} within1step={within}")
    io.save(NAME, {"config": {"dps": dps, "s2": s2, "N_search": N_search, "grid_step": gstep,
                              "rho_set": [3, 5, 7, 9], "sigma_max_set": [40.0, 80.0, 160.0]},
                   "n_configs": len(rows), "n_interior_and_within": n_ok, "rows": rows})
    io.log(f"{NAME} DONE: {n_ok}/{len(rows)} configs interior+within-one-step ({time.time()-t0:.0f}s)")


def e49_edm_miscal(dps=30, s2=4.0, N_search=2048, sigma_min=0.002, sigma_max=80.0, rho=7):
    """Miscalibration is SIGN-asymmetric: churn only adds variance, so it can cancel an under-shooting
    score error (delta<0 here) but never an over-shooting one (delta>0). We sweep both signs and store
    the full KL-vs-churn curve at each delta. For delta<0 the curve keeps a deep interior dip (churn
    restores v_N=s^2 exactly for a Gaussian); for delta>0 it rises monotonically off a delta^2 floor."""
    NAME = "e49_edm_miscal"
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    set_dps(dps); t0 = time.time()
    step = mp.mpf(1) / 4
    grid = [step * k for k in range(0, 81)]                 # S_churn in [0,20] step 0.25
    gridf = [float(x) for x in grid]
    deltas = (-1e-1, -1e-2, -1e-3, 0.0, 1e-3, 1e-2, 1e-1)
    rows = []
    root0 = None
    for delta in deltas:
        kls = [kl_gauss(edm_terminal_var_delta(s2, N_search, sc, sigma_min, sigma_max, rho, delta), s2)
               for sc in grid]
        klsf = [float(x) for x in kls]
        kmin = min(kls); imin = kls.index(kmin)
        interior = bool(0 < imin < len(grid) - 1)
        sc_opt = float(grid[imin])
        if delta == 0.0:
            root0 = sc_opt
        rows.append({"delta": delta, "interior_optimum": interior, "opt_S_churn": sc_opt,
                     "KL_floor_at_opt": float(kmin), "KL_at_churn0": float(kls[0]),
                     "shift_from_exact": None if root0 is None else float(sc_opt - root0),
                     "KL_curve": klsf})
        io.log(f"  e49 delta={delta:+.0e}: interior={interior} opt*={sc_opt:.2f} "
               f"KLmin={float(kmin):.2e}")
    io.save(NAME, {"config": {"dps": dps, "s2": s2, "N_search": N_search, "sigma_max": sigma_max,
                              "rho": rho, "deltas": list(deltas), "churn_grid": gridf,
                              "miscalibration": "denoiser scaled by (1+delta), N-independent"},
                   "rows": rows})
    io.log(f"{NAME} DONE ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    e47_edm_hyper_robust()
    e49_edm_miscal()
    print("wave4 done")
