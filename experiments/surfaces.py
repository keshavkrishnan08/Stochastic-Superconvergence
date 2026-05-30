"""Grid experiments that generate 2D-field data for beautiful 3D surfaces / contour fields /
phase diagrams (the figure gallery). All deterministic exact recursions (mpmath).

  E17 kl_canyon       : log10 KL(lambda, N)            -> 3D canyon + contour (the signature fig)
  E18 coeff_field     : C(lambda, s2)                  -> diverging contour w/ zero-locus lambda*(s2)
  E19 order_field     : measured order p(lambda, N)    -> heatmap, the resonance ridge
  E20 goldilocks_surf : KL(lambda, eps) at fixed N     -> 3D surface, valley moves with eps
  E21 phase_diagram   : regime over (s2, lambda)       -> sign(C): over-contract/ridge/over-churn
  E22 aniso_spaghetti : per-mode C_i(lambda) + K(lambda) for a spectrum -> no-go visual
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
from coefficient import C_closed_vp
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order
import io_utils as io


def e17_kl_canyon(s2=2.0, B=4.0, T=5.0, dps=60, nlam=90):
    NAME = "e17_kl_canyon"
    if io.exists(NAME): io.log(f"{NAME} skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T); lstar = float(lambda_star_vp(B, s2, T))
    lams = np.linspace(0.0, 2.4, nlam)
    Ns = [2 ** k for k in range(4, 14)]   # 16..8192
    Z = []   # log10 KL[lam_i][N_j]
    for lam in lams:
        row = []
        for N in Ns:
            kl = float(kl_gauss(v_terminal(sched, N, mp.mpf(lam) ** 2), s2))
            row.append(float(mp.log10(max(kl, mp.mpf('1e-30')))))
        Z.append(row)
    res = {"config": {"s2": s2, "B": B, "T": T}, "lambda_star": lstar,
           "lams": lams.tolist(), "Ns": Ns, "logKL": Z}
    io.save(NAME, res); io.log(f"{NAME}: {nlam}x{len(Ns)} canyon grid done (lam*={lstar:.4f})")
    return res


def e18_coeff_field(B=4.0, T=5.0, dps=30, nlam=60, ns2=55):
    NAME = "e18_coeff_field"
    if io.exists(NAME): io.log(f"{NAME} skip"); return io.load(NAME)
    set_dps(dps)
    lams = np.linspace(0.0, 2.6, nlam)
    s2s = np.linspace(1.05, 8.0, ns2)
    C = []        # C[s2_i][lam_j]
    lstar_curve = []
    for s2 in s2s:
        row = [float(C_closed_vp(B, lam, float(s2), T)) for lam in lams]
        C.append(row)
        ls = lambda_star_vp(B, float(s2), T)
        lstar_curve.append(float(ls) if ls else None)
    res = {"config": {"B": B, "T": T}, "lams": lams.tolist(), "s2s": s2s.tolist(),
           "C": C, "lambda_star_curve": lstar_curve}
    io.save(NAME, res); io.log(f"{NAME}: {ns2}x{nlam} C(lam,s2) field + zero-locus done")
    return res


def e19_order_field(s2=2.0, B=4.0, T=5.0, dps=60, nlam=55):
    NAME = "e19_order_field"
    if io.exists(NAME): io.log(f"{NAME} skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T); lstar = float(lambda_star_vp(B, s2, T))
    lams = np.linspace(0.4, 2.2, nlam)
    Npairs = [(64, 256), (256, 1024), (1024, 4096)]
    P = []   # P[lam_i][pair_j] = measured order
    for lam in lams:
        row = []
        for (n1, n2) in Npairs:
            k1 = kl_gauss(v_terminal(sched, n1, mp.mpf(lam) ** 2), s2)
            k2 = kl_gauss(v_terminal(sched, n2, mp.mpf(lam) ** 2), s2)
            row.append(local_order(k1, k2, n1, n2))
        P.append(row)
    res = {"config": {"s2": s2, "B": B, "T": T}, "lambda_star": lstar,
           "lams": lams.tolist(), "Npairs": [list(p) for p in Npairs], "order": P}
    io.save(NAME, res); io.log(f"{NAME}: order field {nlam}x{len(Npairs)} done")
    return res


def e20_goldilocks_surf(s2=2.0, B=4.0, T=5.0, N=512, dps=40, nlam=55, neps=32):
    NAME = "e20_goldilocks_surf"
    if io.exists(NAME): io.log(f"{NAME} skip"); return io.load(NAME)
    set_dps(dps)
    sched = vp_const(B, s2, T); lstar = float(lambda_star_vp(B, s2, T))
    lams = np.linspace(0.0, 2.0, nlam)
    epss = np.linspace(0.0, 0.3, neps)
    Z = []   # log10 KL[eps_i][lam_j]
    valley = []
    for eps in epss:
        row = [float(mp.log10(max(kl_gauss(v_terminal(sched, N, mp.mpf(l) ** 2, eps2=float(eps) ** 2), s2), mp.mpf('1e-30')))) for l in lams]
        Z.append(row); valley.append(float(lams[int(np.argmin(row))]))
    res = {"config": {"s2": s2, "B": B, "T": T, "N": N}, "lambda_star": lstar,
           "lams": lams.tolist(), "epss": epss.tolist(), "logKL": Z, "valley_lambda": valley}
    io.save(NAME, res); io.log(f"{NAME}: goldilocks surface {neps}x{nlam} + valley track done")
    return res


def e21_phase_diagram(B=4.0, T=5.0, dps=25, ns2=70, nlam=70):
    NAME = "e21_phase_diagram"
    if io.exists(NAME): io.log(f"{NAME} skip"); return io.load(NAME)
    set_dps(dps)
    s2s = np.linspace(0.4, 8.0, ns2)
    lams = np.linspace(0.0, 3.0, nlam)
    # signed C: <0 over-contracted (ODE-like deficit), >0 over-churned (excess); 0 = ridge
    S = []
    lstar_curve = []
    for s2 in s2s:
        row = [float(mp.sign(C_closed_vp(B, lam, float(s2), T))) for lam in lams]
        S.append(row)
        ls = lambda_star_vp(B, float(s2), T)
        lstar_curve.append(float(ls) if ls else None)
    res = {"config": {"B": B, "T": T}, "s2s": s2s.tolist(), "lams": lams.tolist(),
           "signC": S, "lambda_star_curve": lstar_curve}
    io.save(NAME, res); io.log(f"{NAME}: phase diagram {ns2}x{nlam} done")
    return res


def e22_aniso_spaghetti(B=4.0, T=5.0, dps=35, nlam=90):
    NAME = "e22_aniso_spaghetti"
    if io.exists(NAME): io.log(f"{NAME} skip"); return io.load(NAME)
    set_dps(dps)
    spectrum = [1.3, 2.0, 3.2, 5.0]
    lams = np.linspace(0.0, 3.0, nlam)
    Ci = {f"{s}": [float(C_closed_vp(B, lam, s, T)) for lam in lams] for s in spectrum}
    per_root = [None if (l := lambda_star_vp(B, s, T)) is None else float(l) for s in spectrum]
    # global K(lambda) = sum C_i^2 / s_i^4
    K = [float(sum((C_closed_vp(B, lam, s, T)) ** 2 / mp.mpf(s) ** 4 for s in spectrum)) for lam in lams]
    dagger = float(lams[int(np.argmin(K))])
    res = {"config": {"B": B, "T": T}, "spectrum": spectrum, "lams": lams.tolist(),
           "C_i": Ci, "per_mode_root": per_root, "K": K, "lambda_dagger": dagger}
    io.save(NAME, res); io.log(f"{NAME}: spaghetti (roots {[round(r,3) if r else None for r in per_root]}, dagger {dagger:.3f})")
    return res


ALL = [e17_kl_canyon, e18_coeff_field, e19_order_field, e20_goldilocks_surf, e21_phase_diagram, e22_aniso_spaghetti]

if __name__ == "__main__":
    import time
    for fn in ALL:
        t = time.time()
        try: fn(); io.log(f"[{fn.__name__}] OK {time.time()-t:.1f}s")
        except Exception as e:
            import traceback; io.log(f"[{fn.__name__}] ERROR {e}\n{traceback.format_exc()}")
    io.log("surfaces DONE")
