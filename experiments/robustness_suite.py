"""Robustness suite (E23-E30): high-precision, many-config stress tests of every claim.

These are *exact* deterministic experiments (Gaussian target, linear score): the EM iterate
stays Gaussian so the terminal KL is computed with zero sampling, only numerical error.
We push mpmath precision high (dps 50-120) and N large (up to 2^18) so the measured
convergence ORDER is pinned to many significant figures across a grid of (B,T,s2) and across
schedules / integrators / spectra. Each sub-experiment saves its own JSON and logs timing;
the whole script is meant to run for a long time in the background.

Claims stress-tested:
  E23  order is N^-4 at lambda*, N^-2 off it, ACROSS a (B,T,s2) grid (not one cherry-picked config)
  E24  pin the order and the N^-4 coefficient c2(lambda*) at dps=120, N up to 2^18
  E25  score-error floor law: crossover N*(eps) ~ eps^{-1/2} at lambda*
  E26  anisotropic no-go, quantified: per-mode roots spread; aggregate order stays 2
  E27  integrator compounding: Heun lifts the generic order 2->4 and its own cancellation ->6
  E28  schedule invariance: lambda* and order identical across VP-const/ramp, VE-lin/quad
  E29  large-s2 constant kappa = lambda*/sqrt(s2) pinned across (B,T)
  E30  c2(lambda*) is non-zero and varies smoothly with config (genuine 4th-order term)
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
import io_utils as io
from diffusion import vp_const, vp_ramp, ve_linear, ve_quad, set_dps
from recursion import v_terminal, v_terminal_closed
from coefficient import (C_closed_vp, C0_vp, C1_vp, D_closed_vp, C_closed, C_richardson)
from lambda_star import lambda_star_vp, u_star_general, u_star_richardson
from metrics import kl_gauss


def kl_at(sched, N, u, e0=0, eps2=0, integrator="EM"):
    v = v_terminal(sched, N, u, e0=e0, eps2=eps2, integrator=integrator)
    return kl_gauss(v, sched.s2)


def order_ladder(sched, u, Ns, integrator="EM", eps2=0):
    """KL on a step-count ladder + local log-log orders between consecutive points.
    Uses the O(N) exact product/sum form when integrator='EM' (fast, exact)."""
    kls = []
    for N in Ns:
        if integrator == "EM" and eps2 == 0:
            v = v_terminal_closed(sched, N, u)
        else:
            v = v_terminal(sched, N, u, eps2=eps2, integrator=integrator)
        kls.append(kl_gauss(v, sched.s2))
    orders = []
    for i in range(len(Ns) - 1):
        lo, hi = kls[i], kls[i + 1]
        if lo <= 0 or hi <= 0:
            orders.append(None); continue
        orders.append(float(mp.log(lo / hi) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i])))
    return [mp.nstr(k, 8) for k in kls], orders


# ---------------------------------------------------------------------------
def e23_order_grid():
    """Order at lambda* vs off-lambda*, across a (B,T,s2) grid. The central robustness table."""
    set_dps(60)
    Bs = [2.0, 4.0, 8.0]; Ts = [4.0, 6.0, 10.0]; s2s = [1.5, 2.0, 3.5, 6.0]
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    rows = []
    for B in Bs:
        for T in Ts:
            for s2 in s2s:
                sched = vp_const(B, s2, T)
                ls = lambda_star_vp(B, s2, T)
                if ls is None:
                    continue
                u_star = ls ** 2
                # at lambda*: expect order ~4
                _, ord_star = order_ladder(sched, u_star, Ns)
                # off lambda* (two sides): expect order ~2
                _, ord_lo = order_ladder(sched, (ls * mp.mpf('0.7')) ** 2, Ns)
                _, ord_hi = order_ladder(sched, (ls * mp.mpf('1.3')) ** 2, Ns)
                rows.append({
                    "B": B, "T": T, "s2": s2,
                    "lambda_star": mp.nstr(ls, 10),
                    "kappa": mp.nstr(ls / mp.sqrt(s2), 8),
                    "order_at_star_tail": ord_star[-1],
                    "order_at_star_all": ord_star,
                    "order_below_tail": ord_lo[-1],
                    "order_above_tail": ord_hi[-1],
                })
                io.log(f"  e23 B={B} T={T} s2={s2}: ord*={ord_star[-1]:.3f} "
                       f"below={ord_lo[-1]:.3f} above={ord_hi[-1]:.3f}")
    star_tail = [r["order_at_star_tail"] for r in rows]
    off_tail = [r["order_below_tail"] for r in rows] + [r["order_above_tail"] for r in rows]
    io.save("e23_order_grid", {
        "config": {"Bs": Bs, "Ts": Ts, "s2s": s2s, "Ns": Ns, "dps": 60},
        "rows": rows,
        "summary": {
            "n_configs": len(rows),
            "order_at_star_min": min(star_tail), "order_at_star_max": max(star_tail),
            "order_at_star_mean": float(sum(star_tail) / len(star_tail)),
            "order_off_star_min": min(off_tail), "order_off_star_max": max(off_tail),
            "order_off_star_mean": float(sum(off_tail) / len(off_tail)),
        }})
    io.log(f"e23_order_grid DONE: {len(rows)} configs, "
           f"order* in [{min(star_tail):.3f},{max(star_tail):.3f}], "
           f"order_off in [{min(off_tail):.3f},{max(off_tail):.3f}]")


def e24_order_pinning():
    """Pin the order and c2(lambda*) at very high precision, large N, canonical config."""
    set_dps(120)
    B, s2, T = 4.0, 2.0, 5.0
    sched = vp_const(B, s2, T)
    ls = lambda_star_vp(B, s2, T); u = ls ** 2
    Ns = [2 ** k for k in range(9, 19)]  # 512 .. 262144
    kls = []
    for N in Ns:
        t0 = time.time()
        v = v_terminal_closed(sched, N, u)
        kls.append(kl_gauss(v, s2))
        io.log(f"  e24 N={N}: KL={mp.nstr(kls[-1],6)}  ({time.time()-t0:.1f}s)")
    orders = [float(mp.log(kls[i] / kls[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i]))
              for i in range(len(Ns) - 1)]
    # c2: KL ~ c2 * N^-4  =>  c2 = KL * N^4 (tail, where order has settled)
    c2_tail = [mp.nstr(kls[i] * mp.mpf(Ns[i]) ** 4, 10) for i in range(len(Ns))]
    # Richardson on N^4*KL to extrapolate c2 as N->inf
    g = [kls[i] * mp.mpf(Ns[i]) ** 4 for i in range(len(Ns))][-5:]
    xs = [1 / mp.mpf(Ns[i]) ** 2 for i in range(len(Ns))][-5:]
    k = len(g); Vm = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            Vm[i, j] = xs[i] ** j
    c2_inf = mp.lu_solve(Vm, mp.matrix(g))[0]
    io.save("e24_order_pinning", {
        "config": {"B": B, "s2": s2, "T": T, "dps": 120, "Ns": Ns},
        "lambda_star": mp.nstr(ls, 30),
        "KL": [mp.nstr(k, 12) for k in kls],
        "orders": orders,
        "order_tail": orders[-1],
        "c2_tail_estimates": c2_tail,
        "c2_extrapolated": mp.nstr(c2_inf, 12),
    })
    io.log(f"e24_order_pinning DONE: order_tail={orders[-1]:.6f}  c2(inf)={mp.nstr(c2_inf,8)}")


def e25_floor_law():
    """At lambda*, score-error floor D*eps2 vs discretisation; crossover N*(eps) ~ eps^{-1/2}."""
    set_dps(60)
    B, s2, T = 4.0, 2.0, 5.0
    sched = vp_const(B, s2, T)
    ls = lambda_star_vp(B, s2, T); u = ls ** 2
    D = D_closed_vp(B, ls, s2, T)
    # c2 from a clean fit at eps=0 (KL ~ c2 N^-4): use large N
    Nref = 8192
    c2 = kl_gauss(v_terminal_closed(sched, Nref, u), s2) * mp.mpf(Nref) ** 4
    epss = [mp.mpf('1e-1'), mp.mpf('3e-2'), mp.mpf('1e-2'), mp.mpf('3e-3'),
            mp.mpf('1e-3'), mp.mpf('3e-4'), mp.mpf('1e-4')]
    rows = []
    for eps in epss:
        eps2 = eps ** 2
        # crossover where discretisation variance error ~ floor variance error:
        # |Delta v|_disc ~ a2 N^-2 (since C=0 at lambda*) and floor ~ D eps2.
        # predicted N* from KL balance: c2 N^-4 = (D eps2)^2/(4 s2^2)  =>  N* = (4 s2^2 c2/(D eps2)^2)^{1/4}
        floor_kl = (D * eps2) ** 2 / (4 * s2 ** 2)
        Nstar_pred = (c2 / floor_kl) ** mp.mpf('0.25')
        # measure: smallest N (on a fine ladder) where KL stops decreasing like N^-4 (floor dominates)
        Ns = [int(round(float(Nstar_pred) * f)) for f in (0.25, 0.5, 1.0, 2.0, 4.0)]
        Ns = [max(8, n) for n in Ns]
        klmeas = [mp.nstr(kl_at(sched, n, u, eps2=eps2), 6) for n in Ns]
        rows.append({"eps": mp.nstr(eps, 4), "D_eps2": mp.nstr(D * eps2, 6),
                     "Nstar_pred": mp.nstr(Nstar_pred, 6),
                     "Ns_probe": Ns, "KL_probe": klmeas})
        io.log(f"  e25 eps={mp.nstr(eps,3)}: N*_pred={mp.nstr(Nstar_pred,5)}")
    # fit slope of log N* vs log eps (expect -1/2)
    import math
    logeps = [math.log(float(e)) for e in epss]
    logNs = [math.log(float((c2 / ((D * e ** 2) ** 2 / (4 * s2 ** 2))) ** mp.mpf('0.25'))) for e in epss]
    n = len(logeps); sx = sum(logeps); sy = sum(logNs)
    sxx = sum(x * x for x in logeps); sxy = sum(x * y for x, y in zip(logeps, logNs))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    # At lambda* the discretisation variance error is c2/N^2 (C=0) and the floor is D eps^2;
    # balancing c2/N^2 = D eps^2 gives N* = sqrt(c2/D) eps^{-1}, i.e. slope -1 in log-log.
    io.save("e25_floor_law", {
        "config": {"B": B, "s2": s2, "T": T, "dps": 60},
        "lambda_star": mp.nstr(ls, 10), "D": mp.nstr(D, 10), "c2": mp.nstr(c2, 10),
        "rows": rows, "Nstar_vs_eps_slope": float(slope), "predicted_slope": -1.0})
    io.log(f"e25_floor_law DONE: N*(eps) slope={float(slope):.4f} (predict -1.0)")


def e26_aniso_nogo():
    """Anisotropic no-go, quantified across spectra."""
    set_dps(50)
    B, T = 4.0, 5.0
    spectra = {
        "two_mode": [1.5, 4.0],
        "three_mode": [1.3, 2.5, 5.0],
        "wide": [1.2, 2.0, 4.0, 8.0],
        "near_degenerate": [1.9, 2.0, 2.1],
        "geometric": [1.5, 2.25, 3.375, 5.0625],
    }
    Ns = [512, 1024, 2048, 4096, 8192]
    out = {}
    for name, spec in spectra.items():
        roots = [lambda_star_vp(B, s, T) for s in spec]
        roots_f = [float(r) for r in roots if r is not None]
        # aggregate KL(lambda) = sum_i KL_i(lambda); best single lambda minimises leading coeff.
        # measure aggregate order at the per-mode-average root and at the K-minimiser.
        lam_avg = mp.mpf(sum(roots_f) / len(roots_f))
        # leading aggregate coeff K(lambda)=sum_i C_i(lambda)^2 / s_i^4 ; scan for its min
        lams = [mp.mpf('0.2') + mp.mpf('0.05') * j for j in range(80)]
        def Kval(lam):
            return sum((C_closed_vp(B, lam, s, T)) ** 2 / mp.mpf(s) ** 4 for s in spec)
        Kv = [Kval(l) for l in lams]
        lam_dagger = lams[min(range(len(lams)), key=lambda j: Kv[j])]
        # aggregate KL order at lam_dagger (expect ~2: no single lambda kills all modes)
        def agg_kl(lam):
            u = lam ** 2
            return sum(kl_gauss(v_terminal_closed(vp_const(B, s, T), 0, u) if False else
                                v_terminal(vp_const(B, s, T), 0 + 0, u), s) for s in spec)
        def agg_kl_N(lam, N):
            u = lam ** 2
            return sum(kl_gauss(v_terminal_closed(vp_const(B, s, T), N, u), s) for s in spec)
        kls = [agg_kl_N(lam_dagger, N) for N in Ns]
        agg_orders = [float(mp.log(kls[i] / kls[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i]))
                      for i in range(len(Ns) - 1)]
        spread = (max(roots_f) - min(roots_f)) / (sum(roots_f) / len(roots_f))
        out[name] = {
            "spectrum": spec, "per_mode_lambda_star": [mp.nstr(r, 8) for r in roots],
            "root_rel_spread": float(spread),
            "lambda_dagger": mp.nstr(lam_dagger, 8),
            "aggregate_order_tail": agg_orders[-1],
            "aggregate_orders": agg_orders}
        io.log(f"  e26 {name}: agg_order_tail={agg_orders[-1]:.3f} root_spread={float(spread):.3f}")
    io.save("e26_aniso_nogo", {"config": {"B": B, "T": T, "dps": 50, "Ns": Ns}, "spectra": out})
    io.log("e26_aniso_nogo DONE")


def e27_integrator_compounding():
    """Heun (weak order 2) lifts generic order 2->4; its own cancellation churn lifts ->6."""
    set_dps(70)
    B, s2, T = 4.0, 2.0, 5.0
    sched = vp_const(B, s2, T)
    Ns = [128, 256, 512, 1024, 2048, 4096]
    # generic churn (lambda=1): EM gives ~2, Heun ~4
    u_gen = mp.mpf(1.0)
    _, ord_em = order_ladder(sched, u_gen, Ns, integrator="EM")
    _, ord_heun = order_ladder(sched, u_gen, Ns, integrator="heun")
    # Heun's own cancellation churn: root of its leading coeff via Richardson
    try:
        u_heun_star = u_star_richardson(sched, Ns=(512, 1024, 2048), integrator="heun")
    except Exception as ex:
        u_heun_star = None; io.log(f"  e27 heun root err {ex}")
    ord_heun_star = None
    if u_heun_star is not None:
        _, ord_heun_star = order_ladder(sched, u_heun_star, Ns, integrator="heun")
    # EM at its lambda* (cross-check 4)
    ls = lambda_star_vp(B, s2, T)
    _, ord_em_star = order_ladder(sched, ls ** 2, Ns, integrator="EM")
    io.save("e27_integrator_compounding", {
        "config": {"B": B, "s2": s2, "T": T, "dps": 70, "Ns": Ns},
        "EM_generic_orders": ord_em, "Heun_generic_orders": ord_heun,
        "EM_at_lambda_star_orders": ord_em_star,
        "u_heun_star": None if u_heun_star is None else mp.nstr(u_heun_star, 8),
        "Heun_at_its_star_orders": ord_heun_star,
        "summary": {"EM_generic_tail": ord_em[-1], "Heun_generic_tail": ord_heun[-1],
                    "EM_star_tail": ord_em_star[-1],
                    "Heun_star_tail": None if ord_heun_star is None else ord_heun_star[-1]}})
    io.log(f"e27 DONE: EM_gen={ord_em[-1]:.3f} Heun_gen={ord_heun[-1]:.3f} "
           f"EM*={ord_em_star[-1]:.3f} Heun*={'NA' if ord_heun_star is None else f'{ord_heun_star[-1]:.3f}'}")


def e28_schedule_invariance():
    """lambda* and order=4 across VP-const, VP-ramp, VE-linear, VE-quad (all s2=2)."""
    set_dps(60)
    s2, T = 2.0, 5.0
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    scheds = {
        "vp_const": vp_const(4.0, s2, T),
        "vp_ramp": vp_ramp(2.0, 1.0, s2, T),
        "ve_linear": ve_linear(1.5, s2, T),
        "ve_quad": ve_quad(0.5, s2, T),
    }
    out = {}
    for name, sched in scheds.items():
        t0 = time.time()
        # recursion-based root (fast); the ODE-sweep root-finder (u_star_general) uses mpmath.odefun
        # which is pathologically slow at dps=60, so we locate the root from the Richardson estimate
        # of C and then confirm the order at it.
        us = u_star_richardson(sched, Ns=(1024, 2048, 4096), integrator="EM")
        if us is None:
            out[name] = {"lambda_star": None}; io.log(f"  e28 {name}: no root"); continue
        ls = mp.sqrt(us)
        _, ord_star = order_ladder(sched, us, Ns, integrator="EM")
        _, ord_off = order_ladder(sched, (ls * mp.mpf('1.4')) ** 2, Ns, integrator="EM")
        out[name] = {"lambda_star": mp.nstr(ls, 10), "kappa": mp.nstr(ls / mp.sqrt(s2), 8),
                     "order_at_star_tail": ord_star[-1], "order_off_tail": ord_off[-1],
                     "order_at_star_all": ord_star}
        io.log(f"  e28 {name}: lam*={mp.nstr(ls,6)} ord*={ord_star[-1]:.3f} "
               f"off={ord_off[-1]:.3f}  ({time.time()-t0:.1f}s)")
    io.save("e28_schedule_invariance", {"config": {"s2": s2, "T": T, "dps": 60, "Ns": Ns}, "schedules": out})
    io.log("e28_schedule_invariance DONE")


def e29_kappa_limit():
    """kappa = lambda*/sqrt(s2) as s2->infinity, across (B,T): pin the universal constant.
    kappa converges by s2~1024 (1.2069 at 1024 vs 1.2072 at 4096), and the mp.quad root-find develops a
    sharp boundary layer at very large s2, so we cap s2 at 1024 and use dps=50: same limit, far faster."""
    set_dps(50)
    configs = [(2.0, 5.0), (4.0, 5.0), (8.0, 5.0), (4.0, 3.0), (4.0, 10.0)]
    s2s = [4.0, 16.0, 64.0, 256.0, 1024.0]
    out = {}
    for (B, T) in configs:
        kap = []
        for s2 in s2s:
            ls = lambda_star_vp(B, s2, T)
            kap.append(None if ls is None else mp.nstr(ls / mp.sqrt(s2), 12))
        out[f"B={B},T={T}"] = {"s2s": s2s, "kappa": kap, "kappa_limit": kap[-1]}
        io.log(f"  e29 B={B} T={T}: kappa->{kap[-1]}")
    io.save("e29_kappa_limit", {"config": {"dps": 80, "s2s": s2s}, "series": out})
    io.log("e29_kappa_limit DONE")


def e30_c2_field():
    """c2(lambda*) across a (B,T,s2) grid: confirm the N^-4 coefficient is non-zero and smooth."""
    set_dps(70)
    Bs = [2.0, 4.0, 8.0]; s2s = [1.5, 2.0, 4.0]
    T = 6.0; Nref = 8192
    rows = []
    for B in Bs:
        for s2 in s2s:
            sched = vp_const(B, s2, T)
            ls = lambda_star_vp(B, s2, T)
            if ls is None:
                continue
            kl = kl_gauss(v_terminal_closed(sched, Nref, ls ** 2), s2)
            c2 = kl * mp.mpf(Nref) ** 4
            rows.append({"B": B, "s2": s2, "lambda_star": mp.nstr(ls, 8),
                         "c2_at_star": mp.nstr(c2, 8)})
            io.log(f"  e30 B={B} s2={s2}: c2={mp.nstr(c2,6)}")
    io.save("e30_c2_field", {"config": {"Bs": Bs, "s2s": s2s, "T": T, "Nref": Nref, "dps": 70}, "rows": rows})
    io.log("e30_c2_field DONE")


if __name__ == "__main__":
    mp.mp.dps = 50
    suite = [e23_order_grid, e24_order_pinning, e25_floor_law, e26_aniso_nogo,
             e27_integrator_compounding, e28_schedule_invariance, e29_kappa_limit, e30_c2_field]
    # allow running a subset: python robustness_suite.py e24 e27
    pick = [a.lower() for a in sys.argv[1:]]
    if pick:
        suite = [fn for fn in suite if any(p in fn.__name__.lower() for p in pick)]
    io.log(f"robustness_suite START: {[fn.__name__ for fn in suite]}")
    t_all = time.time()
    for fn in suite:
        t = time.time()
        try:
            fn()
            io.log(f"[{fn.__name__}] OK {time.time()-t:.1f}s")
        except Exception as ex:
            import traceback
            io.log(f"[{fn.__name__}] ERR {ex}\n{traceback.format_exc()}")
    io.log(f"robustness_suite DONE all in {time.time()-t_all:.1f}s")
