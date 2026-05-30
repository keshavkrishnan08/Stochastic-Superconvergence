"""Super-convergence churn lambda* = unique positive root of C(lambda)=0 (Thm A),
and its dependence on (s2, B, T).  Parametrise in u = lambda^2 (>=0); robust bisection.
"""
from __future__ import annotations
import mpmath as mp
from coefficient import C_closed_vp, C_closed


def _bisect(f, lo, hi, maxit=200, tol=None):
    lo = mp.mpf(lo); hi = mp.mpf(hi)
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(maxit):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(hi - lo) < tol:
            break
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def u_star_vp(B, s2, T, hi0=mp.mpf('0.05')):
    """Root u*>0 of C(u)=0 for VP-const. Returns None when there is no POSITIVE root,
    which (Thm A) is exactly when C(0)>=0, i.e. s2<=1 (the ODE does not over-contract)."""
    f = lambda u: C_closed_vp(B, mp.sqrt(u), s2, T)
    C0 = f(mp.mpf(0))
    if C0 >= -mp.mpf('1e-12'):   # no over-contraction -> no positive cancellation root
        return None
    lo, hi = mp.mpf(0), mp.mpf(hi0)
    while f(lo) * f(hi) > 0 and hi < 5000:
        hi *= 2
    if f(lo) * f(hi) > 0:
        return None
    return _bisect(f, lo, hi)


def lambda_star_vp(B, s2, T):
    us = u_star_vp(B, s2, T)
    return None if us is None else mp.sqrt(us)


def u_star_general(sched, hi0=mp.mpf('0.05')):
    """Root via the general closed-form integral C_closed (ODE sweep)."""
    f = lambda u: C_closed(sched, u)
    if f(mp.mpf(0)) >= -mp.mpf('1e-12'):
        return None
    lo, hi = mp.mpf(0), mp.mpf(hi0)
    while f(lo) * f(hi) > 0 and hi < 5000:
        hi *= 2
    if f(lo) * f(hi) > 0:
        return None
    return _bisect(f, lo, hi)


def u_star_richardson(sched, Ns=(1024, 2048, 4096), hi0=mp.mpf('0.05'), integrator="EM"):
    """Root of C(u)=0 estimated from the recursion (Richardson). Robust for any schedule
    / integrator; used in universality (E8) and integrator (E9) experiments."""
    from coefficient import C_richardson
    f = lambda u: C_richardson(sched, u, Ns=Ns, integrator=integrator)[0]
    if f(mp.mpf(0)) >= -mp.mpf('1e-6'):   # looser: Richardson C0 is a finite-N estimate
        return None
    lo, hi = mp.mpf(0), mp.mpf(hi0)
    while f(lo) * f(hi) > 0 and hi < 5000:
        hi *= 2
    if f(lo) * f(hi) > 0:
        return None
    return _bisect(f, lo, hi, maxit=60)


if __name__ == "__main__":
    from diffusion import set_dps
    set_dps(40)
    B, T = 4.0, 5.0
    print("=== lambda* vs s2 (B=4,T=5): root present for s2>1, absent for s2<=1 ===")
    for s2 in [0.5, 0.9, 1.0, 1.5, 2.0, 4.0, 16.0, 64.0]:
        ls = lambda_star_vp(B, s2, T)
        print(f"  s2={s2:7}: lam*={'None' if ls is None else mp.nstr(ls,8):>10}"
              + ("" if ls is None else f"   lam*/sqrt(s2)={mp.nstr(ls/mp.sqrt(s2),6)}"))
    print("\n=== invariance to B,T (s2=2) ===")
    for B in [1.0, 2.0, 4.0, 8.0]:
        print(f"  B={B:5}: lam*={mp.nstr(lambda_star_vp(B,2.0,5.0),8)}")
    for T in [2.0, 5.0, 12.0, 25.0]:
        print(f"  T={T:5}: lam*={mp.nstr(lambda_star_vp(4.0,2.0,T),8)}")
