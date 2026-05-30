"""Exact affine terminal-variance recursion for the churn-lambda reverse sampler.

Because the target is Gaussian and the score linear, the EM iterate stays Gaussian and the
terminal variance is computed EXACTLY with zero sampling:
    v_{k+1} = (1 - A(t_k,u) dt)^2 v_k + u * b2(t_k) * dt,   u = lambda^2,  v_0 = V(T) + e0.
Run k=0..N-1 with t_k = T - k dt, dt = T/N.  KL/W2 to the target follow in closed form.

Integrators:
  'EM'    : Euler-Maruyama, weak order 1  -> generic KL ~ N^-2.
  'exact' : exact per-step linear map (e^{-2 int A} multiplier + exact additive) -> reference.
  'heun'  : deterministic Heun on the variance ODE (exploratory, Thm E) -> weak order 2.

All arithmetic in mpmath at the current precision (set via diffusion.set_dps).
"""
from __future__ import annotations
import mpmath as mp


def v_terminal(sched, N, u, e0=0, eps2=0, integrator="EM"):
    """Terminal sampler variance after N steps.
    e0   = initialization error  v_0 - V(T)  (transient term; Thm B).
    eps2 = persistent score-error power: extra variance injection eps2*dt per step
           (the 'floor' term; Thm B / Goldilocks). Both default 0 (super-convergence regime).
    """
    N = int(N); u = mp.mpf(u); T = sched.T; eps2 = mp.mpf(eps2)
    dt = T / N
    v = sched.V(T) + mp.mpf(e0)
    if integrator == "EM":
        for k in range(N):
            tk = T - k * dt
            A = sched.A(tk, u)
            v = (1 - A * dt) ** 2 * v + (u * sched.b2(tk) + eps2) * dt
    elif integrator == "heun":
        # g(t,v) = dv/dtau = -2 A(t,u) v + u b2(t);  t decreases by dt as tau advances.
        def g(t, vv): return -2 * sched.A(t, u) * vv + u * sched.b2(t)
        for k in range(N):
            tk = T - k * dt
            tk1 = T - (k + 1) * dt
            pred = v + dt * g(tk, v)
            v = v + dt / 2 * (g(tk, v) + g(tk1, pred))
    elif integrator == "exact":
        # exact one-step linear map: v -> M v + R, M = exp(-2 int_{tk1}^{tk} A dt'),
        # R = u * int over step of exp(...) b2.  Use mpmath quad per step (slow; reference only).
        for k in range(N):
            tk = T - k * dt
            tk1 = T - (k + 1) * dt
            M = mp.e ** (-2 * mp.quad(lambda tt: sched.A(tt, u), [tk1, tk]))
            # additive: dv/dtau = -2A v + u b2 ; particular accumulation over [tk1,tk] in tau
            R = u * mp.quad(lambda tt: mp.e ** (-2 * mp.quad(lambda s: sched.A(s, u), [tt, tk])) * sched.b2(tt),
                            [tk1, tk])
            v = M * v + R
    else:
        raise ValueError(integrator)
    return v


def v_continuous_endpoint(sched, u):
    """Lemma 2 check: integrate the EXACT continuous reverse variance ODE
       dv/dtau = -2 A(t) v + u b2(t),  t = T - tau,  v(0)=V(T),
    from tau=0 to tau=T and return v(T) (should equal s2 exactly, since the continuous
    reverse process reproduces the forward marginal). No discretisation error."""
    u = mp.mpf(u); T = sched.T
    def F(tau, y):
        t = T - tau
        return [-2 * sched.A(t, u) * y[0] + u * sched.b2(t)]
    sol = mp.odefun(F, mp.mpf(0), [sched.V(T)], tol=mp.mpf(10) ** (-(mp.mp.dps - 8)))
    return sol(T)[0]


def v_terminal_closed(sched, N, u, e0=0):
    """Closed-form product/sum form of the EM recursion (cross-check of v_terminal EM)."""
    N = int(N); u = mp.mpf(u); T = sched.T; dt = T / N
    ms = []
    for k in range(N):
        tk = T - k * dt
        ms.append((1 - sched.A(tk, u) * dt) ** 2)
    # prod all
    P = mp.mpf(1)
    for m in ms:
        P *= m
    v = P * (sched.V(T) + mp.mpf(e0))
    # additive: sum_k (prod_{j>k} m_j) u b2(t_k) dt
    suffix = mp.mpf(1)
    add = mp.mpf(0)
    # iterate k from N-1 down: suffix after k is prod_{j=k+1}^{N-1} m_j
    for k in range(N - 1, -1, -1):
        tk = T - k * dt
        add += suffix * u * sched.b2(tk) * dt
        suffix *= ms[k]
    return v + add


if __name__ == "__main__":
    from diffusion import vp_const, set_dps
    set_dps(50)
    s2, B, T = 2.0, 4.0, 5.0
    sched = vp_const(B, s2, T)
    print("recursion EM == closed-form product/sum (reldiff should be ~0):")
    for lam in [0.0, 0.5, 1.0, 2.0]:
        for N in [16, 64, 256, 1024]:
            a = v_terminal(sched, N, lam ** 2, integrator="EM")
            b = v_terminal_closed(sched, N, lam ** 2)
            print(f"  lam={lam:4} N={N:5d}  reldiff={mp.nstr(abs(a-b)/abs(b),4)}")
