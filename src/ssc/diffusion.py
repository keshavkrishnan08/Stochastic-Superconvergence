"""Linear diffusion schedules for the affine-variance machinery (mpmath, high precision).

A linear forward SDE  dx = a(t) x dt + b(t) dW  has Gaussian marginal N(0, V(t)) with
    V'(t) = 2 a(t) V(t) + b(t)^2,        exact score  grad log p_t(x) = -x / V(t).

The churn-lambda reverse family, written in forward-running time tau = T - t, has linear
drift  -A(t,u) x  with
    A(t,u) = a(t) + (1+u)/2 * b2(t) / V(t),     u = lambda^2,
and per-step Euler-Maruyama variance map
    v_{k+1} = (1 - A(t_k,u) dt)^2 v_k + u * b2(t_k) * dt.

Leading weak-error (local truncation) density and integrating factor:
    sigma(t,u) = A(t,u)^2 V(t) - 1/2 V''(t),
    Phi(t,u)   = exp(-2 * integral_0^t A(t',u) dt').
Closed-form Talay-Tubaro coefficient (proved in theory/derivations.md):
    C(lambda) = lim_N N (v_N - s2) = T * integral_0^T Phi(t,u) sigma(t,u) dt.

This module provides Schedule objects exposing V, V', V'', a, b2 and the derived A, sigma, Phi
for several families (VP constant beta, VP linear ramp, VE linear, VE quadratic) so the SAME
coefficient/recursion code handles the VP headline and the universality experiments (Thm D).
All quantities are mpmath mpf for exactness; set precision with set_dps().
"""
from __future__ import annotations
import mpmath as mp


def set_dps(dps: int = 50):
    mp.mp.dps = int(dps)


class Schedule:
    """A linear diffusion schedule. All callables take/return mpmath mpf.

    Required: name, s2 (data variance), T (horizon), and callables
      V(t), Vp(t)=V'(t), Vpp(t)=V''(t), a(t), b2(t).
    Derived:  A(t,u), sigma(t,u), Phi(t,u), int_A(t,u).
    """

    def __init__(self, name, s2, T, V, Vp, Vpp, a, b2, kind="VP"):
        self.name = name
        self.s2 = mp.mpf(s2)
        self.T = mp.mpf(T)
        self._V, self._Vp, self._Vpp, self._a, self._b2 = V, Vp, Vpp, a, b2
        self.kind = kind

    # marginal
    def V(self, t):   return self._V(mp.mpf(t))
    def Vp(self, t):  return self._Vp(mp.mpf(t))
    def Vpp(self, t): return self._Vpp(mp.mpf(t))
    def a(self, t):   return self._a(mp.mpf(t))
    def b2(self, t):  return self._b2(mp.mpf(t))

    # reverse-drift coefficient (variance map), u = lambda^2
    def A(self, t, u):
        t = mp.mpf(t); u = mp.mpf(u)
        return self._a(t) + (1 + u) / 2 * self._b2(t) / self._V(t)

    # leading local-error density
    def sigma(self, t, u):
        t = mp.mpf(t); u = mp.mpf(u)
        return self.A(t, u) ** 2 * self._V(t) - mp.mpf('0.5') * self._Vpp(t)

    # log integrating factor  -2 int_0^t A
    def logPhi(self, t, u):
        t = mp.mpf(t); u = mp.mpf(u)
        return -2 * mp.quad(lambda tt: self.A(tt, u), [0, t])

    def Phi(self, t, u):
        return mp.e ** self.logPhi(t, u)


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------
def vp_const(B, s2, T):
    """Variance-preserving OU with constant beta = B.  dx = -(B/2)x dt + sqrt(B) dW.
    V(t) = 1 + (s2-1) e^{-Bt};  a=-B/2; b2=B;  V'=-B(V-1); V''=B^2 (V-1)."""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); one = mp.mpf(1)
    V   = lambda t: one + (s2 - 1) * mp.e ** (-B * t)
    Vp  = lambda t: -B * (V(t) - 1)
    Vpp = lambda t: B ** 2 * (V(t) - 1)
    a   = lambda t: -B / 2
    b2  = lambda t: B
    return Schedule(f"VP(B={mp.nstr(B,4)})", s2, T, V, Vp, Vpp, a, b2, kind="VP")


def vp_ramp(B0, B1, s2, T):
    """Time-varying VP, beta(t)=B0+B1 t (variance-preserving: a=-beta/2, b2=beta).
    int_0^t beta = B0 t + B1 t^2/2;  V = 1 + (s2-1) exp(-that)."""
    B0 = mp.mpf(B0); B1 = mp.mpf(B1); s2 = mp.mpf(s2); T = mp.mpf(T); one = mp.mpf(1)
    beta = lambda t: B0 + B1 * t
    Beta = lambda t: B0 * t + B1 * t ** 2 / 2
    V   = lambda t: one + (s2 - 1) * mp.e ** (-Beta(t))
    Vp  = lambda t: -beta(t) * (V(t) - 1)
    # V'' = (V-1)(beta^2 - beta')  with beta'=B1
    Vpp = lambda t: (V(t) - 1) * (beta(t) ** 2 - B1)
    a   = lambda t: -beta(t) / 2
    b2  = lambda t: beta(t)
    return Schedule(f"VPramp(B0={mp.nstr(B0,3)},B1={mp.nstr(B1,3)})", s2, T, V, Vp, Vpp, a, b2, kind="VP")


def ve_linear(c, s2, T):
    """Variance-exploding, sigma^2(t)=c t.  V=s2+c t;  a=0; b2=c; V'=c; V''=0."""
    c = mp.mpf(c); s2 = mp.mpf(s2); T = mp.mpf(T)
    V   = lambda t: s2 + c * t
    Vp  = lambda t: c
    Vpp = lambda t: mp.mpf(0)
    a   = lambda t: mp.mpf(0)
    b2  = lambda t: c
    return Schedule(f"VElin(c={mp.nstr(c,4)})", s2, T, V, Vp, Vpp, a, b2, kind="VE")


def ve_quad(c, s2, T):
    """Variance-exploding, sigma^2(t)=c t^2 (EDM-like sigma~t).  V=s2+c t^2; a=0; b2=2c t; V'=2c t; V''=2c."""
    c = mp.mpf(c); s2 = mp.mpf(s2); T = mp.mpf(T)
    V   = lambda t: s2 + c * t ** 2
    Vp  = lambda t: 2 * c * t
    Vpp = lambda t: 2 * c
    a   = lambda t: mp.mpf(0)
    b2  = lambda t: 2 * c * t
    return Schedule(f"VEquad(c={mp.nstr(c,4)})", s2, T, V, Vp, Vpp, a, b2, kind="VE")


SCHEDULES = {
    "vp_const": vp_const, "vp_ramp": vp_ramp, "ve_linear": ve_linear, "ve_quad": ve_quad,
}


if __name__ == "__main__":
    set_dps(40)
    print("Consistency check: V'(t) == 2 a V + b2 for each schedule (should be ~0):")
    for sched in [vp_const(4, 2, 5), vp_ramp(2, 1, 2, 5), ve_linear(1.5, 2, 5), ve_quad(0.5, 2, 5)]:
        t = mp.mpf('1.3')
        lhs = sched.Vp(t); rhs = 2 * sched.a(t) * sched.V(t) + sched.b2(t)
        # V'' check via finite diff of V'
        h = mp.mpf('1e-15'); vpp_fd = (sched.Vp(t + h) - sched.Vp(t - h)) / (2 * h)
        print(f"  {sched.name:28s} V'-(2aV+b2)={mp.nstr(lhs-rhs,3):>8}  "
              f"V''-fd={mp.nstr(sched.Vpp(t)-vpp_fd,3):>10}")
