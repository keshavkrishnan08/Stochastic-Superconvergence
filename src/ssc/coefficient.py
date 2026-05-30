"""The Talay-Tubaro leading weak-error coefficient C(lambda) = lim_N N (v_N - s2).

General (any linear schedule):   C(lambda) = T * integral_0^T Phi(t,u) sigma(t,u) dt,
with sigma = A^2 V - 1/2 V''  and  Phi = exp(-2 int_0^t A).   [proved in derivations.md]

For VP-constant-beta the integrating factor and density are elementary:
    Phi(t) = s2^{1+u} e^{-u B t} / V(t)^{1+u},   sigma(t) = (B^2/4)[ -V + (1+u)^2/V - 2u ],
and C(0), C(1) have closed-form antiderivatives (anchors), used as exact unit tests.

C_richardson is a schedule/integrator-agnostic numerical estimate of C from the recursion
(Richardson extrapolation of N (v_N - s2)); E2 checks C_closed == C_richardson.
"""
from __future__ import annotations
import mpmath as mp
from recursion import v_terminal


# ----- general closed-form integral (works for any Schedule) -----
def C_closed(sched, u, method="ode"):
    """C(lambda) = T * integral_0^T Phi sigma dt, Phi=exp(-2 int A).
    'ode'  : single sweep of the joint ODE  I'=Phi*sigma, Phi'=-2A Phi, I(0)=0, Phi(0)=1
             (avoids nested quadrature; fast).  C = T * I(T).
    'quad' : direct nested quadrature (slow; reference only)."""
    u = mp.mpf(u); T = sched.T
    if method == "quad":
        return T * mp.quad(lambda t: sched.Phi(t, u) * sched.sigma(t, u), [0, T])
    # ODE sweep: y=[I, Phi]
    def F(t, y):
        Phi = y[1]
        return [Phi * sched.sigma(t, u), -2 * sched.A(t, u) * Phi]
    sol = mp.odefun(F, mp.mpf(0), [mp.mpf(0), mp.mpf(1)], tol=mp.mpf(10) ** (-(mp.mp.dps - 8)))
    return T * sol(T)[0]


# ----- VP-explicit integrand (fast path; avoids nested quads in Phi) -----
def C_closed_vp(B, lam, s2, T):
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); u = mp.mpf(lam) ** 2; one = mp.mpf(1)
    V = lambda t: s2 * mp.e ** (-B * t) + (one - mp.e ** (-B * t))
    Phi = lambda t: s2 ** (one + u) * mp.e ** (-u * B * t) / V(t) ** (one + u)
    sig = lambda t: (B ** 2 / 4) * (-V(t) + (one + u) ** 2 / V(t) - 2 * u)
    return T * mp.quad(lambda t: Phi(t) * sig(t), [0, T])


# ----- score-floor coefficient D(lambda) = int_0^T Phi dt (Thm B) -----
# With a persistent score-error power eps^2 (extra eps^2*dt injection per step), the steady
# excess variance is D(lambda)*eps^2, where the SAME integrating factor Phi appears:
#   v_N - s2 = C(lambda)/N + D(lambda) eps^2 + O(N^-2)   (e0=0).
# The KL-optimal churn cancels the leading variance error: C(lambda) = -N D(lambda) eps^2,
# a FLOOR-SHIFTED superconvergence that -> lambda* (root of C) as eps -> 0.
def D_closed_vp(B, lam, s2, T):
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); u = mp.mpf(lam) ** 2; one = mp.mpf(1)
    V = lambda t: s2 * mp.e ** (-B * t) + (one - mp.e ** (-B * t))
    Phi = lambda t: s2 ** (one + u) * mp.e ** (-u * B * t) / V(t) ** (one + u)
    return mp.quad(Phi, [0, T])


def c2_closed_vp(B, lam, s2, T):
    """Exact second-order coefficient: v_N - s2 = C/N + c2/N^2 + O(N^-3)  (e0=eps=0).

    Derived from the discrete Duhamel form e_N = -sum_k P_k d_k of the EM variance recursion,
    with one-step defect d_k = -h^2 sigma_k - (h^3/6) V'''(t_k) + O(h^4) and discrete homogeneous
    solution P_k = Phi(t_k)(1 + h g1(t_k) + O(h^2)), g1 = A(t)+A(0)-int_0^t A^2. Euler-Maclaurin of
    the resulting sum gives

        c2 = T^2 [ 1/2 (Phi*sigma|_T - Phi*sigma|_0) + int_0^T Phi (g1*sigma + V'''/6) dt ].

    Verified against Richardson extrapolation of N^2 (v_N - s2) to 12 digits. At lambda* (C=0)
    the terminal KL is exactly c2(lambda*)^2/(4 s^4) N^-4 + O(N^-5), with c2(lambda*) != 0."""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T); u = mp.mpf(lam) ** 2; one = mp.mpf(1)
    V    = lambda t: one + (s2 - 1) * mp.e ** (-B * t)
    Vppp = lambda t: -B ** 3 * (V(t) - 1)
    A    = lambda t: -B / 2 + (one + u) / 2 * B / V(t)
    Phi  = lambda t: s2 ** (one + u) * mp.e ** (-u * B * t) / V(t) ** (one + u)
    sig  = lambda t: (B ** 2 / 4) * (-V(t) + (one + u) ** 2 / V(t) - 2 * u)
    A0   = A(0)
    g1   = lambda t: A(t) + A0 - mp.quad(lambda x: A(x) ** 2, [0, t])
    endpoint = (Phi(T) * sig(T) - Phi(0) * sig(0)) / 2
    integ = mp.quad(lambda t: Phi(t) * (g1(t) * sig(t) + Vppp(t) / 6), [0, T])
    return T ** 2 * (endpoint + integ)


def c2_richardson(sched, u, Ns=(4096, 8192, 16384, 32768)):
    """Second-order coefficient from the recursion: fit N^2 (v_N - s2) = c2 + b/N + ... ."""
    g = [mp.mpf(N) ** 2 * (v_terminal(sched, N, u) - sched.s2) for N in Ns]
    xs = [1 / mp.mpf(N) for N in Ns]
    k = len(Ns); Vm = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            Vm[i, j] = xs[i] ** j
    return mp.lu_solve(Vm, mp.matrix(g))[0]


def D_richardson(sched, u, Ns=(2048, 4096, 8192)):
    """Floor coefficient from the recursion: D = lim_N [v_N(eps2=1) - v_N(eps2=0)] (e0=0)."""
    g = [v_terminal(sched, N, u, eps2=1) - v_terminal(sched, N, u, eps2=0) for N in Ns]
    xs = [1 / mp.mpf(N) for N in Ns]
    k = len(Ns)
    Vmat = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            Vmat[i, j] = xs[i] ** j
    return mp.lu_solve(Vmat, mp.matrix(g))[0]


# ----- elementary anchors (exact unit tests) -----
def C0_vp(B, s2, T):
    """u=0 closed form. F(V) = -ln V + 1/V + ln(V-1)."""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T)
    V_T = 1 + (s2 - 1) * mp.e ** (-B * T)
    F = lambda V: -mp.log(V) + 1 / V + mp.log(V - 1)
    int_invV2 = -(1 / B) * (F(V_T) - F(s2))
    return T * (B ** 2 / 4) * s2 * (-T + int_invV2)


def C1_vp(B, s2, T):
    """u=1 closed form. G(V) = -ln V - 2/V^2 + 2/V."""
    B = mp.mpf(B); s2 = mp.mpf(s2); T = mp.mpf(T)
    V_T = 1 + (s2 - 1) * mp.e ** (-B * T)
    G = lambda V: -mp.log(V) - 2 / V ** 2 + 2 / V
    INT = -s2 ** 2 * B / (4 * (s2 - 1)) * (G(V_T) - G(s2))
    return T * INT


# ----- numerical estimate from the recursion (any schedule/integrator) -----
def C_richardson(sched, u, Ns=(2048, 4096, 8192), integrator="EM"):
    """Estimate C = lim_N N (v_N - s2) by polynomial extrapolation.
    Model g(N) := N (v_N - s2) = C + d1/N + d2/N^2 + ... and fit a degree-(k-1)
    polynomial in x=1/N through k points (x small), reading off the intercept C.
    With 3 points this cancels the 1/N AND 1/N^2 terms -> residual ~ 1/N^3.
    Returns (C_estimate, diagnostic_dict)."""
    s2 = sched.s2
    g = [mp.mpf(N) * (v_terminal(sched, N, u, integrator=integrator) - s2) for N in Ns]
    xs = [1 / mp.mpf(N) for N in Ns]
    k = len(Ns)
    # Vandermonde solve for polynomial coeffs in x; intercept is C.
    Vmat = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            Vmat[i, j] = xs[i] ** j
    coeffs = mp.lu_solve(Vmat, mp.matrix(g))
    C = coeffs[0]
    return C, {"g": [mp.nstr(x, 12) for x in g], "C": mp.nstr(C, 12)}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from diffusion import vp_const, set_dps
    set_dps(50)
    s2, B, T = 2.0, 4.0, 5.0
    sched = vp_const(B, s2, T)
    print(f"=== C(lam): closed-form vs Richardson, vs anchors (s2={s2} B={B} T={T}) ===")
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0]:
        Cc = C_closed_vp(B, lam, s2, T)
        Cg = C_closed(sched, lam ** 2)
        Cr, _ = C_richardson(sched, lam ** 2)
        rd = abs(Cc - Cr) / abs(Cr)
        print(f"  lam={lam:4}: C_vp={mp.nstr(Cc,12):>15}  C_gen={mp.nstr(Cg,12):>15}  "
              f"C_rich={mp.nstr(Cr,12):>15}  rel(vp,rich)={mp.nstr(rd,4)}")
    print("anchors:")
    print(f"  C(0): elem={mp.nstr(C0_vp(B,s2,T),14)}  quad={mp.nstr(C_closed_vp(B,0,s2,T),14)}")
    print(f"  C(1): elem={mp.nstr(C1_vp(B,s2,T),14)}  quad={mp.nstr(C_closed_vp(B,1,s2,T),14)}")
