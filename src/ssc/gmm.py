"""Gaussian-mixture reverse sampler (numpy, Monte-Carlo) for the non-Gaussian boundary (E10).
Target: 1/2 N(-mu, s2) + 1/2 N(+mu, s2) under VP forward dx=-(B/2)x dt + sqrt(B) dW.

Forward marginal of a single Gaussian component N(c, s2) is N(c*m_t, V_t), m_t=e^{-Bt/2},
V_t = s2*m_t^2 + (1-m_t^2)? -- for VP the data mean shrinks by m_t and variance ->
s2 m_t^2 + (1-m_t^2). For the symmetric two-mode mixture the exact score is
    score(x,t) = (-x + mu*m_t*tanh(mu*m_t*x/V_t)) / V_t,
with V_t the per-component marginal variance. (Sign verified in ../stepdoubling/gmm_debug.py:
the forward-running-time reverse drift is  (B/2)x + (1+lam^2)/2 * B * score.)
"""
from __future__ import annotations
import numpy as np


def m_t(t, B):
    return np.exp(-B * t / 2.0)


def V_t(t, B, s2):
    mt = m_t(t, B)
    return s2 * mt ** 2 + (1.0 - mt ** 2)


def score_gmm(x, t, B, s2, mu):
    V = V_t(t, B, s2); mt = m_t(t, B)
    a = mu * mt / V
    return (-x + mu * mt * np.tanh(a * x)) / V


def sample(N, T, B, lam, s2, mu, P, rng):
    """Reverse EM sampler from t=T (prior) to t=0 (data). Returns P samples."""
    dt = T / N
    # init at the true prior marginal at t=T: mixture of N(±mu m_T, V_T)
    mt_T = m_t(T, B); VT = V_t(T, B, s2)
    sgn = rng.integers(0, 2, size=P) * 2 - 1
    x = sgn * mu * mt_T + np.sqrt(VT) * rng.standard_normal(P)
    c = lam * np.sqrt(B); sq = np.sqrt(dt)
    for k in range(N):
        tk = T - k * dt
        sc = score_gmm(x, tk, B, s2, mu)
        drift = (B / 2.0) * x + (1.0 + lam ** 2) / 2.0 * B * sc   # forward-running reverse drift (PLUS sign)
        x = x + drift * dt + c * sq * rng.standard_normal(P)
    return x


def target_sample(P, s2, mu, rng):
    sgn = rng.integers(0, 2, size=P) * 2 - 1
    return sgn * mu + np.sqrt(s2) * rng.standard_normal(P)


if __name__ == "__main__":
    import mpmath as mp
    rng = np.random.default_rng(0)
    # sanity: mu=0 reduces to the Gaussian variance recursion
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from diffusion import vp_const, set_dps
    from recursion import v_terminal
    set_dps(40)
    s2, B, T = 1.0, 4.0, 5.0
    for lam in [0.0, 1.0]:
        x = sample(128, T, B, lam, s2, 0.0, 2_000_000, rng)
        ve = float(v_terminal(vp_const(B, s2, T), 128, lam ** 2))
        print(f"mu=0 lam={lam}: var_emp={np.var(x):.5f} var_exact={ve:.5f} reldiff={abs(np.var(x)-ve)/ve:.2e}")
