"""Distributional metrics. Gaussian closed forms (mpmath) + empirical (numpy) for Monte-Carlo."""
from __future__ import annotations
import mpmath as mp
import numpy as np


def kl_gauss(v, s2):
    """KL( N(0,v) || N(0,s2) ) per coordinate = 1/2 (v/s2 - 1 - ln(v/s2))."""
    v = mp.mpf(v); s2 = mp.mpf(s2); r = v / s2
    return (r - 1 - mp.log(r)) / 2


def w2sq_gauss(v, s2):
    """W2^2 between N(0,v) and N(0,s2) per coordinate = (sqrt(v)-sqrt(s2))^2."""
    v = mp.mpf(v); s2 = mp.mpf(s2)
    return (mp.sqrt(v) - mp.sqrt(s2)) ** 2


def kl_gauss_aniso(vs, s2s):
    """Sum over diagonal modes of per-mode KL."""
    return sum(kl_gauss(v, s2) for v, s2 in zip(vs, s2s))


def w2sq_emp_1d(a, b):
    """Empirical W2^2 in 1D via sorted optimal transport (numpy)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n = min(len(a), len(b))
    # if equal length, exact 1D OT; else interpolate quantiles
    if len(a) == len(b):
        return float(np.mean((a - b) ** 2))
    qs = (np.arange(n) + 0.5) / n
    aa = np.quantile(a, qs); bb = np.quantile(b, qs)
    return float(np.mean((aa - bb) ** 2))


def w2sq_emp_1d_debiased(a, ref_sampler, n_ref_seeds=4, rng=None):
    """Debiased empirical W2^2: subtract the same-distribution sampling floor.
    ref_sampler() returns an independent sample of the TARGET of the same size as a."""
    rng = rng or np.random.default_rng(0)
    raw = w2sq_emp_1d(a, ref_sampler())
    floors = [w2sq_emp_1d(ref_sampler(), ref_sampler()) for _ in range(n_ref_seeds)]
    floor = float(np.mean(floors))
    return raw - floor, raw, floor


def local_order(kl_lo, kl_hi, N_lo, N_hi):
    """Measured order p in KL ~ N^-p between two step counts."""
    return float(mp.log(mp.mpf(kl_lo) / mp.mpf(kl_hi)) / mp.log(mp.mpf(N_hi) / mp.mpf(N_lo)))


def golden_min(f, a, b, iters=60):
    """Golden-section minimisation of scalar f on [a,b] (f returns mpf/float). Returns argmin."""
    gr = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(a), mp.mpf(b)
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    return (a + b) / 2


if __name__ == "__main__":
    print("KL(2->1) per-coord:", mp.nstr(kl_gauss(2.0, 1.0), 8))
    print("local order check (KL halves per N-doubling at p=1):",
          local_order(1.0, 0.5, 100, 200))
