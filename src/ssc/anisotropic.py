"""Anisotropic target N(0, diag(s_1^2,...,s_d^2)) (Thm C).
Modes decouple, each with its own coefficient C_i(lambda)=C(lambda; s_i^2).
Total KL_N = (1/4N^2) sum_i C_i(lambda)^2 / s_i^4 + o(N^-2).
Super-convergence across all modes iff the active spectrum is degenerate; otherwise the
KL-optimal compromise churn lambda_dagger minimises K(lambda)=sum_i C_i^2/s_i^4.
"""
from __future__ import annotations
import mpmath as mp
from diffusion import vp_const
from recursion import v_terminal
from coefficient import C_closed_vp
from lambda_star import lambda_star_vp
from metrics import kl_gauss


def per_mode_C(B, lam, spectrum, T):
    return [C_closed_vp(B, lam, s2, T) for s2 in spectrum]


def per_mode_lambda_star(B, spectrum, T):
    return [lambda_star_vp(B, s2, T) for s2 in spectrum]


def K_of_lambda(B, lam, spectrum, T):
    """N^-2 prefactor (x4): sum_i C_i^2 / s_i^4."""
    return sum((C_closed_vp(B, lam, s2, T)) ** 2 / mp.mpf(s2) ** 4 for s2 in spectrum)


def lambda_dagger(B, spectrum, T, lam_lo=0.0, lam_hi=None):
    """Global-KL-optimal compromise churn: argmin_lambda K(lambda). Scan + golden refine."""
    if lam_hi is None:
        ls = [l for l in per_mode_lambda_star(B, spectrum, T) if l is not None]
        lam_hi = float(max(ls)) * 1.5 + 1.0 if ls else 5.0
    f = lambda lam: K_of_lambda(B, lam, spectrum, T)
    # coarse scan
    import numpy as np
    grid = np.linspace(float(lam_lo), float(lam_hi), 60)
    vals = [f(g) for g in grid]
    i = int(np.argmin([float(v) for v in vals]))
    a = grid[max(i - 1, 0)]; b = grid[min(i + 1, len(grid) - 1)]
    # golden-section refine
    gr = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(a), mp.mpf(b)
    c = b - gr * (b - a); d = a + gr * (b - a)
    for _ in range(60):
        if f(c) < f(d): b = d
        else: a = c
        c = b - gr * (b - a); d = a + gr * (b - a)
    return (a + b) / 2


def total_kl(B, lam, spectrum, T, N):
    return sum(kl_gauss(v_terminal(vp_const(B, s2, T), N, mp.mpf(lam) ** 2), s2) for s2 in spectrum)


if __name__ == "__main__":
    from diffusion import set_dps
    from metrics import local_order
    set_dps(60)
    B, T = 4.0, 5.0
    print("degenerate spectrum [2,2,2] -> common lambda*, order 4:")
    spec = [2.0, 2.0, 2.0]; ls = lambda_star_vp(B, 2.0, T)
    k1 = total_kl(B, ls, spec, T, 1024); k2 = total_kl(B, ls, spec, T, 4096)
    print(f"  lam*={mp.nstr(ls,6)} order={local_order(k1,k2,1024,4096):.3f}")
    print("non-degenerate [1.5,2.5,4.0] -> lambda_dagger, order 2:")
    spec = [1.5, 2.5, 4.0]; ld = lambda_dagger(B, spec, T)
    print(f"  per-mode lam*={[mp.nstr(x,5) for x in per_mode_lambda_star(B,spec,T)]}  lam_dagger={mp.nstr(ld,6)}")
    k1 = total_kl(B, ld, spec, T, 1024); k2 = total_kl(B, ld, spec, T, 4096)
    print(f"  order at lam_dagger={local_order(k1,k2,1024,4096):.3f}")
