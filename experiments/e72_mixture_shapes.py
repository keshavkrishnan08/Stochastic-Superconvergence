"""E72 -- Does the Gaussian cancellation churn lambda* stay the bias-minimising dial across non-Gaussian
SHAPES, not just the symmetric two-Gaussian of E67? Exact 1-D density propagation (no Monte Carlo), four
mixture families all with mean 0 and total variance s^2=2 (so lambda*(s^2=2)=1.2705 applies):

  sym2     symmetric two-Gaussian          w=[.5,.5]  m=[-.7,.7]   v=[1.51,1.51]
  asym2    asymmetric weighted two-Gaussian w=[.7,.3]  m=[-.6,1.4]  v=[1.16,1.16]
  tri      three-component                  w=[.25,.5,.25] m=[-1.4,0,1.4] v=[1.02]*3
  uneqv    unequal-variance two-Gaussian    w=[.5,.5]  m=[-.5,.5]   v=[.75,2.75]

For each we propagate the reverse VP-Euler sampler with the exact (nonlinear) mixture score and measure the
terminal variance-error order at lambda=0 and at lambda*. The cancellation needs a Gaussian iterate, so the
order does not reach the Gaussian 4; what we test is whether lambda* still IMPROVES the order over the
deterministic sampler across shapes (graceful, shape-robust degradation).
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io

NAME = "e72_mixture_shapes"


def _phi(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def mix_score(x, ms, Vs, ws):
    phis = [w * _phi(x, m, V) for w, m, V in zip(ws, ms, Vs)]
    p = sum(phis) + 1e-300
    num = sum(ph * (-(x - m) / V) for ph, m, V in zip(phis, ms, Vs))
    return num / p


def terminal_density(means, vars, ws, B, T, lam, N, L=16.0, G=8192):
    u = lam ** 2; dt = T / N; x = np.linspace(-L, L, G); dx = x[1] - x[0]
    abar = lambda t: math.exp(-B * t)
    mt = lambda t, m: m * math.sqrt(abar(t))
    Vt = lambda t, v: abar(t) * v + (1 - abar(t))
    p = sum(w * _phi(x, mt(T, m), Vt(T, v)) for w, m, v in zip(ws, means, vars)); p /= np.trapz(p, x)
    nvs = u * B * dt
    for k in range(N):
        t = T - k * dt
        ms = [mt(t, m) for m in means]; Vs = [Vt(t, v) for v in vars]
        sc = mix_score(x, ms, Vs, ws)
        y = x + ((B / 2.0) * x + (1.0 + u) / 2.0 * B * sc) * dt
        dydx = np.clip(np.gradient(y, x), 1e-6, None); q = p / dydx
        o = np.argsort(y); p = np.interp(x, y[o], q[o], left=0.0, right=0.0)
        if nvs > 1e-300:
            kx = np.fft.fftfreq(G, d=dx) * 2 * np.pi
            p = np.fft.irfft(np.fft.rfft(p) * np.exp(-0.5 * nvs * kx[:G // 2 + 1] ** 2), n=G)
        p = np.clip(p, 0, None); p /= (np.trapz(p, x) + 1e-300)
    return x, p


def ve(means, vars, ws, B, T, lam, N):
    x, p = terminal_density(means, vars, ws, B, T, lam, N)
    mean = np.trapz(p * x, x); var = np.trapz(p * (x - mean) ** 2, x)
    s2 = sum(w * (m ** 2 + v) for w, m, v in zip(ws, means, vars)) - sum(w * m for w, m in zip(ws, means)) ** 2
    return float(abs(var - s2))


def _slope(Ns, ys):
    lx = [math.log(n) for n in Ns]; ly = [math.log(max(y, 1e-300)) for y in ys]; n = len(lx)
    mx = sum(lx) / n; my = sum(ly) / n
    return -sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def run(B=4.0, T=5.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    from lambda_star import lambda_star_vp
    from diffusion import set_dps
    set_dps(30)
    lam_star = float(lambda_star_vp(B, 2.0, T))   # s^2=2 for every shape
    t0 = time.time()
    shapes = {
        "sym2":  (([-0.7, 0.7],       [1.51, 1.51],       [0.5, 0.5])),
        "asym2": (([-0.6, 1.4],       [1.16, 1.16],       [0.7, 0.3])),
        "tri":   (([-1.4, 0.0, 1.4],  [1.02, 1.02, 1.02], [0.25, 0.5, 0.25])),
        "uneqv": (([-0.5, 0.5],       [0.75, 2.75],       [0.5, 0.5])),
    }
    Ns = [12, 16, 24, 32, 48, 64, 96]
    lams = [0.0, 0.8, lam_star, 1.6]
    rows = []
    for name, (means, vars, ws) in shapes.items():
        veL = {lam: [ve(means, vars, ws, B, T, lam, N) for N in Ns] for lam in lams}
        best = min(lams, key=lambda L: veL[L][-1])
        o_det = _slope(Ns, veL[0.0]); o_star = _slope(Ns, veL[lam_star]); o_best = _slope(Ns, veL[best])
        rows.append({"shape": name, "order_det": o_det, "order_lamstar": o_star,
                     "best_lam": best, "order_best": o_best,
                     "improves": o_star > o_det + 0.3})
        io.log(f"  E72 {name:6s}: order@0={o_det:.2f}  order@lam*={o_star:.2f}  best_lam={best:.3f}  "
               f"(lam* improves: {o_star > o_det + 0.3})")
    io.save(NAME, {"config": {"B": B, "T": T, "s2": 2.0, "lambda_star": lam_star, "Ns": Ns, "lams": lams},
                   "rows": rows})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
