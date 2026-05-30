"""E67 -- Graceful degradation of superconvergence off the Gaussian (exact density propagation, no MC).

Target: symmetric two-Gaussian mixture  p0 = 1/2 N(-mu,sig2) + 1/2 N(+mu,sig2), with TOTAL variance
s2 = mu^2 + sig2 held fixed; mu tunes the non-Gaussianity (mu=0 is exactly Gaussian N(0,s2)). For each mu we
run the reverse VP Euler-Maruyama sampler with the EXACT (nonlinear) mixture score and propagate the full 1-D
density on a grid -- deterministic transport (change of variables) followed by Gaussian convolution per step,
no Monte Carlo. We measure the terminal variance-error convergence order at lambda=0 and at the empirically
best churn. mu=0 reproduces the exact-recursion order (a self-check); as mu grows the best-churn order slides
from 2 (variance-error; KL order 4) toward 1 (KL order 2), so the cancellation degrades gracefully and the
order gain is a continuous function of the non-Gaussianity.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io

NAME = "e67_nongaussian"


def _phi(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def mixture_score(x, m, Vt):
    """d/dx log [1/2 phi(x;-m,Vt)+1/2 phi(x;m,Vt)]  (exact, nonlinear for m>0)."""
    pm = _phi(x, -m, Vt); pp = _phi(x, m, Vt); p = 0.5 * (pm + pp) + 1e-300
    return (0.5 * pm * (-(x + m) / Vt) + 0.5 * pp * (-(x - m) / Vt)) / p


def terminal_density(mu, sig2, B, T, lam, N, L=14.0, G=8192):
    """Propagate the reverse VP-Euler sampler density to t=0. Returns (grid x, density p)."""
    s2 = mu ** 2 + sig2; u = lam ** 2; dt = T / N
    x = np.linspace(-L, L, G); dx = x[1] - x[0]
    abar = lambda t: math.exp(-B * t)
    m_of = lambda t: mu * math.sqrt(abar(t)); V_of = lambda t: abar(t) * sig2 + (1 - abar(t))
    # start at the exact marginal at t=T (prior)
    mT, VT = m_of(T), V_of(T)
    p = 0.5 * (_phi(x, -mT, VT) + _phi(x, mT, VT)); p /= np.trapz(p, x)
    noise_var_step = u * B * dt
    for k in range(N):
        t = T - k * dt; m, Vt = m_of(t), V_of(t)
        sc = mixture_score(x, m, Vt)
        drift = (B / 2.0) * x + (1.0 + u) / 2.0 * B * sc
        y = x + drift * dt
        # local Jacobian dy/dx by finite difference (monotone for small dt)
        dydx = np.gradient(y, x); dydx = np.clip(dydx, 1e-6, None)
        q = p / dydx                                   # pushforward density at y
        order = np.argsort(y)
        p = np.interp(x, y[order], q[order], left=0.0, right=0.0)
        if noise_var_step > 1e-300:                    # convolve with N(0, u B dt) via FFT
            kx = np.fft.fftfreq(G, d=dx) * 2 * np.pi
            p = np.fft.irfft(np.fft.rfft(p) * np.exp(-0.5 * noise_var_step * kx[:G // 2 + 1] ** 2), n=G)
        p = np.clip(p, 0, None); s = np.trapz(p, x); p = p / (s + 1e-300)
    return x, p


def var_and_kl(x, p, mu, sig2):
    s2 = mu ** 2 + sig2
    var = np.trapz(p * x ** 2, x)                      # mean is 0 by symmetry
    ptgt = 0.5 * (_phi(x, -mu, sig2) + _phi(x, mu, sig2))
    mask = p > 1e-12
    kl = np.trapz(np.where(mask, p * np.log((p + 1e-300) / (ptgt + 1e-300)), 0.0), x)
    return float(var), float(abs(var - s2)), float(max(kl, 0.0))


def _slope(Ns, ys):
    lx = [math.log(n) for n in Ns]; ly = [math.log(max(y, 1e-300)) for y in ys]; n = len(lx)
    mx = sum(lx) / n; my = sum(ly) / n
    return -sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def run(B=4.0, T=5.0, s2=2.0):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return
    t0 = time.time()
    from lambda_star import lambda_star_vp
    from diffusion import set_dps
    set_dps(30)
    lam_star = float(lambda_star_vp(B, s2, T))
    mus = [0.0, 0.25, 0.5, 0.75, 1.0, 1.2]             # mu^2<=s2; sig2=s2-mu^2
    Ns = [12, 16, 24, 32, 48, 64, 96]
    lams = [0.0, 0.4, 0.8, lam_star, 1.6, 2.0]
    rows = []
    for mu in mus:
        sig2 = s2 - mu ** 2
        # variance-error curve at each churn
        ve = {}
        for lam in lams:
            ve[lam] = [var_and_kl(*terminal_density(mu, sig2, B, T, lam, N), mu, sig2)[1] for N in Ns]
        # best churn = min variance error at the largest N
        best_lam = min(lams, key=lambda L: ve[L][-1])
        order_det = _slope(Ns, ve[0.0]); order_best = _slope(Ns, ve[best_lam])
        order_lamstar = _slope(Ns, ve[lam_star])
        rows.append({"mu": mu, "sig2": sig2, "best_lam": best_lam, "order_det": order_det,
                     "order_best": order_best, "order_lamstar": order_lamstar,
                     "ve_det": ve[0.0], "ve_best": ve[best_lam], "Ns": Ns})
        io.log(f"  E67 mu={mu:.2f}: order@lam0={order_det:.2f} order@best(lam={best_lam:.2f})={order_best:.2f} "
               f"order@lam*={order_lamstar:.2f}  ({time.time()-t0:.0f}s)")
    io.save(NAME, {"config": {"B": B, "T": T, "s2": s2, "lambda_star": lam_star, "mus": mus, "Ns": Ns,
                              "lams": lams}, "rows": rows})
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s")


def figure():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from figstyle import COL
    r = io.load(NAME)
    if not r:
        return
    rows = r["rows"]; mus = np.array([x["mu"] for x in rows])
    o_best = np.array([x["order_best"] for x in rows]); o_det = np.array([x["order_det"] for x in rows])
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 175, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel A: best-churn variance-error order vs non-Gaussianity
    axA.axhline(2, ls=":", color=COL["lstar"], lw=1.2); axA.axhline(1, ls=":", color=COL["ode"], lw=1.2)
    axA.plot(mus, o_best, "s-", color=COL["lstar"], lw=2.2, ms=6, label="best churn")
    axA.plot(mus, o_det, "o--", color=COL["ode"], lw=1.8, ms=5, label=r"$\lambda{=}0$")
    axA.text(mus[-1], 2.05, "Gaussian cancellation (order 2)", ha="right", fontsize=8, color=COL["lstar"])
    axA.text(mus[-1], 1.05, "no cancellation (order 1)", ha="right", fontsize=8, color=COL["ode"])
    axA.set_xlabel(r"non-Gaussianity (mixture separation $\mu$)")
    axA.set_ylabel("variance-error order")
    axA.set_title("the cancellation degrades gracefully off-Gaussian")
    axA.set_ylim(0.7, 2.3); axA.legend(fontsize=9, framealpha=0.9); axA.grid(True, alpha=0.18)
    # Panel B: variance-error vs N for a few mu (best churn), showing the slope flattening
    Ns = np.array(rows[0]["Ns"], float); cols = plt.cm.viridis(np.linspace(0.12, 0.82, len(rows)))
    for x, c in zip(rows, cols):
        axB.loglog(Ns, np.clip(x["ve_best"], 1e-300, None), "o-", color=c, lw=1.7, ms=3.5,
                   label=rf"$\mu={x['mu']:.2f}$")
    axB.loglog(Ns, rows[0]["ve_best"][0] * (Ns[0] / Ns) ** 2, ":", color="0.5", lw=1.1, label=r"$N^{-2}$")
    axB.set_xlabel(r"sampler steps $N$"); axB.set_ylabel("variance error at best churn")
    axB.set_title(r"best-churn descent flattens as $\mu$ grows")
    axB.legend(fontsize=7.6, ncol=2, framealpha=0.9); axB.grid(True, which="both", alpha=0.18)
    plt.tight_layout(); plt.savefig(os.path.join(io.FIG_DIR, "fig_nongaussian.png")); plt.close()
    io.log("fig_nongaussian.png", "figs.log")


if __name__ == "__main__":
    run(); figure()
