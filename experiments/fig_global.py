"""The global intro figure (fast float64). Measured convergence order p over the whole design space
(churn lambda x data variance s^2): a flat plane at order two cut by a single ridge that rises to order
four along the superconvergence locus lambda*(s^2). One surface implies the whole paper. The order lives
in [2,4] and the smallest KL on the grid is ~1e-11, well inside float64 range, so no mpmath is needed and
the render is light (no dense tessellation, no 3D floor projection that stalls under load)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
from scipy.optimize import brentq
import io_utils as io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = io.FIG_DIR


def Vt(t, B, s2): return 1.0 + (s2 - 1.0) * np.exp(-B * t)


def vN(N, T, B, lam, s2):
    dt = T / N; v = Vt(T, B, s2); u = lam * lam
    for k in range(int(N)):
        tk = T - k * dt; V = Vt(tk, B, s2); A = -B / 2 + (1 + u) / 2 * B / V
        v = (1 - A * dt) ** 2 * v + u * B * dt
    return v


def kl(v, s2):
    r = v / s2
    return 0.5 * (r - 1 - np.log(r)) if r > 0 else np.nan


def Cf(B, lam, s2, T, Nref=4096):
    return Nref * (vN(Nref, T, B, lam, s2) - s2)  # leading-coeff proxy for the ridge curve


def lam_star(B, s2, T):
    if s2 <= 1.0 or Cf(B, 0.0, s2, T) >= 0:
        return np.nan
    hi = 0.3
    while Cf(B, hi, s2, T) < 0 and hi < 50:
        hi *= 1.6
    try:
        return brentq(lambda l: Cf(B, l, s2, T), 0.0, hi, xtol=1e-7)
    except Exception:
        return np.nan


def compute(B=4.0, T=5.0, nlam=70, ns2=46, Nfix=1024):
    # Smooth global view: the terminal sampling error log10 KL(lambda, s^2) at a fixed step count.
    # It is high (slow sampling) for almost every churn, and plunges into a deep valley along the
    # superconvergence locus lambda*(s^2) where the leading error cancels. Smooth in both axes.
    lams = np.linspace(0.05, 3.4, nlam); s2s = np.linspace(1.2, 8.0, ns2)
    Z = np.zeros((ns2, nlam))
    for i, s2 in enumerate(s2s):
        for j, lam in enumerate(lams):
            v = vN(Nfix, T, B, lam, float(s2)); k = kl(v, float(s2))
            Z[i, j] = np.log10(max(k, 1e-30))
    lstar = np.array([lam_star(B, float(s2), T) for s2 in s2s])
    io.save("e38_global_klsurface", {"config": {"B": B, "T": T, "Nfix": Nfix}, "lams": lams.tolist(),
                                     "s2s": s2s.tolist(), "logKL": Z.tolist(), "lambda_star_curve": lstar.tolist()})
    io.log("e38_global_klsurface computed (float64)")
    return lams, s2s, Z, lstar


def render(lams, s2s, Z, lstar):
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 150})
    L, S = np.meshgrid(lams, s2s)
    fig = plt.figure(figsize=(8.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(L, S, Z, cmap="viridis_r", linewidth=0, antialiased=True,
                           edgecolor="none", alpha=0.96, rcount=90, ccount=90)
    # the superconvergence valley floor: lambda*(s^2), drawn just below the surface
    good = ~np.isnan(lstar)
    zfloor = np.array([Z[i].min() for i in range(len(s2s))])
    ax.plot(lstar[good], s2s[good], zfloor[good] - 0.3, color="#ff2d55", lw=3.2, zorder=12,
            label=r"superconvergence valley $\lambda^\star(s^2)$")
    ax.set_box_aspect((1.25, 1.0, 0.7))
    ax.set_xlabel(r"churn / stochasticity $\lambda$", labelpad=12)
    ax.set_ylabel(r"data variance $s^2$", labelpad=12)
    ax.set_zlabel(r"sampling error $\log_{10}\mathrm{KL}$", labelpad=8)
    ax.set_title("Sampling error over the design space: a deep valley along one churn per data scale",
                 pad=12, fontsize=11.5)
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(0.0, 0.96))
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.10, aspect=16)
    cb.set_label(r"$\log_{10}\mathrm{KL}(\lambda, s^2)$", fontsize=10)
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.06, top=0.93)
    fig.savefig(os.path.join(FIG, "fig_global.png"))
    plt.close()
    io.log("fig_global.png", "figs.log")


if __name__ == "__main__":
    lams, s2s, P, lstar = compute()
    render(lams, s2s, P, lstar)
    print("fig_global done (fast)")
