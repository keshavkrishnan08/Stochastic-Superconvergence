"""Figure gallery: 3D surfaces, contour fields, phase diagram, spaghetti plot.
Reads results/e17..e22 (surfaces.py). Physics/probability-style visuals. -> figures/."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa
import io_utils as io

FIG = io.FIG_DIR
plt.rcParams.update({
    "font.size": 11, "figure.dpi": 170, "savefig.dpi": 170,
    "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})


def _surf(ax, X, Y, Z, cmap, vmin=None, vmax=None):
    """High-quality surface: dense mesh, subtle edge shading, antialiased."""
    return ax.plot_surface(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, rcount=120, ccount=120,
                           linewidth=0.0, antialiased=True, edgecolor="none", alpha=0.97)


def _save3d(fig, name, margins, dpi=180):
    """Save a 3D figure WITHOUT tight bbox -- tight mis-measures 3D axis labels and clips them.
    `margins` = (left, right, bottom, top) chosen to keep the z-axis label inside the canvas."""
    l, r, b, t = margins
    fig.subplots_adjust(left=l, right=r, bottom=b, top=t)
    old = plt.rcParams["savefig.bbox"]; plt.rcParams["savefig.bbox"] = None
    fig.savefig(os.path.join(FIG, name), dpi=dpi)
    plt.rcParams["savefig.bbox"] = old
    io.log(name, "figs.log")


def fig_canyon():
    r = io.load("e17_kl_canyon")
    if not r: return
    lams = np.array(r["lams"]); Ns = np.array(r["Ns"]); Z = np.array(r["logKL"])  # [lam][N]
    plt.figure(figsize=(7.2, 5.0))
    cf = plt.contourf(np.log2(Ns), lams, Z, levels=45, cmap="viridis")
    plt.contour(np.log2(Ns), lams, Z, levels=12, colors="k", linewidths=0.3, alpha=0.35)
    plt.axhline(r["lambda_star"], color="crimson", lw=2.2, ls="--",
                label=rf"$\lambda^\star={r['lambda_star']:.3f}$ (valley floor)")
    plt.colorbar(cf, label=r"$\log_{10}\,\mathrm{KL}(\lambda, N)$")
    plt.xlabel(r"$\log_2 N$"); plt.ylabel(r"churn $\lambda$")
    plt.title(r"$\log_{10}\,\mathrm{KL}(\lambda, N)$")
    plt.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.grid(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_canyon.png")); plt.close()
    io.log("fig_canyon.png", "figs.log")


def fig_coeff_field():
    """Single 3D panel: the sign-change surface C(lambda, s^2) with the zero-locus drawn on it."""
    r = io.load("e18_coeff_field")
    if not r: return
    lams = np.array(r["lams"]); s2s = np.array(r["s2s"]); C = np.array(r["C"])  # [s2][lam]
    Zt = np.arcsinh(C / 4.0)  # arcsinh height tames the wide signed range -> smooth surface
    tmax = float(np.percentile(np.abs(Zt), 99))
    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    L, S = np.meshgrid(lams, s2s)
    _surf(ax, L, S, Zt, "RdBu_r", -tmax, tmax)
    ax.contour(L, S, Zt, levels=[0.0], colors="k", linewidths=3.0)              # zero-locus = lambda*(s^2)
    ax.contourf(L, S, Zt, zdir="z", offset=-tmax * 1.25, levels=30, cmap="RdBu_r",
                vmin=-tmax, vmax=tmax, alpha=0.55)                               # shadow on the floor
    ax.set_box_aspect((1.0, 1.0, 0.72))                                         # give the relief presence
    ax.set_zlim(-tmax * 1.25, tmax)
    ax.set_xlabel(r"churn $\lambda$", labelpad=12); ax.set_ylabel(r"data variance $s^2$", labelpad=12)
    ax.set_zlabel(r"$\mathrm{arcsinh}\,C(\lambda,s^2)$", labelpad=10)
    ax.tick_params(axis="both", pad=2)
    ax.view_init(elev=22, azim=-58)                          # height encodes C; colorbar dropped (redundant)
    _save3d(fig, "fig_coeff_field.png", margins=(0.02, 0.80, 0.08, 0.99))   # z-label on right -> leave right room
    plt.close()


def fig_order_field():
    r = io.load("e19_order_field")
    if not r: return
    lams = np.array(r["lams"]); P = np.clip(np.array(r["order"]), 2.0, 4.0)  # [lam][Npair]
    Nmid = np.array(r.get("Nmid", [(p[0] * p[1]) ** 0.5 for p in r["Npairs"]]))
    x = np.log2(Nmid); lstar = r["lambda_star"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))                  # 2D heatmap (a different type than the 3D figs)
    pcm = ax.pcolormesh(x, lams, P, cmap="viridis", vmin=2, vmax=4, shading="gouraud")
    cs = ax.contour(x, lams, P, levels=[2.5, 3.0, 3.5], colors="white", linewidths=0.7, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax.axhline(lstar, color="#ff2d55", lw=2.4, ls="--", label=rf"$\lambda^\star={lstar:.3f}$ (order $4$ ridge)")
    cb = fig.colorbar(pcm, ax=ax, label=r"measured order $p(\lambda,N)$")
    ax.set_xlabel(r"sampler steps $\log_2 N$"); ax.set_ylabel(r"churn $\lambda$")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9); ax.grid(False)
    plt.tight_layout(); fig.savefig(os.path.join(FIG, "fig_order_field.png"), dpi=170); plt.close()
    io.log("fig_order_field.png", "figs.log")


def fig_goldilocks3d():
    r = io.load("e20_goldilocks_surf")
    if not r: return
    lams = np.array(r["lams"]); epss = np.array(r["epss"]); Z = np.array(r["logKL"])  # [eps][lam]
    L, E = np.meshgrid(lams, epss)
    fig = plt.figure(figsize=(8.4, 6.2))                       # single large 3D panel, no companion
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(L, E, Z, cmap="plasma", rcount=120, ccount=120, linewidth=0,
                           antialiased=True, edgecolor="none", alpha=0.98)
    zfloor = float(Z.min()) - 0.16 * (float(Z.max()) - float(Z.min()))   # shadow just below the surface
    ax.contourf(L, E, Z, zdir="z", offset=zfloor, levels=30, cmap="plasma", alpha=0.4)   # floor shadow (layer)
    ax.set_zlim(zfloor, float(Z.max()))
    ax.set_xlabel(r"churn $\lambda$", labelpad=14); ax.set_ylabel(r"score floor $\varepsilon$", labelpad=14)
    ax.set_zlabel(r"$\log_{10}\mathrm{KL}$", labelpad=10)
    ax.tick_params(axis="both", pad=2)
    ax.view_init(elev=30, azim=-128); ax.set_box_aspect((1.25, 1.0, 0.92))
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.12, aspect=16); cb.set_label(r"$\log_{10}\mathrm{KL}$", fontsize=10)
    _save3d(fig, "fig_goldilocks3d.png", margins=(0.10, 0.88, 0.12, 0.98), dpi=170)  # z-label left, cbar right
    plt.close()


def _vN_grid(lams, N, B=4.0, s2=4.0, T=5.0):
    """Exact terminal variance of the VP Euler--Maruyama sampler, vectorised over churn (float64)."""
    u = lams ** 2                                  # u = lambda^2, vectorised
    dt = T / N
    v = np.full_like(lams, 1.0 + (s2 - 1.0) * np.exp(-B * T))   # v_0 = V(T)
    for k in range(N):
        t = T - k * dt
        V = 1.0 + (s2 - 1.0) * np.exp(-B * t)
        A = -B / 2.0 + (1.0 + u) / 2.0 * B / V
        v = (1.0 - A * dt) ** 2 * v + u * B * dt
    return v


def fig_layers3d():
    """A cool layered 3D sheet: the *signed* terminal variance error v_N - s^2 over (churn, steps).
    The deterministic side undershoots (blue, error < 0), strong churn overshoots (red, error > 0), and the
    surface threads the zero plane along a single ridge -- the cancellation churn lambda*(N). As N grows the
    ridge stiffens into a vertical wall at lambda*, which is exactly where the order jumps to four. Built
    from the exact recursion (no stored result needed); rendered with a floor-projected shadow as a second
    layer and the zero-locus drawn on the sheet."""
    s2, B, T = 4.0, 4.0, 5.0
    lams = np.linspace(0.6, 2.2, 120)
    Ns = np.unique(np.round(2.0 ** np.linspace(7.0, 12.0, 28)).astype(int))  # 128 .. 4096 (fine, smooth)
    x = np.log2(Ns.astype(float))
    Z = np.empty((len(Ns), len(lams)))                                      # log10 KL over (N, lambda)
    for k, N in enumerate(Ns):
        v = _vN_grid(lams, int(N), B, s2, T)
        r = np.clip(v / s2, 1e-300, None)
        kl = 0.5 * (r - 1.0 - np.log(r))
        Z[k] = np.clip(np.log10(np.clip(kl, 1e-13, None)), -13.0, -2.5)     # clip both ends: no corner spike
    X, L = np.meshgrid(x, lams, indexing="ij")                             # X,L,Z all [N][lam]
    fig = plt.figure(figsize=(8.6, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, L, Z, cmap="viridis", rcount=150, ccount=150,
                           linewidth=0, antialiased=True, edgecolor="none", alpha=0.97)
    zfloor = float(Z.min()) - 0.16 * (float(Z.max()) - float(Z.min()))
    ax.contourf(X, L, Z, zdir="z", offset=zfloor, levels=30, cmap="viridis", alpha=0.4)  # floor shadow
    ax.set_zlim(zfloor, float(Z.max()))
    ax.set_box_aspect((1.5, 1.2, 1.02))
    ax.set_xlabel(r"sampler steps  $\log_2 N$", labelpad=14)
    ax.set_ylabel(r"churn $\lambda$", labelpad=12)
    ax.set_zlabel(r"$\log_{10}\mathrm{KL}$", labelpad=10)
    ax.tick_params(axis="both", pad=2)
    ax.view_init(elev=26, azim=-58)
    _save3d(fig, "fig_layers3d.png", margins=(-0.03, 0.93, -0.04, 1.04))   # zoomed in: fill the canvas
    plt.close()


def fig_band():
    """Superconvergence band half-width vs N (E13): the resonance narrows as ~1/N,
    which is why the order spike in fig_resonance sharpens with N."""
    r = io.load("e13_window")
    if not r: return
    rows = r["rows"]; Ns = np.array([x["N"] for x in rows], float)
    hw = np.array([x["half_width"] for x in rows], float)
    plt.figure(figsize=(6.2, 4.2))
    plt.loglog(Ns, hw, "o-", color="#7b3294", lw=2, ms=6, label="band half-width")
    ref = hw[0] * (Ns[0] / Ns)  # N^-1 guide through the first point
    plt.loglog(Ns, ref, "k--", lw=1.2, alpha=0.7, label=r"$\propto N^{-1}$ reference")
    plt.xlabel(r"sampler steps $N$"); plt.ylabel(r"churn band half-width")
    plt.title(r"width of the superconvergence band")
    plt.legend(fontsize=9, framealpha=0.9); plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_band.png")); plt.close()
    io.log("fig_band.png", "figs.log")


def fig_phase():
    r = io.load("e21_phase_diagram")
    if not r: return
    lams = np.array(r["lams"]); s2s = np.array(r["s2s"]); S = np.array(r["signC"])  # [s2][lam]
    plt.figure(figsize=(7.5, 5))
    cmap = matplotlib.colors.ListedColormap(["#4575b4", "#ffffbf", "#d73027"])
    plt.contourf(lams, s2s, S, levels=[-1.5, -0.5, 0.5, 1.5], colors=["#4575b4", "#ffffbf", "#d73027"])
    good = [(l, s) for l, s in zip(r["lambda_star_curve"], s2s) if l]
    if good:
        plt.plot([g[0] for g in good], [g[1] for g in good], "k-", lw=2.5, label=r"$\lambda^\star(s^2)$ ridge")
    plt.axhline(1.0, color="gray", ls=":", lw=1, label=r"$s^2=1$ (no ridge below)")
    plt.xlabel(r"churn $\lambda$"); plt.ylabel(r"data variance $s^2$")
    plt.title(r"Sign of $C(\lambda,\, s^2)$")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_phase.png")); plt.close()
    io.log("fig_phase.png", "figs.log")


def fig_spaghetti():
    r = io.load("e22_aniso_spaghetti")
    if not r: return
    lams = np.array(r["lams"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = cm.viridis(np.linspace(0, 0.85, len(r["spectrum"])))
    for (s, Ci), c, root in zip(r["C_i"].items(), colors, r["per_mode_root"]):
        ax[0].plot(lams, Ci, color=c, label=rf"$s^2={s}$ ($\lambda^\star={root:.2f}$)")
        if root: ax[0].plot(root, 0, "o", color=c)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel(r"churn $\lambda$"); ax[0].set_ylabel(r"$C_i(\lambda)$")
    ax[0].set_title(r"Per-mode coefficients $C_i(\lambda)$")
    ax[0].legend(fontsize=8)
    ax[1].semilogy(lams, r["K"], "k-", lw=2)
    ax[1].axvline(r["lambda_dagger"], color="red", ls="--", label=rf"$\lambda^\dagger={r['lambda_dagger']:.3f}$")
    for root, c in zip(r["per_mode_root"], colors):
        if root: ax[1].axvline(root, color=c, ls=":", alpha=0.6)
    ax[1].set_xlabel(r"churn $\lambda$"); ax[1].set_ylabel(r"$K(\lambda)=\sum_i C_i^2/s_i^4$")
    ax[1].set_title(r"Global error $K(\lambda)=\sum_i C_i^2/s_i^4$")
    ax[1].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_spaghetti.png")); plt.close()
    io.log("fig_spaghetti.png", "figs.log")


def fig_concept():
    """Intro methods figure: a large, clean 3D surface of the measured convergence order p(lambda, N).
    The order sits on a flat plane at two for almost every churn, but a single sharp ridge rises to four
    exactly at lambda*, and the ridge sharpens as N grows. That ridge is stochastic superconvergence:
    one churn level buys a full extra order of accuracy. Rendered big with manual margins (no overlap)."""
    r = io.load("e19_order_field")
    if not r: return
    lams = np.array(r["lams"]); P = np.clip(np.array(r["order"]), 2.0, 4.0)  # [lam][Npair]
    Nmid = np.array(r.get("Nmid", [(p[0] * p[1]) ** 0.5 for p in r["Npairs"]]))
    x = np.log2(Nmid); lstar = r["lambda_star"]
    X, Y = np.meshgrid(x, lams)
    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, P, cmap="viridis", vmin=2, vmax=4, rcount=160, ccount=160,
                           linewidth=0, antialiased=True, edgecolor="none", alpha=0.98)
    j = int(np.argmin(np.abs(lams - lstar)))
    ax.plot(x, np.full_like(x, lstar), P[j] + 0.03, color="#ff2d55", lw=3.2, zorder=10,
            label=r"$\lambda^\star$: order $4$ ridge")
    ax.contourf(X, Y, P, zdir="z", offset=1.85, levels=30, cmap="viridis", vmin=2, vmax=4, alpha=0.5)
    ax.set_zlim(1.85, 4.05)
    ax.set_box_aspect((1.25, 1.0, 0.72))
    ax.set_xlabel(r"sampler steps  $\log_2 N$", labelpad=14)
    ax.set_ylabel(r"churn $\lambda$", labelpad=10)
    ax.set_zlabel(r"order $p$", labelpad=8)
    ax.set_title(r"One churn level lifts the convergence order from two to four", pad=16, fontsize=12.5)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(0.0, 0.93))
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.10, aspect=16)   # pad away from the z-axis label
    cb.set_label(r"order $p(\lambda, N)$", fontsize=10)
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.06, top=0.92)
    fig.savefig(os.path.join(FIG, "fig_concept.png"), dpi=180)
    plt.close()
    io.log("fig_concept.png", "figs.log")


ALL = [fig_concept, fig_canyon, fig_coeff_field, fig_order_field, fig_goldilocks3d, fig_layers3d,
       fig_phase, fig_spaghetti]

if __name__ == "__main__":
    for f in ALL:
        try: f()
        except Exception as e:
            io.log(f"GALLERY ERROR {f.__name__}: {e}", "figs.log")
    print("gallery figures in", FIG)
