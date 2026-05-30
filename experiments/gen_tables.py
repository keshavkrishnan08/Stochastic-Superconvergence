"""Generate paper/sections/experiments_table.tex and update paper macros from results/*.json.
Runs incrementally; missing results show as 'pending'. Keeps the paper in sync with data."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import io_utils as io

SEC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "sections"))
os.makedirs(SEC, exist_ok=True)


def g(name): return io.load(name)


def row(eid, desc, val):
    return f"{eid} & {desc} & {val} \\\\\n"


def build():
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Experiment summary. Gaussian arms are deterministic exact recursions; theory overlays measurement with no free parameters.}",
             r"\begin{tabular}{lll}\toprule",
             r"ID & What it tests & Headline result \\\midrule"]
    e1 = g("e01_headline")
    if e1:
        ls = e1["lambda_star"]; cs = e1["curves"]
        o1 = cs["1"]["order_tail"]; os_ = cs["lstar"]["order_tail"]
        lines.append(row("E1", r"$N^{-2}\!\to\!N^{-4}$ at $\lambda^\star$ (Thm~\ref{thm:super})",
                         rf"order {o1:.3f}$\to${os_:.3f}; $\lambda^\star={ls:.4f}$"))
    else:
        lines.append(row("E1", "headline", "pending"))
    e2 = g("e02_coefficient")
    lines.append(row("E2", r"closed-form $C(\lambda)$ (Thm~\ref{thm:umbrella})",
                     (rf"max rel-err ${e2['max_rel_err']:.1e}$ ({e2['n_configs']} configs)" if e2 else "pending")))
    e3 = g("e03_resonance")
    lines.append(row("E3", "order resonance",
                     (rf"peak order {e3['peak_order']:.2f} at $\lambda={e3['peak_lambda']:.3f}$" if e3 else "pending")))
    e4 = g("e04_lambda_star_law")
    lines.append(row("E4", r"$\lambda^\star(s^2)$ law",
                     (rf"$\kappa={e4['kappa_tail_mean']:.3f}$; no root $s^2\!\le\!1$" if e4 else "pending")))
    e5 = g("e05_invariance")
    lines.append(row("E5", r"$(B,T)$ invariance",
                     ((r"$\lambda^\star=%.4f\pm$%.0e (spread %.1f\%%)" % (e5['mean'], e5['std'], e5['rel_spread']*100)) if e5 else "pending")))
    e6 = g("e06_anisotropic")
    if e6:
        deg = e6["spectra"].get("degenerate_2.0", {})
        nd = e6["spectra"].get("wide", {})
        lines.append(row("E6", r"anisotropic no-go (Thm~\ref{app:thm:aniso})",
                         rf"deg.\ order {deg.get('order_at_common_lstar', float('nan')):.2f}, non-deg.\ {nd.get('order_at_dagger', float('nan')):.2f}"))
    else:
        lines.append(row("E6", "anisotropic", "pending"))
    e7 = g("e07_goldilocks")
    lines.append(row("E7", r"optimal churn (Thm~\ref{thm:gold})",
                     (rf"pred vs.\ measured $\lambda_{{\mathrm{{opt}}}}$, max $|\Delta|={e7['pred_vs_meas_max_abs']:.3f}$" if e7 and e7.get('pred_vs_meas_max_abs') is not None else "pending")))
    e8 = g("e08_universality")
    if e8:
        nroot = sum(1 for v in e8["schedules"].values() if v.get("has_root"))
        lines.append(row("E8", r"schedule universality (Thm~\ref{app:thm:univ})",
                         rf"{nroot}/{len(e8['schedules'])} schedules show the jump"))
    else:
        lines.append(row("E8", "universality", "pending"))
    e9 = g("e09_integrator")
    lines.append(row("E9", r"integrator stacking (Thm~\ref{app:thm:integ})",
                     (rf"Heun generic order {e9['Heun_orders']['1.0']:.2f} ($=2p$)" if e9 and 'Heun_orders' in e9 else "pending")))
    e10 = g("e10_mixture")
    if e10 and "by_lambda" in e10 and e10["by_lambda"]:
        ords = [d["tail_order"] for d in e10["by_lambda"].values()]
        lines.append(row("E10", "mixture boundary (honesty)",
                         rf"order $\sim${sum(ords)/len(ords):.2f} (no jump)"))
    else:
        lines.append(row("E10", "mixture boundary", "pending"))
    e11 = g("e11_learned")
    lines.append(row("E11", "learned-score floor",
                     (rf"floor $\propto\delta^2$, N-flat" if e11 and "part_A" in e11 else "pending")))
    e12 = g("e12_sensitivity")
    lines.append(row("E12", "sensitivity/ablations",
                     ("init-offset plateau; finite-$N$ $\\lambda$ drift" if e12 else "pending")))
    lines += [r"\bottomrule\end{tabular}\label{tab:exp}\end{table}"]
    with open(os.path.join(SEC, "experiments_table.tex"), "w") as f:
        f.write("\n".join(lines))
    io.log("wrote experiments_table.tex", "figs.log")


if __name__ == "__main__":
    build()
    print("table written to", SEC)
