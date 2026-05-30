"""Generate paper/sections/extended_tables.tex (supplementary) from results/*.json.
Emits NeurIPS-style detailed tables: anchor exactness, full KL-vs-N curves, the lambda*(s2)
law, anisotropic spectra, the floor-shift table, schedule universality, Heun stacking,
mixture boundary, learned-score floors."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import io_utils as io

SEC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "sections"))


def L(n): return io.load(n)


def tbl_e1(o):
    r = L("e01_headline")
    if not r: return
    Ns = r["Ns"]; cs = r["curves"]
    o.append(r"\begin{table}[h]\centering\small\caption{E1: terminal KL vs steps $N$ (exact recursion, 120-digit). Generic churns track $N^{-2}$; $\lambda^\star$ tracks $N^{-4}$.}")
    o.append(r"\begin{tabular}{r" + "r" * 4 + r"}\toprule")
    o.append(r"$N$ & $\lambda{=}0$ & $\lambda{=}1$ & $\lambda{=}2$ & $\lambda^\star{=}" + f"{r['lambda_star']:.4f}" + r"$\\\midrule")
    for i, N in enumerate(Ns):
        if N not in (16, 64, 256, 1024, 4096, 16384, 65536): continue
        row = f"{N}"
        for tag in ["0", "1", "2", "lstar"]:
            row += f" & {cs[tag]['KL'][i]:.2e}"
        o.append(row + r"\\")
    o.append(r"\midrule order (tail) & " + " & ".join(f"{cs[t]['order_tail']:.3f}" for t in ["0","1","2","lstar"]) + r"\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e1}\end{table}")


def tbl_anchor(o):
    r = L("e02_coefficient")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E2: closed-form $C(\lambda)$ vs Richardson estimate across a $(s^2,B,T,\lambda)$ grid; elementary anchors $C(0),C(1)$ vs quadrature.}")
    o.append(r"\begin{tabular}{lr}\toprule Quantity & Value\\\midrule")
    o.append(rf"configs & {r['n_configs']}\\")
    o.append(rf"max rel-err ($|C|>10^{{-2}}$) & ${r['max_rel_err']:.1e}$\\")
    o.append(rf"max abs-err & ${r['max_abs_err']:.1e}$\\")
    o.append(rf"anchor rel-err & ${r['max_anchor_rel_err']:.1e}$\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:anchor}\end{table}")


def tbl_e4(o):
    r = L("e04_lambda_star_law")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E4: $\lambda^\star(s^2)$ and the ratio $\lambda^\star/\sqrt{s^2}$, which rises toward the limit $\kappa\approx1.207$ pinned at larger $s^2$ by E29. No positive root for $s^2\le1$.}")
    o.append(r"\begin{tabular}{rrr}\toprule $s^2$ & $\lambda^\star$ & $\lambda^\star/\sqrt{s^2}$\\\midrule")
    sel = [x for x in r["rows"] if x["lambda_star"]]
    pick = sel[::max(1, len(sel) // 10)]
    for x in pick:
        o.append(rf"{x['s2']:.2f} & {x['lambda_star']:.4f} & {x['ratio']:.4f}\\")
    o.append(rf"\midrule ratio at largest $s^2$ & \multicolumn{{2}}{{r}}{{{r['kappa_tail_mean']:.4f}}}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e4}\end{table}")


def tbl_e6(o):
    r = L("e06_anisotropic")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E6: anisotropic no-go. Degenerate spectra super-converge (order 4); non-degenerate give order 2 at the compromise churn $\lambda^\dagger$.}")
    o.append(r"\begin{tabular}{llrr}\toprule spectrum & degenerate & $\lambda^\dagger$ & KL order\\\midrule")
    for name, d in r["spectra"].items():
        oo = d.get("order_at_common_lstar", d.get("order_at_dagger"))
        o.append(rf"{name.replace('_',' ')} & {d['degenerate']} & {d['lambda_dagger']:.4f} & {oo:.3f}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e6}\end{table}")


def tbl_e7(o):
    r = L("e07_goldilocks")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E7: floor-shifted superconvergence (Thm B). Predicted optimal churn $\lambda_{\mathrm{opt}}$ (argmin$|C/N+D\varepsilon^2|$) vs measured argmin-KL; dip = KL$(\lambda^\star)$/KL$(\lambda_{\mathrm{opt}})$.}")
    o.append(r"\begin{tabular}{rrrrlr}\toprule $\varepsilon$ & $N$ & pred & meas & root? & dip\\\midrule")
    for row in r["opt_rows"]:
        if row["N"] not in (256, 1024, 2048): continue
        o.append(rf"{row['eps']} & {row['N']} & {row['pred_lambda_opt']:.4f} & {row['meas_lambda_opt']:.4f} & {row['is_cancellation_root']} & {row['dip_vs_lstar']:.2f}\\")
    o.append(rf"\midrule \multicolumn{{4}}{{l}}{{max $|$pred$-$meas$|$}} & \multicolumn{{2}}{{r}}{{{r['pred_vs_meas_max_abs']:.4f}}}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e7}\end{table}")


def tbl_e8(o):
    r = L("e08_universality")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E8: schedule universality. A superconvergence root exists iff $C(0)<0$; where it does, order jumps to 4.}")
    o.append(r"\begin{tabular}{lrlrr}\toprule schedule & $C(0)$ & root? & $\lambda^\star$ & order@$\lambda^\star$\\\midrule")
    for name, d in r["schedules"].items():
        ls = f"{d['lambda_star']:.4f}" if d.get("lambda_star") else "--"
        oo = f"{d['order_at_star']:.3f}" if d.get("order_at_star") else "--"
        o.append(rf"{name.replace('_',' ')} & {d['C0']:.3f} & {d['has_root']} & {ls} & {oo}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e8}\end{table}")


def tbl_e9(o):
    r = L("e09_integrator")
    if not r: return
    hr = r.get("heun_root"); hro = r.get("heun_root_order")
    o.append(r"\begin{table}[h]\centering\small\caption{E9: integrator stacking. EM ($p{=}1$) gives KL order 2 generically, 4 at $\lambda^\star$; Heun ($p{=}2$) gives order 4 generically.}")
    o.append(r"\begin{tabular}{lrrr}\toprule & $\lambda{=}0$ & $\lambda{=}1$ & $\lambda^\star_{\mathrm{EM}}$\\\midrule")
    em = r["EM_orders"]; he = r["Heun_orders"]; ls = str(r["lambda_star_EM"])
    o.append(rf"EM order & {em['0.0']:.3f} & {em['1.0']:.3f} & {em[ls]:.3f}\\")
    o.append(rf"Heun order & {he['0.0']:.3f} & {he['1.0']:.3f} & {he[ls]:.3f}\\")
    o.append(r"\midrule \multicolumn{4}{l}{Heun $C_2$ root: " + (f"{hr:.3f} (order {hro:.2f})" if hr else "none in $[0,5]$") + r"}\\")
    o.append(rf"\multicolumn{{4}}{{l}}{{consistency (Lemma 2) KL $={r['consistency_continuous_KL']:.1e}$}}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e9}\end{table}")


def tbl_e10(o):
    r = L("e10_mixture")
    if not r or "by_lambda" not in r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E10: Gaussian-mixture boundary (Monte-Carlo, debiased $W_2^2$). No superconvergence: order stays $\sim2$ for every churn; bias-minimising churn near 1.}")
    o.append(r"\begin{tabular}{rr}\toprule $\lambda$ & tail order\\\midrule")
    for k, d in r["by_lambda"].items():
        o.append(rf"{d['lam']} & {d['tail_order']:.3f}\\")
    o.append(rf"\midrule bias-min $\lambda$ & {r.get('bias_min_lambda_at_bigN','--')}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e10}\end{table}")


def tbl_e11(o):
    r = L("e11_learned")
    if not r: return
    if "part_A" in r:
        o.append(r"\begin{table}[h]\centering\small\caption{E11A: score miscalibration $\delta$ produces an $N$-independent KL floor $\propto\delta^2$ (deterministic, exact). Floor at $N{=}16384$.}")
        o.append(r"\begin{tabular}{rrr}\toprule $\delta$ & floor KL & tail order\\\midrule")
        for k, d in r["part_A"].items():
            o.append(rf"{d['delta']} & {d['floor']:.2e} & {d['tail_order']:.3f}\\")
        o.append(r"\bottomrule\end{tabular}\label{app:tab:e11a}\end{table}")
    pb = [x for x in r.get("part_B", []) if x.get("residual_rmse") == x.get("residual_rmse")]
    if pb:
        o.append(r"A genuinely trained network shows the same floor, now controlled by the residual score "
                 r"RMSE rather than a hand-set $\delta$ (Table~\ref{app:tab:e11b}).")
        o.append(r"\begin{table}[h]\centering\small\caption{E11B: trained score MLP. Larger capacity / more epochs $\to$ smaller residual score RMSE $\to$ lower KL floor at $\lambda^\star$.}")
        o.append(r"\begin{tabular}{rrrrr}\toprule width & depth & epochs & resid RMSE & floor KL\\\midrule")
        for x in sorted(pb, key=lambda z: (z["width"], z["depth"], z["epochs"])):
            o.append(rf"{x['width']} & {x['depth']} & {x['epochs']} & {x['residual_rmse']:.4f} & {x['floor_KL']:.2e}\\")
        o.append(r"\bottomrule\end{tabular}\label{app:tab:e11b}\end{table}")


def tbl_e5(o):
    r = L("e05_invariance")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E5: invariance of $\lambda^\star$ to the rate $B$ and horizon $T$ at fixed $s^2$.}")
    o.append(r"\begin{tabular}{lr}\toprule Quantity & Value\\\midrule")
    o.append(rf"mean $\lambda^\star$ & {r['mean']:.5f}\\")
    o.append(rf"std over grid & ${r['std']:.2e}$\\")
    o.append(r"relative spread (max$-$min)/mean & %.2f\%%\\" % (r["rel_spread"] * 100))
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e5}\end{table}")


def tbl_e12(o):
    r = L("e12_sensitivity")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E12: sensitivity ablations. Init-offset $e_0$ converts the $N^{-4}$ descent to a plateau; the finite-$N$ optimal churn approaches $\lambda^\star$.}")
    o.append(r"\begin{tabular}{rrr}\toprule $e_0$ & tail order & predicted plateau KL\\\midrule")
    for k, d in r["init_offset"].items():
        o.append(rf"{d['e0']} & {d['tail_order']:.3f} & {d['pred_plateau']:.2e}\\")
    fn_bits = ", ".join("$N{=}%d{:}\\,%.4f$" % (x["N"], x["lambda_opt"]) for x in r["finite_N_opt"][::2])
    o.append(r"\midrule\multicolumn{3}{l}{finite-$N$ optimal churn: " + fn_bits +
             (r"; $\lambda^\star=%.4f$}\\" % r["lambda_star"]))
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e12}\end{table}")


def tbl_e13(o):
    r = L("e13_window")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E13: superconvergence band half-width versus $N$. The band where KL$\le10\times$KL$(\lambda^\star)$ shrinks as the order resonance sharpens.}")
    o.append(r"\begin{tabular}{rrr}\toprule $N$ & KL$(\lambda^\star)$ & half-width\\\midrule")
    for row in r["rows"]:
        o.append(rf"{row['N']} & {row['kl_star']:.2e} & {row['half_width']:.4e}\\")
    o.append(rf"\midrule half-width $\sim N^{{{r['halfwidth_slope']:.2f}}}$\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e13}\end{table}")


def tbl_e14(o):
    r = L("e14_rotated")
    if not r: return
    eigs = r["config"]["eigs"]; th = r["config"]["theta"]
    o.append(r"\begin{table}[h]\centering\small\caption{Full-covariance (rotated) target with eigenvalues "
             rf"$({eigs[0]},{eigs[1]})$ rotated by $\theta={th}$. The Monte-Carlo terminal-covariance "
             r"eigenvalues match the per-eigenmode recursion at every churn, confirming that the decomposition "
             r"acts mode by mode in the eigenbasis and is rotation-invariant.}")
    o.append(r"\begin{tabular}{rcccccc}\toprule"
             r" & \multicolumn{2}{c}{eigenvalue 1} & \multicolumn{2}{c}{eigenvalue 2} & \\"
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
             r"$\lambda$ & measured & per-mode & measured & per-mode & max rel.\ err.\\\midrule")
    for row in r["rows"]:
        m, p = row["meas_eig"], row["pred_eig"]
        o.append(rf"{row['lam']} & {m[0]:.4f} & {p[0]:.4f} & {m[1]:.4f} & {p[1]:.4f} & {row['max_rel']:.2e}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e14}\end{table}")


def tbl_e15(o):
    r = L("e15_edm_schedule")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E15: EDM-like variance-exploding schedule ($\sigma\!\sim\!t$). A superconvergence root and the order jump appear wherever $C(0)<0$.}")
    o.append(r"\begin{tabular}{lrlr}\toprule case & $C(0)$ & root? & order@$\lambda^\star$\\\midrule")
    for k, d in r["cases"].items():
        oo = f"{d['order_at_star']:.3f}" if d.get("order_at_star") else "--"
        o.append(rf"{k} & {d['C0']:+.3f} & {d['has_root']} & {oo}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e15}\end{table}")


def tbl_e16(o):
    r = L("e16_kappa")
    if not r: return
    o.append(r"\begin{table}[h]\centering\small\caption{E16: the $\kappa$ constant. Closed-form root of the limiting equation versus the measured tail ratio $\lambda^\star/\sqrt{s^2}$ at large $s^2$.}")
    o.append(r"\begin{tabular}{lr}\toprule Quantity & Value\\\midrule")
    kc = r.get("kappa_closed_form")
    o.append(rf"closed-form $\kappa$ (root of limiting eqn) & {kc:.4f}\\" if kc else r"closed-form $\kappa$ & --\\")
    for m in r["measured_ratio"]:
        if m["ratio"]: o.append(rf"$\lambda^\star/\sqrt{{s^2}}$ at $s^2={m['s2']:.0f}$ & {m['ratio']:.4f}\\")
    o.append(r"\bottomrule\end{tabular}\label{app:tab:e16}\end{table}")


def build():
    # Each table is pinned [H] and preceded by a one-line lead-in, so no two tables are ever adjacent
    # without intervening text. Grouped into thematic subsections. tbl_e13 omitted (Figure fig_band);
    # tbl_e16 omitted (kappa covered by E4 and the E29 sweep + figure).
    groups = [
        (r"\subsection{Convergence and the coefficient}", [
            (r"The terminal KL falls as $N^{-2}$ at the generic churns and as $N^{-4}$ at $\lambda^\star$ "
             r"(Table~\ref{app:tab:e1}).", tbl_e1),
            (r"The closed-form coefficient $C(\lambda)$ matches an independent Richardson estimate, and its "
             r"elementary anchors $C(0),C(1)$ match quadrature, to twelve figures (Table~\ref{app:tab:anchor}).", tbl_anchor),
        ]),
        (r"\subsection{The cancellation churn}", [
            (r"The cancellation churn grows as $\kappa\sqrt{s^2}$ and has no positive root below $s^2=1$ "
             r"(Table~\ref{app:tab:e4}).", tbl_e4),
            (r"It is nearly invariant to the rate $B$ and horizon $T$ at fixed $s^2$ (Table~\ref{app:tab:e5}).", tbl_e5),
        ]),
        (r"\subsection{Regimes of the decomposition}", [
            (r"An anisotropic spectrum loses global superconvergence, dropping to order two at the compromise "
             r"churn $\lambda^\dagger$ (Table~\ref{app:tab:e6}).", tbl_e6),
            (r"The floor-shifted optimal churn matches the prediction $\arg\min_\lambda|C/N+D\varepsilon^2|$ "
             r"(Table~\ref{app:tab:e7}).", tbl_e7),
            (r"Every linear schedule with $C(0)<0$ exhibits the root and the order jump (Table~\ref{app:tab:e8}).", tbl_e8),
            (r"A second-order Heun step reaches generic order four (Table~\ref{app:tab:e9}).", tbl_e9),
        ]),
        (r"\subsection{Boundaries of the effect}", [
            (r"A non-Gaussian Gaussian-mixture target keeps order two for every churn (Table~\ref{app:tab:e10}).", tbl_e10),
            (r"A miscalibrated or trained score sets an $N$-independent floor that grows with the score error "
             r"(Table~\ref{app:tab:e11a}).", tbl_e11),
            (r"The sensitivity ablations follow the transient and $O(1/N)$ corrections of the decomposition "
             r"(Table~\ref{app:tab:e12}).", tbl_e12),
        ]),
        (r"\subsection{Additional robustness}", [
            (r"A rotated full-covariance target matches the per-eigenmode recursion, confirming rotation "
             r"invariance (Table~\ref{app:tab:e14}).", tbl_e14),
            (r"An EDM-like variance-exploding schedule shows the root and the order jump wherever $C(0)<0$ "
             r"(Table~\ref{app:tab:e15}).", tbl_e15),
        ]),
    ]
    o = []
    for header, items in groups:
        o.append(header)
        for lead, fn in items:
            o.append(lead)
            try:
                fn(o)
            except Exception as e:
                o.append(f"% ERROR {fn.__name__}: {e}")
    text = "\n".join(o).replace(r"\begin{table}[h]", r"\begin{table}[H]")  # pin: never stack
    with open(os.path.join(SEC, "extended_tables.tex"), "w") as f:
        f.write(text)
    io.log("wrote extended_tables.tex", "figs.log")


if __name__ == "__main__":
    build()
    print("extended tables written")
