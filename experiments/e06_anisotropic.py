"""E6 — Anisotropic no-go + optimal compromise (Thm C).
Degenerate spectrum -> common lambda*, order 4. Non-degenerate -> lambda_dagger, order 2 with
reduced constant. Per-mode roots, global compromise, measured KL orders."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from anisotropic import per_mode_lambda_star, lambda_dagger, total_kl, K_of_lambda
from lambda_star import lambda_star_vp
from metrics import local_order
import io_utils as io

NAME = "e06_anisotropic"


def run(B=4.0, T=5.0, dps=60, N1=1024, N2=8192):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    spectra = {
        "degenerate_2.0": [2.0, 2.0, 2.0, 2.0],
        "two_level": [1.5, 1.5, 4.0, 4.0],
        "geometric": [1.5, 2.25, 3.375, 5.0625],
        "wide": [1.2, 2.0, 8.0, 32.0],
    }
    out = {}
    for name, spec in spectra.items():
        per_ls = [None if l is None else float(l) for l in per_mode_lambda_star(B, spec, T)]
        ld = lambda_dagger(B, spec, T)
        # order at the compromise churn
        k1 = total_kl(B, ld, spec, T, N1); k2 = total_kl(B, ld, spec, T, N2)
        order_dagger = local_order(k1, k2, N1, N2)
        # order at a single mode's lambda* (the mean) — should NOT super-converge unless degenerate
        order_dagger_dict = {"lambda_dagger": float(ld), "order_at_dagger": order_dagger}
        degenerate = len(set(spec)) == 1
        if degenerate:
            ls = lambda_star_vp(B, spec[0], T)
            kk1 = total_kl(B, ls, spec, T, N1); kk2 = total_kl(B, ls, spec, T, N2)
            order_dagger_dict["order_at_common_lstar"] = local_order(kk1, kk2, N1, N2)
        out[name] = {"spectrum": spec, "per_mode_lambda_star": per_ls,
                     "degenerate": degenerate, "K_at_dagger": float(K_of_lambda(B, ld, spec, T)),
                     **order_dagger_dict}
        io.log(f"  {name:16s} deg={degenerate} per-mode lam*={[round(x,4) if x else None for x in per_ls]} "
               f"lam_dagger={float(ld):.4f} order@dagger={order_dagger:.3f}"
               + (f" order@common*={order_dagger_dict.get('order_at_common_lstar'):.3f}" if degenerate else ""))
    res = {"config": {"B": B, "T": T, "dps": dps, "N1": N1, "N2": N2}, "spectra": out}
    io.save(NAME, res)
    io.log(f"{NAME} done.")
    return res


if __name__ == "__main__":
    run()
