"""E37 - Anisotropic superconvergence via per-eigenmode churn (the fix for the scalar no-go).

The scalar no-go (E6/E26) says a single global churn cannot superconverge a non-degenerate spectrum,
because each eigenmode has its own cancellation root lambda*(s_i^2). The resolution is a DIAGONAL churn:
inject stochasticity per eigen-coordinate, lambda_i = lambda*(s_i^2). Then every mode's leading
coefficient C_i is annihilated simultaneously, so the aggregate terminal KL converges at order N^-4 for
ANY diagonal (hence, by rotation, any) Gaussian covariance. This turns the impossibility into a positive
result and removes the isotropy restriction. Exact, deterministic, no sampling.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
import io_utils as io
from diffusion import vp_const, set_dps
from recursion import v_terminal_closed
from lambda_star import lambda_star_vp
from metrics import kl_gauss


def run(B=4.0, T=5.0, dps=50):
    set_dps(dps)
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    spectra = {
        "two_mode": [1.5, 4.0],
        "three_mode": [1.5, 4.0, 7.0],
        "wide_five": [1.3, 2.2, 3.7, 6.0, 9.0],
    }
    def order(kls):
        return [float(mp.log(kls[i] / kls[i + 1]) / mp.log(mp.mpf(Ns[i + 1]) / Ns[i])) for i in range(len(Ns) - 1)]
    out = {}
    for name, spec in spectra.items():
        roots = [lambda_star_vp(B, s, T) for s in spec]
        def agg(churns):
            return [sum(kl_gauss(v_terminal_closed(vp_const(B, s, T), N, churns[i] ** 2), s)
                        for i, s in enumerate(spec)) for N in Ns]
        per = agg(roots)                                   # diagonal (per-eigenmode) churn
        lam_g = sum(roots) / len(roots)
        glob = agg([lam_g] * len(spec))                    # best single scalar churn
        out[name] = {"spectrum": spec, "per_mode_roots": [mp.nstr(r, 8) for r in roots],
                     "per_mode_order_tail": order(per)[-1], "global_order_tail": order(glob)[-1],
                     "per_mode_order": order(per), "global_order": order(glob)}
        io.log(f"  e37 {name}: per-mode order={order(per)[-1]:.3f}  single-churn order={order(glob)[-1]:.3f}")
    io.save("e37_aniso_super", {"config": {"B": B, "T": T, "Ns": Ns, "dps": dps}, "spectra": out})
    io.log("e37_aniso_super DONE")


if __name__ == "__main__":
    run()
