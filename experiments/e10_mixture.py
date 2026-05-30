"""E10 — Non-Gaussian boundary (honesty arm). For a 2-mode Gaussian mixture the exact score
is nonlinear, the iterate is not Gaussian, and super-convergence does NOT occur: KL/W2 order
stays ~2 with a bias-minimising churn near 1. Heavy Monte-Carlo, multi-seed, debiased W2^2.
Resumable: checkpoints after each churn value.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
from gmm import sample, target_sample
from metrics import w2sq_emp_1d, local_order
import io_utils as io

NAME = "e10_mixture"


def run(mu=2.0, s2=1.0, B=4.0, T=5.0, P=3_000_000, seeds=6,
        Ns=(16, 32, 64, 128, 256, 512), lams=(0.0, 0.5, 1.0, 1.27)):
    partial = io.load(NAME) or {"config": {"mu": mu, "s2": s2, "B": B, "T": T, "P": P,
                                            "seeds": seeds, "Ns": list(Ns), "lams": list(lams)},
                                "by_lambda": {}}
    for lam in lams:
        key = f"{lam}"
        if key in partial["by_lambda"]:
            io.log(f"{NAME}: lam={lam} done, skip"); continue
        rows = []
        for N in Ns:
            w2s = []
            for sd in range(seeds):
                rng = np.random.default_rng(1000 * sd + 7)
                x = sample(N, T, B, lam, s2, mu, P, rng)
                ref = target_sample(P, s2, mu, rng)
                floor = w2sq_emp_1d(target_sample(P, s2, mu, rng), target_sample(P, s2, mu, rng))
                w2s.append(w2sq_emp_1d(x, ref) - floor)   # debiased
            w2s = np.array(w2s)
            rows.append({"N": N, "w2_mean": float(w2s.mean()), "w2_std": float(w2s.std()),
                         "floor_subtracted": True})
            io.log(f"  lam={lam} N={N}: W2^2(debiased)={w2s.mean():.3e} +/- {w2s.std():.1e}")
        # measured order from the largest two N (where the law is asymptotic)
        order = local_order(max(rows[-2]["w2_mean"], 1e-12), max(rows[-1]["w2_mean"], 1e-12),
                            Ns[-2], Ns[-1])
        partial["by_lambda"][key] = {"lam": lam, "rows": rows, "tail_order": order}
        io.save(NAME, partial)   # checkpoint
        io.log(f"  lam={lam} tail order ~ {order:.3f} (expect ~2, NOT 4)")
    # bias-minimising churn at the largest N
    bigN = Ns[-1]
    best = min(partial["by_lambda"].values(),
               key=lambda d: [r for r in d["rows"] if r["N"] == bigN][0]["w2_mean"])
    partial["bias_min_lambda_at_bigN"] = best["lam"]
    io.save(NAME, partial)
    io.log(f"{NAME} done. bias-min churn at N={bigN}: lam={best['lam']}")
    return partial


if __name__ == "__main__":
    run()
