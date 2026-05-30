"""Master sequential runner for the deterministic batch (Stage 1). Resumable: each experiment
skips if its results/*.json already exists. Logs timing to logs/run.log. Heavy stochastic
arms (E10 GMM, E11 learned score) are launched separately.

Usage:
  python3 experiments/run_all.py            # run all deterministic experiments in order
  python3 experiments/run_all.py e03 e04    # run a subset
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
sys.path.insert(0, os.path.dirname(__file__))
import io_utils as io

import e01_headline, e02_coefficient, e03_resonance, e04_lambda_star_law
import e05_invariance, e06_anisotropic, e07_goldilocks, e08_universality
import e09_integrator, e12_sensitivity
import e34_uniqueness_cert, e35_edm_bridge, e36_edm_toy2d, e39_edm_mnist, e60_churn_law
import e61_sampler_shootout, e62_coarse_churn, e63_integrator_root

REGISTRY = {
    "e01": e01_headline.run, "e02": e02_coefficient.run, "e03": e03_resonance.run,
    "e04": e04_lambda_star_law.run, "e05": e05_invariance.run, "e06": e06_anisotropic.run,
    "e07": e07_goldilocks.run, "e08": e08_universality.run, "e09": e09_integrator.run,
    "e12": e12_sensitivity.run,
    "e34": e34_uniqueness_cert.run, "e35": e35_edm_bridge.run,
    "e36": e36_edm_toy2d.run, "e39": e39_edm_mnist.run, "e60": e60_churn_law.run,
    "e61": e61_sampler_shootout.run, "e62": e62_coarse_churn.run,
    "e63": e63_integrator_root.run,
}
ORDER = ["e01", "e02", "e03", "e04", "e05", "e06", "e07", "e08", "e09", "e12", "e34", "e35", "e36"]


def main(which=None):
    which = which or ORDER
    io.log("=" * 60, "run.log")
    io.log(f"run_all: {which}", "run.log")
    t0 = time.time()
    for key in which:
        fn = REGISTRY[key]
        t = time.time()
        try:
            fn()
            io.log(f"[{key}] OK in {time.time()-t:.1f}s")
        except Exception as e:
            io.log(f"[{key}] ERROR: {e}\n{traceback.format_exc()}")
    io.log(f"run_all DONE in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    main(args if args else None)
