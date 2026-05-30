"""Single sequential driver for the four corrected experiments: E02, E07, E09, then E11.
One process (no parallelism) to avoid CPU starvation. Resumable via each experiment's own
skip/checkpoint logic."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
sys.path.insert(0, os.path.dirname(__file__))
import io_utils as io
import e02_coefficient, e07_goldilocks, e09_integrator, e11_learned

io.log("rerun_fixed: E02,E07,E09,E11 (single sequential process)", "run.log")
for key, fn in [("e02", e02_coefficient.run), ("e07", e07_goldilocks.run),
                ("e09", e09_integrator.run), ("e11", e11_learned.run)]:
    t = time.time()
    try:
        fn(); io.log(f"[{key}] OK in {time.time()-t:.1f}s")
    except Exception as e:
        import traceback; io.log(f"[{key}] ERROR: {e}\n{traceback.format_exc()}")
io.log("rerun_fixed DONE")
