"""Wave-2 ablations (E13–E16): window width, rotated covariance, EDM schedule, kappa constant.
Single sequential process. Run after rerun_fixed completes."""
import sys, os, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
sys.path.insert(0, os.path.dirname(__file__))
import io_utils as io
import e13_window, e14_rotated, e15_edm_schedule, e16_kappa

io.log("run_more: E13 window, E14 rotated, E15 EDM-schedule, E16 kappa", "run.log")
for key, fn in [("e13", e13_window.run), ("e14", e14_rotated.run),
                ("e15", e15_edm_schedule.run), ("e16", e16_kappa.run)]:
    t = time.time()
    try:
        fn(); io.log(f"[{key}] OK in {time.time()-t:.1f}s")
    except Exception as e:
        io.log(f"[{key}] ERROR: {e}\n{traceback.format_exc()}")
io.log("run_more DONE")
