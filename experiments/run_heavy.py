"""Heavy stochastic arms (Stage 2): E10 (GMM Monte-Carlo) then E11 (learned score).
Resumable; each checkpoints internally. Launched separately from the deterministic batch."""
import sys, os, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
sys.path.insert(0, os.path.dirname(__file__))
import io_utils as io
import e10_mixture, e11_learned

REG = [("e10", e10_mixture.run), ("e11", e11_learned.run)]


def main():
    io.log("=" * 60, "heavy.log")
    io.log("run_heavy: E10 mixture + E11 learned score", "heavy.log")
    for key, fn in REG:
        t = time.time()
        try:
            fn(); io.log(f"[{key}] OK in {time.time()-t:.1f}s", "heavy.log")
        except Exception as e:
            io.log(f"[{key}] ERROR: {e}\n{traceback.format_exc()}", "heavy.log")
    io.log("run_heavy DONE", "heavy.log")


if __name__ == "__main__":
    main()
