"""E5 — Invariance of lambda* to (B,T) at fixed s2 (Thm A). lambda* depends on the SHAPE of V,
not its time-scaling, so it is essentially flat over B and T."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import set_dps
from lambda_star import lambda_star_vp
import io_utils as io

NAME = "e05_invariance"


def run(s2=2.0, dps=35):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    Bs = np.linspace(1.0, 16.0, 9)
    Ts = np.linspace(2.0, 40.0, 9)
    grid = []
    vals = []
    for B in Bs:
        row = []
        for T in Ts:
            ls = lambda_star_vp(float(B), s2, float(T))
            v = None if ls is None else float(ls)
            row.append(v)
            if v is not None: vals.append(v)
        grid.append(row)
    vals = np.array(vals)
    res = {"config": {"s2": s2, "dps": dps}, "Bs": Bs.tolist(), "Ts": Ts.tolist(),
           "lambda_star_grid": grid,
           "mean": float(vals.mean()), "std": float(vals.std()),
           "min": float(vals.min()), "max": float(vals.max()),
           "rel_spread": float((vals.max() - vals.min()) / vals.mean())}
    io.save(NAME, res)
    io.log(f"{NAME}: lambda* over B in[1,16] x T in[2,40]: mean={vals.mean():.5f} "
           f"std={vals.std():.2e} rel-spread={(vals.max()-vals.min())/vals.mean():.2%}")
    return res


if __name__ == "__main__":
    run()
