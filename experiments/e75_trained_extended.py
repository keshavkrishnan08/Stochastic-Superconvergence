"""E75 -- extend the trained-score floor sweep (e11 part_B) into the regime the mandate flagged: more epochs
and more capacity, to drive the residual score RMSE below the previous best (~0.009) and fill in the
floor-vs-RMSE law with cleaner, lower-error points. Reuses e11's DSM trainer and learned-score sampler
verbatim, so the floor measurement is identical and the new points sit on the same axes as Fig 6c /
the trained-network floor. Heavy CPU (long training); resumable per config; saves incrementally.
"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from diffusion import set_dps
from learned_score import train_dsm, sample_learned
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e75_trained_extended"


def run(s2=2.0, B=4.0, T=5.0):
    set_dps(40)
    lstar = float(lambda_star_vp(B, s2, T))
    # configs beyond e11's grid: longer training (lower RMSE) and more capacity
    configs = [
        {"width": 256, "depth": 4, "epochs": 16000},
        {"width": 256, "depth": 4, "epochs": 32000},
        {"width": 512, "depth": 4, "epochs": 8000},
        {"width": 512, "depth": 4, "epochs": 16000},
        {"width": 256, "depth": 6, "epochs": 16000},
    ]
    Ns = [32, 128, 512]; P = 100_000; seeds = 3
    cur = io.load(NAME) or {}
    rows = cur.get("rows", [])
    have = {(r["width"], r["depth"], r["epochs"]) for r in rows if r.get("residual_rmse") == r.get("residual_rmse")}
    io.log(f"E75 extended trained-score sweep starting ({len(configs)} configs, {len(have)} already done)")
    t0 = time.time()
    for cfg in configs:
        key = (cfg["width"], cfg["depth"], cfg["epochs"])
        if key in have:
            io.log(f"  E75 {key} done, skip"); continue
        tc = time.time()
        model, torch, rmse = train_dsm(s2, B, T, width=cfg["width"], depth=cfg["depth"],
                                       epochs=cfg["epochs"], seed=0)
        if not (rmse == rmse):
            io.log(f"  E75 {key} NaN, skip"); continue
        per_N = []
        for N in Ns:
            vs = [sample_learned(model, torch, N, T, B, float(lstar), s2, P, seed=sd)[0] for sd in range(seeds)]
            v_mean = float(np.mean(vs))
            per_N.append({"N": N, "var": v_mean, "KL": float(kl_gauss(v_mean, s2))})
        order = local_order(max(per_N[-2]["KL"], 1e-14), max(per_N[-1]["KL"], 1e-14), Ns[-2], Ns[-1])
        rec = {**cfg, "residual_rmse": rmse, "per_N": per_N, "floor_KL": per_N[-1]["KL"], "tail_order": order}
        io.log(f"  E75 w={cfg['width']} d={cfg['depth']} ep={cfg['epochs']}: rmse={rmse:.4f} "
               f"floor_KL={per_N[-1]['KL']:.3e} ({time.time()-tc:.0f}s)")
        cur = io.load(NAME) or {}
        cur.setdefault("rows", []).append(rec)
        cur["config"] = {"s2": s2, "B": B, "T": T, "lambda_star": lstar, "Ns": Ns, "P": P, "seeds": seeds}
        io.save(NAME, cur)
    best = min((r["residual_rmse"] for r in (io.load(NAME) or {}).get("rows", [])), default=None)
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s; best RMSE this sweep = {best}")


if __name__ == "__main__":
    run()
