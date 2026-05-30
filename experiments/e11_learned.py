"""E11 — Learned-score boundary (practical reach).
Part A (deterministic, bulletproof): miscalibrated LINEAR score s_hat=-x/(V+delta) -> exact
  recursion -> N-independent KL floor that masks super-convergence. Validates the umbrella
  score-floor term: floor ~ (c*delta)^2.
Part B (trained torch MLP, heavy): DSM-trained score, sweep width/depth/epochs; measure
  residual score error and the resulting sampler KL floor; floor shrinks with capacity/epochs.
Resumable: Part A and each Part-B config checkpoint to results/e11_learned.json.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import set_dps
from learned_score import miscalibrated_recursion, train_dsm, sample_learned
from lambda_star import lambda_star_vp
from metrics import kl_gauss, local_order
import io_utils as io

NAME = "e11_learned"


def part_A(s2, B, T, lstar, dps=90):
    set_dps(dps)
    Ns = [2 ** k for k in range(5, 15)]   # 32..16384
    out = {}
    for delta in [0.0, 0.005, 0.02, 0.05, 0.1, 0.2]:
        kls = [float(kl_gauss(miscalibrated_recursion(N, T, B, float(lstar), s2, delta), s2)) for N in Ns]
        tail = local_order(kls[-3], kls[-1], Ns[-3], Ns[-1])
        out[f"{delta}"] = {"delta": delta, "Ns": Ns, "KL": kls,
                           "floor": kls[-1], "tail_order": tail}
        io.log(f"  [A] delta={delta:5}: floor(KL,N=16384)={kls[-1]:.3e} tail_order={tail:.3f}")
    return out


def _valid(rec):
    return rec and rec.get("residual_rmse") is not None and rec["residual_rmse"] == rec["residual_rmse"]  # not NaN


def part_B(s2, B, T, lstar):
    import math
    configs = []
    for width in [16, 64, 256]:
        for depth in [2, 4]:
            for epochs in [2000, 8000]:
                configs.append({"width": width, "depth": depth, "epochs": epochs})
    Ns = [32, 128, 512]; P = 100_000; seeds = 2
    # keep only valid prior results; re-run the rest
    cur = io.load(NAME) or {}
    prior = [r for r in cur.get("part_B", []) if _valid(r) and not (isinstance(r.get("floor_KL"), float) and math.isnan(r["floor_KL"]))]
    have = {(r["width"], r["depth"], r["epochs"]) for r in prior}
    cur["part_B"] = prior
    io.save(NAME, cur)
    for cfg in configs:
        key = (cfg["width"], cfg["depth"], cfg["epochs"])
        if key in have:
            io.log(f"  [B] {key} valid, skip"); continue
        model, torch, rmse = train_dsm(s2, B, T, width=cfg["width"], depth=cfg["depth"],
                                       epochs=cfg["epochs"], seed=0)
        if not (rmse == rmse):   # NaN guard
            io.log(f"  [B] {key} STILL NaN, skipping"); continue
        per_N = []
        for N in Ns:
            vs = [sample_learned(model, torch, N, T, B, float(lstar), s2, P, seed=sd)[0] for sd in range(seeds)]
            v_mean = float(np.mean(vs))
            per_N.append({"N": N, "var": v_mean, "KL": float(kl_gauss(v_mean, s2))})
        order = local_order(max(per_N[-2]["KL"], 1e-14), max(per_N[-1]["KL"], 1e-14), Ns[-2], Ns[-1])
        rec = {**cfg, "residual_rmse": rmse, "per_N": per_N,
               "floor_KL": per_N[-1]["KL"], "tail_order": order}
        io.log(f"  [B] w={cfg['width']} d={cfg['depth']} ep={cfg['epochs']}: "
               f"resid_rmse={rmse:.4f} floor_KL={per_N[-1]['KL']:.3e} order={order:.2f}")
        cur = io.load(NAME) or {}
        cur.setdefault("part_B", []).append(rec)
        io.save(NAME, cur)


def run(s2=2.0, B=4.0, T=5.0):
    cur = io.load(NAME) or {}
    lstar = lambda_star_vp(B, s2, T)
    if "part_A" not in cur:
        cur["config"] = {"s2": s2, "B": B, "T": T, "lambda_star": float(lstar)}
        cur["part_A"] = part_A(s2, B, T, lstar)
        io.save(NAME, cur)
        io.log(f"{NAME}: Part A done (deterministic miscalibration floors).")
    valid_B = [r for r in cur.get("part_B", []) if r.get("residual_rmse") == r.get("residual_rmse")
               and r.get("residual_rmse") is not None]
    if len(valid_B) < 12:
        io.log(f"{NAME}: Part B (trained MLP ablations): {len(valid_B)}/12 valid, running rest...")
        part_B(s2, B, T, lstar)
        io.log(f"{NAME}: Part B done.")
    return io.load(NAME)


if __name__ == "__main__":
    run()
