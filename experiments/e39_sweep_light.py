"""E39 light churn sweep from the already-trained MNIST U-Net checkpoint.

The full e39 run trains a U-Net (~3 h, done) and then sweeps S_churn with P=2500 samples x 2 seeds, which is
impractically slow on a contended CPU. The expensive part -- training -- is finished and saved to e39_unet.pt.
This script reuses that checkpoint and runs a lighter, denser sweep (more churn points, fewer samples) to get
the SWD-vs-churn curve and its interior optimum quickly. Saves to e39_edm_mnist.json in the same format the
figure generator expects.
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io
import e39_edm_mnist as e39

NAME = "e39_edm_mnist"


def run(P=300, n_seeds=6, N=18):
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    t0 = time.time()
    train_x = e39.load_mnist(train=True)
    test_x = e39.load_mnist(train=False).reshape(-1, e39.RES * e39.RES)
    sigma_data = float(np.std(train_x))
    net = e39.build_unet(torch)()
    ck = torch.load(e39.CKPT, map_location="cpu")
    net.load_state_dict(ck["model"]); net.eval()
    io.log(f"  [light] loaded trained U-Net (step {ck.get('step','?')}), sigma_data={sigma_data:.4f}")

    rng = np.random.default_rng(0); fl = []
    for k in range(n_seeds):
        idx = rng.permutation(test_x.shape[0])
        fl.append(e39.sliced_w1(test_x[idx[:P]], test_x[idx[P:2 * P]], 512, k))
    swd_floor = float(np.mean(fl))
    io.log(f"  [light] SWD floor (real vs real) = {swd_floor:.5f}")

    churn_grid = [0.0, 3.0, 6.0, 10.0, 16.0, 24.0, 40.0, 70.0, 120.0]
    ref = test_x[:P]; swd_mean, swd_std, grids, swd_all = [], [], {}, []
    for sc in churn_grid:
        vals, gen0 = [], None
        for sd in range(n_seeds):
            gen = e39.edm_sample(net, torch, sigma_data, N, sc, 0.002, 80.0, 7, P, seed=10 + sd)
            if gen0 is None: gen0 = gen
            vals.append(e39.sliced_w1(gen, ref, 512, sd))
        swd_mean.append(float(np.mean(vals))); swd_std.append(float(np.std(vals))); swd_all.append([float(v) for v in vals])
        gp = os.path.join(io.FIG_DIR, f"fig_mnist_churn{int(sc)}.png")
        e39.save_grid(gen0, gp); grids[str(sc)] = os.path.basename(gp)
        io.log(f"  [light] S_churn={sc:6.1f}: SWD={swd_mean[-1]:.5f} +/- {swd_std[-1]:.5f} ({time.time()-t0:.0f}s)")

    imin = int(np.argmin(swd_mean))
    res = {"config": {"N": N, "sigma_min": 0.002, "sigma_max": 80.0, "rho": 7, "P": P, "n_seeds": n_seeds,
                      "sigma_data": sigma_data, "target": "MNIST (14x14, [-1,1])", "note": "light sweep from checkpoint"},
           "churn_grid": churn_grid, "swd_mean": swd_mean, "swd_std": swd_std, "swd_all": swd_all, "swd_floor": swd_floor,
           "loss_history": (json.load(open(os.path.join(io.RESULTS_DIR, "e39_loss.json")))
                            if os.path.exists(os.path.join(io.RESULTS_DIR, "e39_loss.json")) else None),
           "measured_opt_S_churn": churn_grid[imin],
           "interior_optimum": bool(0 < imin < len(churn_grid) - 1), "sample_grids": grids}
    io.save(NAME, res)
    io.log(f"  [light] e39 DONE in {time.time()-t0:.0f}s: opt S_churn={churn_grid[imin]}, "
           f"SWD min={swd_mean[imin]:.5f} floor={swd_floor:.5f} interior={res['interior_optimum']}")
    return res


if __name__ == "__main__":
    run()
