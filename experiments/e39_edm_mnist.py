"""E39 -- Tier 2: literal EDM sampler on MNIST with a trained U-Net score (CPU, overnight).

The image-scale companion to E36. Target: MNIST (a real, high-dimensional, strongly non-Gaussian
distribution). Score: a small EDM-preconditioned U-Net trained by denoising score matching
(Karras et al. 2022). Sampler: the literal EDM Algorithm 2 (stochastic Heun + S_churn + rho=7).
We sweep S_churn and measure a sample-based sliced-Wasserstein distance (on pixels) to held-out
MNIST test images, plus the real-vs-real floor.

Outputs: SWD(S_churn) curve with interior optimum, sample grids at churn=0 vs the optimum, and a
checkpointed model so the (long) training is resumable. CPU-only; designed to run unattended.
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import io_utils as io

NAME = "e39_edm_mnist"
CKPT = os.path.join(io.RESULTS_DIR, "e39_unet.pt")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mnist"))
SQRT2_MINUS_1 = float(np.sqrt(2) - 1)


# ------------------------------- data -------------------------------
RES = 14   # work at 14x14 (2x2 avg-pool of MNIST): clearly digits, ~4x cheaper on CPU


def load_mnist(train=True):
    import torchvision, torchvision.datasets as ds
    d = ds.MNIST(root=DATA, train=train, download=True)
    x = d.data.numpy().astype(np.float32) / 127.5 - 1.0      # [-1,1], (n,28,28)
    x = x.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))         # 2x2 avg-pool -> (n,14,14)
    return x[:, None, :, :]                                   # (n,1,14,14)


# ------------------------------- small EDM U-Net -------------------------------
def build_unet(torch):
    import torch.nn as nn

    class TimeEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        def forward(self, c_noise):
            return self.lin(c_noise[:, None])

    class Block(nn.Module):
        def __init__(self, cin, cout, tdim):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.emb = nn.Linear(tdim, cout)
            self.act = nn.SiLU()
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        def forward(self, x, t):
            h = self.act(self.c1(x))
            h = h + self.emb(t)[:, :, None, None]
            h = self.act(self.c2(h))
            return h + self.skip(x)

    def up_block(cin, cout):
        # nearest-upsample + conv: far faster than ConvTranspose2d backward on CPU
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                             nn.Conv2d(cin, cout, 3, padding=1))

    class UNet(nn.Module):
        """Two-level U-Net for 14x14 inputs (14 <-> 7)."""
        def __init__(self, base=48, tdim=96):
            super().__init__()
            self.temb = TimeEmb(tdim)
            self.in_conv = nn.Conv2d(1, base, 3, padding=1)
            self.d1 = Block(base, base, tdim)
            self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)      # 14->7
            self.mid = Block(base * 2, base * 2, tdim)
            self.up1 = up_block(base * 2, base)                                # 7->14
            self.u1 = Block(base * 2, base, tdim)
            self.out = nn.Conv2d(base, 1, 3, padding=1)
        def forward(self, x, c_noise):
            t = self.temb(c_noise)
            h0 = self.in_conv(x)
            h1 = self.d1(h0, t)
            hm = self.mid(self.down1(h1), t)
            u1 = self.u1(torch.cat([self.up1(hm), h1], dim=1), t)
            return self.out(u1)

    return UNet


def denoise(net, x, sigma, sigma_data, torch):
    s = sigma if torch.is_tensor(sigma) else torch.full((x.shape[0],), float(sigma))
    s = s.reshape(-1, 1, 1, 1)
    sd2 = sigma_data ** 2
    c_skip = sd2 / (s ** 2 + sd2)
    c_out = s * sigma_data / torch.sqrt(s ** 2 + sd2)
    c_in = 1.0 / torch.sqrt(s ** 2 + sd2)
    c_noise = torch.log(s.reshape(-1)) / 4.0
    return c_skip * x + c_out * net(c_in * x, c_noise)


# ------------------------------- training (resumable) -------------------------------
def train(torch, data, sigma_data, total_steps=40000, batch=128, lr=2e-4,
          P_mean=-1.2, P_std=1.2, seed=0, log_every=250, ckpt_every=1000):
    import torch.nn as nn
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    torch.manual_seed(seed)
    net = build_unet(torch)()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    step0 = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
        io.log(f"  resume from step {step0}")
    X = torch.from_numpy(data)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    loss_path = os.path.join(io.RESULTS_DIR, "e39_loss.json")
    loss_hist = json.load(open(loss_path)) if os.path.exists(loss_path) else {"step": [], "loss": []}
    net.train()
    for step in range(step0, total_steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = X[idx]
        ln_sigma = P_mean + P_std * torch.randn(batch, generator=g)
        sigma = torch.exp(ln_sigma)
        noise = torch.randn(batch, 1, RES, RES, generator=g) * sigma[:, None, None, None]
        D = denoise(net, x0 + noise, sigma, sigma_data, torch)
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        loss = (weight[:, None, None, None] * (D - x0) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if (step + 1) % log_every == 0:
            io.log(f"  step {step+1}/{total_steps} loss={loss.item():.4f} "
                   f"({time.time()-t0:.0f}s, {(step+1-step0)/(time.time()-t0+1e-9):.1f} it/s)")
            loss_hist["step"].append(step + 1); loss_hist["loss"].append(float(loss.item()))
            json.dump(loss_hist, open(loss_path, "w"))
        if (step + 1) % ckpt_every == 0:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "sigma_data": sigma_data}, CKPT)
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                "step": total_steps, "sigma_data": sigma_data}, CKPT)
    net.eval()
    return net


# ------------------------------- EDM Algorithm-2 sampler -------------------------------
def edm_sigmas(N, sigma_min, sigma_max, rho=7):
    i = np.arange(N)
    a = sigma_max ** (1 / rho) + i / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    return np.concatenate([a ** rho, [0.0]])


def edm_sample(net, torch, sigma_data, N, S_churn, sigma_min, sigma_max, rho, P, seed,
               S_noise=1.0, batch=2000):
    g = torch.Generator().manual_seed(seed)
    sig = edm_sigmas(N, sigma_min, sigma_max, rho)
    gcap = SQRT2_MINUS_1
    outs = []
    done = 0
    with torch.no_grad():
        while done < P:
            b = min(batch, P - done)
            x = torch.randn(b, 1, RES, RES, generator=g) * float(sig[0])
            for i in range(N):
                si = float(sig[i]); si1 = float(sig[i + 1])
                gamma = min(S_churn / N, gcap)
                shat = si * (1 + gamma)
                if gamma > 0:
                    x = x + np.sqrt(max(shat ** 2 - si ** 2, 0.0)) * S_noise * torch.randn(b, 1, RES, RES, generator=g)
                d = (x - denoise(net, x, shat, sigma_data, torch)) / shat
                xn = x + (si1 - shat) * d
                if si1 > 0:
                    d2 = (xn - denoise(net, xn, si1, sigma_data, torch)) / si1
                    xn = x + (si1 - shat) * (0.5 * d + 0.5 * d2)
                x = xn
            outs.append(x.clamp(-1, 1).numpy())
            done += b
    return np.concatenate(outs, axis=0).reshape(P, -1)


def sliced_w1(a, b, n_proj=512, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.standard_normal((n_proj, a.shape[1]))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    pa = np.sort(a @ th.T, axis=0); pb = np.sort(b @ th.T, axis=0)
    n = min(pa.shape[0], pb.shape[0])
    return float(np.mean(np.abs(pa[:n] - pb[:n])))


def save_grid(samples, path, nrow=8):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    imgs = samples[:nrow * nrow].reshape(-1, RES, RES)
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im, cmap="gray", vmin=-1, vmax=1); ax.axis("off")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def run(N=24, sigma_min=0.002, sigma_max=80.0, rho=7, P=2500, n_seeds=2,
        total_steps=12000):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    import torch
    t0 = time.time()
    train_x = load_mnist(train=True)
    test_x = load_mnist(train=False).reshape(-1, RES * RES)
    sigma_data = float(np.std(train_x))
    io.log(f"  MNIST loaded: train {train_x.shape}, sigma_data={sigma_data:.4f}")

    net = train(torch, train_x, sigma_data, total_steps=total_steps)
    io.log(f"  training done ({time.time()-t0:.0f}s)")

    # SWD floor: two independent held-out real subsets
    rng = np.random.default_rng(0)
    fl = []
    for k in range(n_seeds):
        idx = rng.permutation(test_x.shape[0])
        a = test_x[idx[:P]]; b = test_x[idx[P:2 * P]]
        fl.append(sliced_w1(a, b, 512, k))
    swd_floor = float(np.mean(fl))
    io.log(f"  SWD floor (real vs real) = {swd_floor:.5f}")

    churn_grid = [0.0, 5.0, 15.0, 35.0, 70.0, 120.0]
    ref = test_x[:P]
    swd_mean, swd_std = [], []
    grids = {}
    for sc in churn_grid:
        vals = []
        gen0 = None
        for sd in range(n_seeds):
            gen = edm_sample(net, torch, sigma_data, N, sc, sigma_min, sigma_max, rho, P, seed=10 + sd)
            if gen0 is None:
                gen0 = gen
            vals.append(sliced_w1(gen, ref, 512, sd))
        swd_mean.append(float(np.mean(vals))); swd_std.append(float(np.std(vals)))
        gp = os.path.join(io.FIG_DIR, f"fig_mnist_churn{int(sc)}.png")
        save_grid(gen0, gp)
        grids[str(sc)] = os.path.basename(gp)
        io.log(f"  S_churn={sc:5.1f}: SWD={swd_mean[-1]:.5f} +/- {swd_std[-1]:.5f} ({time.time()-t0:.0f}s)")

    imin = int(np.argmin(swd_mean))
    res = {
        "config": {"N": N, "sigma_min": sigma_min, "sigma_max": sigma_max, "rho": rho,
                   "P": P, "n_seeds": n_seeds, "total_steps": total_steps,
                   "sigma_data": sigma_data, "target": "MNIST (28x28, [-1,1])"},
        "churn_grid": churn_grid, "swd_mean": swd_mean, "swd_std": swd_std, "swd_floor": swd_floor,
        "loss_history": (json.load(open(os.path.join(io.RESULTS_DIR, "e39_loss.json")))
                         if os.path.exists(os.path.join(io.RESULTS_DIR, "e39_loss.json")) else None),
        "measured_opt_S_churn": churn_grid[imin],
        "interior_optimum": bool(0 < imin < len(churn_grid) - 1),
        "swd_at_churn0": swd_mean[0], "swd_at_opt": swd_mean[imin],
        "swd_improvement_factor": float(swd_mean[0] / swd_mean[imin]) if swd_mean[imin] > 0 else None,
        "sample_grids": grids,
    }
    io.save(NAME, res)
    io.log(f"{NAME} DONE in {time.time()-t0:.0f}s | interior={res['interior_optimum']} "
           f"measured*={res['measured_opt_S_churn']} floor={swd_floor:.4f} "
           f"SWD {swd_mean[0]:.4f}->{swd_mean[imin]:.4f}")
    return res


if __name__ == "__main__":
    run()
