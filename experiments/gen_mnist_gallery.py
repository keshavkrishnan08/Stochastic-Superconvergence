"""Build the MNIST 'massive gallery': the SWD-vs-churn curve (with error band + floor) and a 3x3
gallery of generated-digit grids at every churn in the sweep. Reads the multi-seed e39 result."""
import os, sys, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io_utils as io
from figstyle import COL

GREEN = "#0f9d6b"; GREY = "#6b7280"; RED = "#d1495b"

d = json.load(open(os.path.join(io.RESULTS_DIR, "e39_edm_mnist.json")))["data"]
churn = d["churn_grid"]; mean = np.array(d["swd_mean"]); std = np.array(d["swd_std"])
floor = d["swd_floor"]; allv = np.array(d["swd_all"]); opt = d["measured_opt_S_churn"]
oi = churn.index(opt)

# ---- paired significance: deterministic (churn 0) vs interior optimum ----
diff = allv[0] - allv[oi]
md, sd = diff.mean(), diff.std(ddof=1)
sem = sd / math.sqrt(len(diff)); t = md / sem
nfav = int((diff > 0).sum())
print(f"paired churn0 - churn{opt}: mean={md:.5f} sd={sd:.5f} t={t:.3f} df={len(diff)-1} "
      f"favour_opt={nfav}/{len(diff)}")

# ---- (1) the SWD-vs-churn curve ----
plt.rcParams.update({"font.size": 11, "savefig.dpi": 190, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(5.0, 3.5))
ax.fill_between(churn, mean - std, mean + std, color=COL["sde"], alpha=0.18, lw=0)
ax.plot(churn, mean, "-o", color=COL["sde"], ms=4, lw=1.6, label="stochastic EDM sampler")
ax.axhline(floor, ls="--", color=GREY, lw=1.3, label="real-vs-real floor")
ax.plot([opt], [mean[oi]], "*", ms=17, color=GREEN, zorder=6, label=f"interior optimum ($S_{{\\rm churn}}{{=}}{int(opt)}$)")
ax.plot([0], [mean[0]], "o", ms=7, color=GREY, zorder=6)
ax.annotate("deterministic", (0, mean[0]), textcoords="offset points", xytext=(8, 8), fontsize=9, color=GREY)
ax.set_xlabel(r"churn $S_{\rm churn}$"); ax.set_ylabel("sliced-Wasserstein to test set")
ax.set_title("Trained MNIST U-Net: churn sweep (6 seeds)", fontsize=11)
ax.legend(fontsize=8, loc="center right", framealpha=0.95)
ax.margins(x=0.03)
plt.savefig(os.path.join(io.FIG_DIR, "fig_mnist_curve.png")); plt.close()
print("wrote fig_mnist_curve.png")

# ---- (2) 3x3 gallery of digit grids at every churn ----
fig, axes = plt.subplots(3, 3, figsize=(8.6, 9.0))
for k, (ax, v) in enumerate(zip(axes.ravel(), churn)):
    p = os.path.join(io.FIG_DIR, d["sample_grids"][f"{v}"])
    ax.axis("off")
    if os.path.exists(p):
        ax.imshow(mpimg.imread(p))
    is_opt = (v == opt)
    edge = GREEN if is_opt else (GREY if v == 0 else (RED if v >= 70 else "0.8"))
    lw = 3.2 if is_opt else 1.6
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(edge); s.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])
    tag = "  (optimum)" if is_opt else ("  (deterministic)" if v == 0 else ("  (over-churned)" if v >= 70 else ""))
    ax.set_title(rf"$S_{{\rm churn}}={int(v)}${tag}", fontsize=10,
                 color=edge if edge != "0.8" else "0.3", fontweight="bold" if is_opt else "normal")
plt.subplots_adjust(wspace=0.04, hspace=0.12, left=0.01, right=0.99, top=0.97, bottom=0.01)
plt.savefig(os.path.join(io.FIG_DIR, "fig_mnist_gallery9.png")); plt.close()
print("wrote fig_mnist_gallery9.png")
io.log("mnist curve + 3x3 gallery", "figs.log")
