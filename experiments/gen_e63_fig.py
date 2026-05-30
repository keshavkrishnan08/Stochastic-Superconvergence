"""Standalone figure for E63 (integrator-cancellation rule fires). No shared-file edits."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
d = json.load(open(os.path.join(ROOT, "results", "e63_integrator_root.json")))["data"]

Ns = np.array(d["Ns"], float)
a = d["fired_alpha"]; root = d["fired_root"]
kr = np.array(d["curve_root"], float)
kg = np.array(d["curve_generic"], float)
kh = np.array(d["curve_heun"], float)

def slope(x, y):
    m = y > 0
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

# ---- Panel A: KL vs N, the order jump 4 -> 6 ----
axL.loglog(Ns, kh, "o--", color="#9aa7b5", lw=1.8, ms=5, label=f"Heun ($\\alpha{{=}}1$), $\\lambda{{=}}1$  (slope {slope(Ns,kh):.2f})")
axL.loglog(Ns, kg, "s-.", color="#2c6fbb", lw=1.8, ms=5, label=f"midpoint ($\\alpha{{=}}0.5$), $\\lambda{{=}}1$  (slope {slope(Ns,kg):.2f})")
axL.loglog(Ns, kr, "o-", color="#d1495b", lw=2.2, ms=6, label=f"midpoint at $\\lambda^\\dagger{{=}}{root:.3f}$  (slope {slope(Ns,kr):.2f})")
# reference triangles
x0 = Ns[2]
for p, c, dy in [(4, "#2c6fbb", kg[2]), (6, "#d1495b", kr[2])]:
    axL.loglog([x0, x0*2], [dy, dy*2.0**(-p)], color=c, lw=1, alpha=0.5)
axL.set_xlabel("sampler steps $N$"); axL.set_ylabel("KL to target (exact, $\\geq 60$-digit)")
axL.set_title("A churn root lifts a 2nd-order integrator to 6th order")
axL.legend(fontsize=8.5, loc="lower left"); axL.grid(True, which="both", alpha=0.25)

# ---- Panel B: cancellation churn lambda^dagger(alpha); vanishes at Heun ----
fam = d["alpha_family"]
al = [f["alpha"] for f in fam]
rt = [f["root"] for f in fam]
al_root = [a_ for a_, r_ in zip(al, rt) if r_ is not None]
rt_root = [r_ for r_ in rt if r_ is not None]
al_none = [a_ for a_, r_ in zip(al, rt) if r_ is None]
axR.plot(al_root, rt_root, "o-", color="#d1495b", lw=2, ms=7, label="cancellation churn $\\lambda^\\dagger(\\alpha)$")
axR.scatter(al_none, [0]*len(al_none), marker="x", color="#9aa7b5", s=55, zorder=5, label="no root (incl. Heun $\\alpha{=}1$)")
axR.axvline(1.0, color="#444", ls=":", lw=1)
axR.annotate("Heun", (1.0, max(rt_root)*0.5), fontsize=9, color="#444",
             ha="left", xytext=(1.04, max(rt_root)*0.55))
axR.set_xlabel("integrator parameter $\\alpha$ (2-stage RK2 family)")
axR.set_ylabel("cancellation churn $\\lambda^\\dagger$")
axR.set_title("Each member has its own churn root; it dies at Heun")
axR.set_ylim(bottom=-0.05)
axR.legend(fontsize=8.5, loc="upper right"); axR.grid(True, alpha=0.25)

fig.tight_layout()
out = os.path.join(ROOT, "figures", "fig_integrator_root.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
