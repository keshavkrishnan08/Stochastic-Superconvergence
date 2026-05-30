import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ssc"))
import io_utils as io
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
need = ["fig_concept","fig_canyon","fig_headline","fig_coeff_field","fig_resonance","fig_order_field",
        "fig_goldilocks3d","fig_spaghetti","fig_phase","fig_mixture","fig_floor","fig_lambda_law",
        "fig_goldilocks","fig_coeff_validation","fig_invariance","fig_sensitivity","fig_schedules"]
made=0
for n in need:
    p=os.path.join(io.FIG_DIR,n+".png")
    if not os.path.exists(p):
        plt.figure(figsize=(6,4)); plt.text(0.5,0.5,n+"\n(pending)",ha="center",va="center")
        plt.axis("off"); plt.savefig(p); plt.close(); made+=1
print(f"figguard: {made} placeholders created")
