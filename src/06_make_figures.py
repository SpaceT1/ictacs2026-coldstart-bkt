"""Step 6: produce Fig. 1 and Fig. 2 (PDF, vector)."""
import numpy as np, pickle, json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
os.makedirs("figures", exist_ok=True)
S = json.load(open("results/exp1_summary.json"))
ns = [10, 25, 50, 100, 200]; auc = [S[f"Fitted-{n}"]["auc"] for n in ns]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.plot(ns, auc, "o-", color="k", lw=1, ms=3.5, label="BKT fitted on $n$ pilot learners")
ax.axhline(S["Fitted-all"]["auc"], ls="--", color="0.3", lw=0.8, label="BKT fitted on full training fold")
ax.axhline(S["Canonical"]["auc"], ls=":", color="0.3", lw=1, label="Canonical literature parameters")
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
ax.set_xlabel("Pilot learners per skill used for calibration ($n$)"); ax.set_ylabel("Pooled test AUC")
ax.set_ylim(0.635, 0.725); ax.grid(alpha=.25, lw=.4)
ax.legend(frameon=True, framealpha=1, edgecolor="none", fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.14))
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_calibration.pdf")
out = pickle.load(open("results/exp2.pkl", "rb"))
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
sty = {"Random": ("0.75", ":"), "Fixed curriculum": ("0.45", "-"), "Counter rules": ("0.45", "--"),
       "Flat BKT+rules": ("k", ":"), "Level-BKT+rules": ("k", "-"), "Level-BKT+rules (cal.)": ("k", "--"), "Oracle": ("0.65", "-.")}
for ax, pop in zip(axs, ["weak", "typical", "strong"]):
    for name, (c, ls) in sty.items():
        ax.plot(np.arange(1, 101), np.stack(out[(pop, name)].curve.values).mean(0), color=c, ls=ls, lw=1, label=name)
    ax.set_title(f"{pop.capitalize()} population", fontsize=8); ax.set_xlabel("Tasks completed"); ax.grid(alpha=.25, lw=.4)
axs[0].set_ylabel("Share of skills truly mastered"); axs[0].legend(frameon=False, fontsize=5.8, loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_curves.pdf")
print("figures saved")
