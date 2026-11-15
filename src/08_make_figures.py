"""Step 8: figures of the paper (PDF, vector) and of the supplement.
Fig. 1  calibration curve (Experiment 1)
Fig. 2  share of truly mastered skills over tasks (Experiment 2)
Fig. S1 precision / mastery trade-off of mastery criteria (supplement)
"""
import numpy as np, pandas as pd, json, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
os.makedirs("figures", exist_ok=True)

# ---------------- Fig. 1
S = json.load(open("results/exp1_summary.json"))
ns = [10, 25, 50, 100, 200]
fig, ax = plt.subplots(figsize=(3.4, 2.15))
for v, lab, st in (("all", "Pilot fit: all four parameters", "o-"),
                   ("noprior", "Pilot fit: prior fixed at 0.10", "s--")):
    mean = [S["pilot"][f"{v}-{n}"]["auc_mean"] for n in ns]
    lo = [S["pilot"][f"{v}-{n}"]["auc_min"] for n in ns]; hi = [S["pilot"][f"{v}-{n}"]["auc_max"] for n in ns]
    ax.plot(ns, mean, st, color="k" if v == "all" else "0.45", lw=1, ms=3, label=lab)
    ax.fill_between(ns, lo, hi, color="0.85" if v == "all" else "0.92", lw=0)
ax.axhline(S["Fitted-all"]["auc"], ls="--", color="0.3", lw=0.8, label="BKT fitted on full training fold")
ax.axhline(S["PFA"]["auc"], ls="-.", color="0.55", lw=0.8, label="PFA baseline (full training fold)")
ax.axhline(S["Canonical"]["auc"], ls=":", color="0.3", lw=1, label="Canonical literature parameters")
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
ax.set_xlabel("Pilot learners per skill used for calibration ($n$)"); ax.set_ylabel("Pooled test AUC")
lo_all = min(S["Canonical"]["auc"], min(S["pilot"][f"noprior-{n}"]["auc_min"] for n in ns))
ax.set_ylim(lo_all - 0.032, max(S["Fitted-all"]["auc"], S["PFA"]["auc"]) + 0.008)
ax.grid(alpha=.25, lw=.4)
ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.0, loc="lower right", ncol=1, borderaxespad=0.3, handlelength=2.2)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_calibration.pdf")

# ---------------- Fig. 2
e2 = __import__("05_experiment2")
J = json.load(open("results/exp2_jobs.json"))
M = {(j[0], j[1], j[3], j[4]): pd.read_pickle(f"results/exp2_cache/{e2.key(tuple(j))}.pkl")[1] for j in J["main"]}
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
sty = {"Random": ("0.75", ":", "Random"), "Fixed curriculum": ("0.55", "-", "Fixed curriculum, hand-set"),
       "Fixed curriculum (tuned)": ("0.35", (0, (5, 1, 1, 1)), "Fixed curriculum, tuned"),
       "Counter rules": ("0.5", "--", "Counter rules"), "Flat BKT+rules": ("k", ":", "Flat BKT+rules"),
       "KT-IDEM-style BKT+rules (tuned)": ("0.3", (0, (1, 1)), "KT-IDEM-style, tuned"),
       "Level-BKT+rules": ("k", "-", "Level-BKT+rules"), "Level-BKT+rules (cal.)": ("k", "--", "Level-BKT+rules, cal. $P(T)$"),
       "Oracle": ("0.65", "-.", "Oracle")}
for ax, pop in zip(axs, ["weak", "typical", "strong"]):
    for name, (c, ls, lab) in sty.items():
        cur = np.stack(M[("main", "base", pop, name)].curve.values).mean(0)
        ax.plot(np.arange(1, 101), cur, color=c, ls=ls, lw=1.1 if "Level" in name else 0.9, label=lab)
    ax.set_title(f"{pop.capitalize()} population", fontsize=8); ax.set_xlabel("Tasks completed"); ax.grid(alpha=.25, lw=.4)
axs[0].set_ylabel("Share of skills truly mastered")
axs[0].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.0, loc="upper left", handlelength=2.6, borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_curves.pdf")

# ---------------- Fig. S1 (supplement): precision vs mastery of mastery criteria
PT = pd.read_csv("results/exp2_policy_sensitivity.csv"); T = pd.read_csv("results/exp2_table.csv")
pts = []
for name, g in T[T.policy.isin(["Flat BKT+rules", "KT-IDEM-style BKT+rules (tuned)", "Level-BKT+rules",
                                 "Level-BKT+rules (cal.)", "Counter rules"])].groupby("policy"):
    pts.append((name, g.mastery.mean(), g.precision_pooled.iloc[0]))
for name, g in PT.groupby("variant"):
    pts.append((name, g.mastery.mean(), g.precision_pooled.iloc[0]))
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for name, m, p in pts:
    ax.scatter(100 * p, 100 * m, s=12, color="k" if name.startswith("Level") else "0.6")
    ax.annotate(name, (100 * p, 100 * m), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("Declaration precision (%), pooled over populations")
ax.set_ylabel("Share of skills truly mastered (%), mean of populations")
ax.grid(alpha=.25, lw=.4); fig.tight_layout(pad=0.3); fig.savefig("figures/figS1_precision_tradeoff.pdf")
fig.savefig("figures/figS1_precision_tradeoff.png", dpi=160)
print("figures saved")
