#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path("figures/final")
OUT.mkdir(parents=True, exist_ok=True)

city = pd.read_csv("results/final_hgb_confirmation/city_summary.csv")
city = city[city["delta_pp"] == 0.5].copy()
cls = pd.read_csv("results/final_hgb_confirmation/class_aggregate_summary.csv")
cls = cls[cls["delta_pp"] == 0.5].copy()

# Figure 1: method logic schematic
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")
boxes = [
    (0.3, 1.45, 2.0, 2.0, "Fixed physical patch\n320 m × 320 m\nBuildings + LCZ fixed"),
    (3.0, 2.75, 2.2, 1.35, "Hard zoning branch\n4 × 4 morphology\n9 configurations"),
    (3.0, 0.85, 2.2, 1.35, "Continuous branch\n16 anchors\nGaussian weighting"),
    (6.0, 1.45, 2.0, 2.0, "Probability fusion\n(1 − α) p_h + α p_c"),
    (8.75, 1.45, 2.0, 2.0, "Nested city selection\nminimize PIER\nsubject to worst-city\naccuracy loss ≤ 0.5 pp"),
]
for x,y,w,h,t in boxes:
    ax.add_patch(Rectangle((x,y),w,h,fill=False,linewidth=1.2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=10)
for a,b in [((2.3,2.45),(3.0,3.4)),((2.3,2.45),(3.0,1.5)),((5.2,3.4),(6.0,2.75)),((5.2,1.5),(6.0,2.15)),((8.0,2.45),(8.75,2.45))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=12,linewidth=1.0))
ax.text(5.5,4.65,"Fixed-support zoning robustness framework",ha="center",fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"fig1_method_schematic.svg",bbox_inches="tight")
plt.close(fig)

# Figure 2: held-out city trade-off
fig, ax = plt.subplots(figsize=(7.4, 6.0))
x = city["accuracy_change_pp"].to_numpy()
y = 100*(city["selected_pier"]-city["baseline_pier"]).to_numpy()
ax.scatter(x,y,s=42,facecolors="none",edgecolors="black")
abbr = {
    "guangzhou":"GZ","jakarta":"JKT","moscow":"MOS","mumbai":"MUM","munich":"MUC",
    "nairobi":"NBO","sanfrancisco":"SF","santiago":"SCL","sydney":"SYD","tehran":"THR"
}
for _,r in city.iterrows():
    ax.annotate(abbr.get(r["city"],r["city"]),(
        r["accuracy_change_pp"],
        100*(r["selected_pier"]-r["baseline_pier"])
    ),xytext=(4,4),textcoords="offset points",fontsize=8)
ax.axvline(-0.5,linestyle="--",linewidth=1.0)
ax.axvline(0,linewidth=0.8)
ax.axhline(0,linewidth=0.8)
ax.set_xlabel("Held-out accuracy change (percentage points)")
ax.set_ylabel("Held-out PIER change (percentage points)")
ax.set_title("City-held-out accuracy–stability outcomes")
ax.text(-0.49, ax.get_ylim()[1]*0.94, "0.5 pp constraint", fontsize=8, va="top")
fig.tight_layout()
fig.savefig(OUT/"fig2_city_tradeoff.svg",bbox_inches="tight")
plt.close(fig)

# Figure 3: class-level accuracy changes
fig, ax = plt.subplots(figsize=(7.6, 4.8))
labels = cls["label"].astype(int).to_numpy()
vals = cls["accuracy_change_pp"].to_numpy()
ax.bar(labels,vals,fill=False,edgecolor="black")
ax.axhline(0,linewidth=0.8)
ax.set_xticks(labels)
ax.set_xlabel("LCZ built class")
ax.set_ylabel("Accuracy change (percentage points)")
ax.set_title("Class-level predictive change under the primary specification")
fig.tight_layout()
fig.savefig(OUT/"fig3_class_accuracy.svg",bbox_inches="tight")
plt.close(fig)

# Figure 4: class-level PIER changes
fig, ax = plt.subplots(figsize=(7.6, 4.8))
vals = cls["pier_change_pp"].to_numpy()
ax.bar(labels,vals,fill=False,edgecolor="black")
ax.axhline(0,linewidth=0.8)
ax.set_xticks(labels)
ax.set_xlabel("LCZ built class")
ax.set_ylabel("PIER change (percentage points)")
ax.set_title("Class-level reduction in zoning-induced instability")
fig.tight_layout()
fig.savefig(OUT/"fig4_class_pier.svg",bbox_inches="tight")
plt.close(fig)
