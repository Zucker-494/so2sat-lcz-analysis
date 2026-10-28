#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = Path("figures/final")
OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-aware palette with fixed semantics across figures.
COL_HARD = "#4477AA"       # hard zoning
COL_CONT = "#EE7733"       # continuous representation
COL_FUSION = "#AA3377"     # probability fusion
COL_ROBUST = "#228833"     # worst-city geographic constraint / favorable change
COL_MEAN = "#CCBB44"       # mean-city constraint
COL_BAD = "#CC3311"        # constraint violation / deterioration
COL_NEUTRAL = "#6B7280"    # baseline / fixed support
COL_LIGHT = "#F3F4F6"
COL_TEXT = "#222222"
COL_GRID = "#D1D5DB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "text.color": COL_TEXT,
        "axes.labelcolor": COL_TEXT,
        "axes.titlecolor": COL_TEXT,
        "xtick.color": COL_TEXT,
        "ytick.color": COL_TEXT,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

city = pd.read_csv("results/final_hgb_confirmation/city_summary.csv")
city = city[city["delta_pp"] == 0.5].copy()

cls = pd.read_csv("results/final_hgb_confirmation/class_aggregate_summary.csv")
cls = cls[cls["delta_pp"] == 0.5].copy()

mean_city = pd.read_csv("results/nested_accuracy_constrained/outer_city_results.csv")
mean_city = mean_city[
    (mean_city["model"] == "hgb") & (mean_city["delta_pp"] == 0.5)
].copy()

robust_city = pd.read_csv("results/robust_nested_accuracy_constraint/outer_city_results.csv")
robust_city = robust_city[
    (robust_city["model"] == "hgb") & (robust_city["delta_pp"] == 0.5)
].copy()

abbr = {
    "guangzhou": "GZ",
    "jakarta": "JKT",
    "moscow": "MOS",
    "mumbai": "MUM",
    "munich": "MUC",
    "nairobi": "NBO",
    "sanfrancisco": "SF",
    "santiago": "SCL",
    "sydney": "SYD",
    "tehran": "THR",
}


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=COL_GRID, linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


# Figure 1: method logic schematic with semantic color encoding.
fig, ax = plt.subplots(figsize=(11.2, 4.9))
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 5.1)
ax.axis("off")

boxes = [
    (0.25, 1.50, 2.10, 2.00, "Fixed physical patch\n320 m × 320 m\nBuildings + LCZ fixed", COL_NEUTRAL, "#F8FAFC"),
    (3.00, 2.80, 2.25, 1.35, "Hard zoning branch\n4 × 4 morphology\n9 configurations", COL_HARD, "#EAF1F8"),
    (3.00, 0.80, 2.25, 1.35, "Continuous branch\n16 anchors\nGaussian weighting", COL_CONT, "#FDF0E6"),
    (6.05, 1.50, 2.05, 2.00, "Probability fusion\n(1 − α) p_h + α p_c", COL_FUSION, "#F5EAF2"),
    (8.80, 1.35, 2.15, 2.30, "Nested city selection\nminimize PIER\nworst-city accuracy\nloss ≤ 0.5 pp", COL_ROBUST, "#EAF5EC"),
]

for x, y, w, h, text, edge, face in boxes:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.7,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

arrows = [
    ((2.35, 2.50), (3.00, 3.45)),
    ((2.35, 2.50), (3.00, 1.47)),
    ((5.25, 3.45), (6.05, 2.82)),
    ((5.25, 1.47), (6.05, 2.18)),
    ((8.10, 2.50), (8.80, 2.50)),
]
for start, end in arrows:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.15,
            color="#4B5563",
        )
    )

ax.text(
    5.60,
    4.75,
    "Geographically constrained reduction of zoning-induced prediction instability",
    ha="center",
    fontsize=12.5,
    weight="semibold",
)
ax.text(4.12, 4.32, "representation branches", ha="center", fontsize=8.3, color=COL_NEUTRAL)
ax.text(7.07, 3.72, "stabilization", ha="center", fontsize=8.3, color=COL_FUSION)
ax.text(9.88, 3.92, "geographic selection", ha="center", fontsize=8.3, color=COL_ROBUST)

fig.tight_layout()
fig.savefig(OUT / "fig1_method_schematic.svg", bbox_inches="tight")
plt.close(fig)


# Figure 2: held-out city trade-off with performance-preservation semantics.
fig, ax = plt.subplots(figsize=(7.6, 6.1))
x = city["accuracy_change_pp"].to_numpy()
y = 100 * (city["selected_pier"] - city["baseline_pier"]).to_numpy()

point_colors = [
    COL_ROBUST if val >= 0 else (COL_MEAN if val >= -0.5 else COL_BAD)
    for val in x
]

ax.axvspan(-0.5, max(0.8, x.max() + 0.25), color=COL_ROBUST, alpha=0.06, zorder=0)
ax.scatter(x, y, s=58, c=point_colors, edgecolors="white", linewidths=0.8, zorder=3)

for _, r in city.iterrows():
    ax.annotate(
        abbr.get(r["city"], r["city"]),
        (
            r["accuracy_change_pp"],
            100 * (r["selected_pier"] - r["baseline_pier"]),
        ),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax.axvline(-0.5, color=COL_BAD, linestyle="--", linewidth=1.2)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.axhline(0, color="#555555", linewidth=0.9)
ax.set_xlabel("Held-out accuracy change (percentage points)")
ax.set_ylabel("Held-out PIER change (percentage points)")
ax.set_title("Held-out cities: predictive change versus zoning-instability change")
clean_axis(ax, grid_axis="both")

legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_ROBUST, markeredgecolor="white", markersize=7, label="Accuracy preserved / improved"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_MEAN, markeredgecolor="white", markersize=7, label="Accuracy decrease within 0.5 pp"),
    Line2D([0], [0], color=COL_BAD, linestyle="--", linewidth=1.2, label="0.5 pp non-inferiority boundary"),
]
ax.legend(handles=legend, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig2_city_tradeoff.svg", bbox_inches="tight")
plt.close(fig)


# Figure 3: class-level predictive effects.
fig, ax = plt.subplots(figsize=(7.8, 4.9))
labels = cls["label"].astype(int).to_numpy()
vals = cls["accuracy_change_pp"].to_numpy()
bar_colors = [COL_ROBUST if v >= 0 else COL_BAD for v in vals]
bars = ax.bar(labels, vals, color=bar_colors, width=0.68, edgecolor="white", linewidth=0.6)

ax.axhline(0, color="#555555", linewidth=0.9)
ax.set_xticks(labels)
ax.set_xlabel("LCZ built class")
ax.set_ylabel("Accuracy change (percentage points)")
ax.set_title("LCZ class-level predictive effects under the primary specification")
clean_axis(ax)
ax.bar_label(bars, labels=[f"{v:+.2f}" for v in vals], padding=2, fontsize=7.2)

ax.legend(
    handles=[
        Patch(facecolor=COL_ROBUST, label="Higher accuracy"),
        Patch(facecolor=COL_BAD, label="Lower accuracy"),
    ],
    frameon=False,
    ncol=2,
    loc="upper left",
)
fig.tight_layout()
fig.savefig(OUT / "fig3_class_accuracy.svg", bbox_inches="tight")
plt.close(fig)


# Figure 4: class-level zoning-instability effects.
fig, ax = plt.subplots(figsize=(7.8, 4.9))
vals = cls["pier_change_pp"].to_numpy()
bar_colors = [COL_ROBUST if v <= 0 else COL_BAD for v in vals]
bars = ax.bar(labels, vals, color=bar_colors, width=0.68, edgecolor="white", linewidth=0.6)

ax.axhline(0, color="#555555", linewidth=0.9)
ax.set_xticks(labels)
ax.set_xlabel("LCZ built class")
ax.set_ylabel("PIER change (percentage points)")
ax.set_title("LCZ class-level zoning-instability effects")
clean_axis(ax)
ax.bar_label(bars, labels=[f"{v:+.2f}" for v in vals], padding=2, fontsize=7.2)

ax.legend(
    handles=[
        Patch(facecolor=COL_ROBUST, label="Lower zoning instability"),
        Patch(facecolor=COL_BAD, label="Higher zoning instability"),
    ],
    frameon=False,
    loc="lower right",
)
fig.tight_layout()
fig.savefig(OUT / "fig4_class_pier.svg", bbox_inches="tight")
plt.close(fig)


# Figure 5: mean-city versus worst-city constraint ablation.
cmp = mean_city[
    [
        "outer_city",
        "accuracy_change_pp",
        "baseline_pier",
        "selected_pier",
    ]
].merge(
    robust_city[
        [
            "outer_city",
            "accuracy_change_pp",
            "baseline_pier",
            "selected_pier",
        ]
    ],
    on="outer_city",
    suffixes=("_mean", "_worst"),
)

cmp["pier_change_mean_pp"] = 100 * (
    cmp["selected_pier_mean"] - cmp["baseline_pier_mean"]
)
cmp["pier_change_worst_pp"] = 100 * (
    cmp["selected_pier_worst"] - cmp["baseline_pier_worst"]
)
cmp = cmp.sort_values("accuracy_change_pp_mean", ascending=True).reset_index(drop=True)

ypos = np.arange(len(cmp))
city_labels = [abbr.get(c, c) for c in cmp["outer_city"]]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.8), sharey=True)

# A: predictive preservation.
ax = axes[0]
for i, r in cmp.iterrows():
    ax.plot(
        [r["accuracy_change_pp_mean"], r["accuracy_change_pp_worst"]],
        [i, i],
        color="#BFC5CC",
        linewidth=1.0,
        zorder=1,
    )
ax.scatter(
    cmp["accuracy_change_pp_mean"],
    ypos,
    s=48,
    marker="s",
    color=COL_MEAN,
    edgecolor="white",
    linewidth=0.7,
    label="Mean-city constraint",
    zorder=3,
)
ax.scatter(
    cmp["accuracy_change_pp_worst"],
    ypos,
    s=52,
    marker="o",
    color=COL_ROBUST,
    edgecolor="white",
    linewidth=0.7,
    label="Worst-city constraint",
    zorder=3,
)
ax.axvline(-0.5, color=COL_BAD, linestyle="--", linewidth=1.2)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(city_labels)
ax.set_xlabel("Held-out accuracy change (pp)")
ax.set_title("A  Predictive preservation", loc="left", weight="semibold")
clean_axis(ax, grid_axis="x")

# B: zoning-instability reduction.
ax = axes[1]
for i, r in cmp.iterrows():
    ax.plot(
        [r["pier_change_mean_pp"], r["pier_change_worst_pp"]],
        [i, i],
        color="#BFC5CC",
        linewidth=1.0,
        zorder=1,
    )
ax.scatter(
    cmp["pier_change_mean_pp"],
    ypos,
    s=48,
    marker="s",
    color=COL_MEAN,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
ax.scatter(
    cmp["pier_change_worst_pp"],
    ypos,
    s=52,
    marker="o",
    color=COL_ROBUST,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.set_xlabel("Held-out PIER change (pp)")
ax.set_title("B  Zoning-instability reduction", loc="left", weight="semibold")
clean_axis(ax, grid_axis="x")

fig.suptitle(
    "Constraint ablation across held-out cities",
    fontsize=12.5,
    weight="semibold",
    y=1.01,
)
fig.legend(
    handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_MEAN, markeredgecolor="white", markersize=7, label="Mean-city constraint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_ROBUST, markeredgecolor="white", markersize=7, label="Worst-city constraint"),
        Line2D([0], [0], color=COL_BAD, linestyle="--", linewidth=1.2, label="−0.5 pp boundary"),
    ],
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
)

fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig5_constraint_ablation.svg", bbox_inches="tight")
plt.close(fig)
