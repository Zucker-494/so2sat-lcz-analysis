#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = Path("figures/final")
OUT.mkdir(parents=True, exist_ok=True)

for old in OUT.glob("*.svg"):
    old.unlink()

COL_DISCRETE = "#4477AA"
COL_CONTINUOUS = "#EE7733"
COL_COMBINED = "#AA3377"
COL_POSITIVE = "#228833"
COL_MEAN_RULE = "#CCBB44"
COL_NEGATIVE = "#CC3311"
COL_NEUTRAL = "#6B7280"
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

SAVE = {"dpi": 600, "bbox_inches": "tight"}

city = pd.read_csv("results/final_hgb_confirmation/city_summary.csv")
city = city[city["delta_pp"] == 0.5].copy()

cls = pd.read_csv("results/final_hgb_confirmation/class_aggregate_summary.csv")
cls = cls[cls["delta_pp"] == 0.5].copy()

mean_city = pd.read_csv("results/nested_accuracy_constrained/outer_city_results.csv")
mean_city = mean_city[
    (mean_city["model"] == "hgb") & (mean_city["delta_pp"] == 0.5)
].copy()

max_city = pd.read_csv("results/robust_nested_accuracy_constraint/outer_city_results.csv")
max_city = max_city[
    (max_city["model"] == "hgb") & (max_city["delta_pp"] == 0.5)
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


# Figure 1
fig, ax = plt.subplots(figsize=(11.2, 4.9))
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 5.1)
ax.axis("off")

boxes = [
    (
        0.25,
        1.50,
        2.10,
        2.00,
        "Fixed spatial support\n320 m × 320 m\nBuildings and LCZ fixed",
        COL_NEUTRAL,
        "#F8FAFC",
    ),
    (
        3.00,
        2.80,
        2.25,
        1.35,
        "Discrete zoning\n4 × 4 morphology\n9 zonings",
        COL_DISCRETE,
        "#EAF1F8",
    ),
    (
        3.00,
        0.80,
        2.25,
        1.35,
        "Continuous spatial\nrepresentation\n16 fixed anchors",
        COL_CONTINUOUS,
        "#FDF0E6",
    ),
    (
        6.05,
        1.50,
        2.05,
        2.00,
        "Weighted class probabilities\n(1 − α) p_h + α p_c",
        COL_COMBINED,
        "#F5EAF2",
    ),
    (
        8.80,
        1.35,
        2.15,
        2.30,
        "Nested validation by city\nminimize zoning sensitivity\nmaximum city accuracy\nloss ≤ 0.5 pp",
        COL_POSITIVE,
        "#EAF5EC",
    ),
]

for x, y, w, h, label, edge, face in boxes:
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
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)

for start, end in [
    ((2.35, 2.50), (3.00, 3.45)),
    ((2.35, 2.50), (3.00, 1.47)),
    ((5.25, 3.45), (6.05, 2.82)),
    ((5.25, 1.47), (6.05, 2.18)),
    ((8.10, 2.50), (8.80, 2.50)),
]:
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
    "Reducing zoning sensitivity under fixed spatial support",
    ha="center",
    fontsize=12.5,
    weight="semibold",
)

fig.tight_layout()
fig.savefig(OUT / "fig1_method.png", **SAVE)
plt.close(fig)


# Figure 2
fig, ax = plt.subplots(figsize=(7.6, 6.1))
x = city["accuracy_change_pp"].to_numpy()
y = 100 * (city["selected_zoning_sensitivity"] - city["baseline_zoning_sensitivity"]).to_numpy()

point_colors = [
    COL_POSITIVE if value >= 0 else (COL_MEAN_RULE if value >= -0.5 else COL_NEGATIVE)
    for value in x
]

ax.axvspan(-0.5, max(0.8, x.max() + 0.25), color=COL_POSITIVE, alpha=0.06, zorder=0)
ax.scatter(x, y, s=58, c=point_colors, edgecolors="white", linewidths=0.8, zorder=3)

for _, row in city.iterrows():
    ax.annotate(
        abbr.get(row["city"], row["city"]),
        (
            row["accuracy_change_pp"],
            100 * (row["selected_zoning_sensitivity"] - row["baseline_zoning_sensitivity"]),
        ),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax.axvline(-0.5, color=COL_NEGATIVE, linestyle="--", linewidth=1.2)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.axhline(0, color="#555555", linewidth=0.9)
ax.set_xlabel("Accuracy change in outer city (percentage points)")
ax.set_ylabel("Change in zoning sensitivity in outer city (percentage points)")
ax.set_title("Accuracy and zoning sensitivity across cities")
clean_axis(ax, grid_axis="both")

ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COL_POSITIVE,
            markeredgecolor="white",
            markersize=7,
            label="Accuracy unchanged or higher",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COL_MEAN_RULE,
            markeredgecolor="white",
            markersize=7,
            label="Accuracy decrease within 0.5 pp",
        ),
        Line2D(
            [0],
            [0],
            color=COL_NEGATIVE,
            linestyle="--",
            linewidth=1.2,
            label="0.5 pp accuracy limit",
        ),
    ],
    frameon=False,
    loc="lower left",
)

fig.tight_layout()
fig.savefig(OUT / "fig2_city_results.png", **SAVE)
plt.close(fig)


# Figure 3
fig, ax = plt.subplots(figsize=(7.8, 5.3))
labels = cls["label"].astype(int).to_numpy()
vals = cls["accuracy_change_pp"].to_numpy()
bar_colors = [COL_POSITIVE if value >= 0 else COL_NEGATIVE for value in vals]
bars = ax.bar(
    labels,
    vals,
    color=bar_colors,
    width=0.68,
    edgecolor="white",
    linewidth=0.6,
)

ax.axhline(0, color="#555555", linewidth=0.9)
ax.set_xticks(labels)
ax.set_xlabel("LCZ built class")
ax.set_ylabel("Accuracy change (percentage points)")
ax.set_title("Accuracy change by LCZ class", pad=12)
clean_axis(ax)

for bar, value in zip(bars, vals):
    x0 = bar.get_x() + bar.get_width() / 2
    offset = 4 if value >= 0 else -12
    ax.annotate(
        f"{value:+.2f}",
        (x0, value),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=7.3,
    )

fig.legend(
    handles=[
        Patch(facecolor=COL_POSITIVE, label="Increase"),
        Patch(facecolor=COL_NEGATIVE, label="Decrease"),
    ],
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
)

fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(OUT / "fig3_class_accuracy.png", **SAVE)
plt.close(fig)


# Figure 4
fig, ax = plt.subplots(figsize=(7.8, 5.3))
vals = cls["zoning_sensitivity_change_pp"].to_numpy()
bar_colors = [COL_POSITIVE if value <= 0 else COL_NEGATIVE for value in vals]
bars = ax.bar(
    labels,
    vals,
    color=bar_colors,
    width=0.68,
    edgecolor="white",
    linewidth=0.6,
)

ax.axhline(0, color="#555555", linewidth=0.9)
ax.set_xticks(labels)
ax.set_xlabel("LCZ built class")
ax.set_ylabel("Change in zoning sensitivity (percentage points)")
ax.set_title("Change in zoning sensitivity by LCZ class", pad=12)
clean_axis(ax)

for bar, value in zip(bars, vals):
    x0 = bar.get_x() + bar.get_width() / 2
    offset = 4 if value >= 0 else -12
    ax.annotate(
        f"{value:+.2f}",
        (x0, value),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=7.3,
    )

fig.legend(
    handles=[
        Patch(facecolor=COL_POSITIVE, label="Lower zoning sensitivity"),
        Patch(facecolor=COL_NEGATIVE, label="Higher zoning sensitivity"),
    ],
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
)

fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(OUT / "fig4_class_zoning_sensitivity.png", **SAVE)
plt.close(fig)


# Figure 5
cmp = mean_city[
    ["outer_city", "accuracy_change_pp", "baseline_zoning_sensitivity", "selected_zoning_sensitivity"]
].merge(
    max_city[["outer_city", "accuracy_change_pp", "baseline_zoning_sensitivity", "selected_zoning_sensitivity"]],
    on="outer_city",
    suffixes=("_mean", "_max"),
)

cmp["zoning_sensitivity_change_mean_pp"] = 100 * (
    cmp["selected_zoning_sensitivity_mean"] - cmp["baseline_zoning_sensitivity_mean"]
)
cmp["zoning_sensitivity_change_max_pp"] = 100 * (
    cmp["selected_zoning_sensitivity_max"] - cmp["baseline_zoning_sensitivity_max"]
)
cmp = cmp.sort_values("accuracy_change_pp_mean", ascending=True).reset_index(drop=True)

ypos = np.arange(len(cmp))
city_labels = [abbr.get(city_name, city_name) for city_name in cmp["outer_city"]]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.8), sharey=True)

ax = axes[0]
for i, row in cmp.iterrows():
    ax.plot(
        [row["accuracy_change_pp_mean"], row["accuracy_change_pp_max"]],
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
    color=COL_MEAN_RULE,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
ax.scatter(
    cmp["accuracy_change_pp_max"],
    ypos,
    s=52,
    marker="o",
    color=COL_POSITIVE,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
ax.axvline(-0.5, color=COL_NEGATIVE, linestyle="--", linewidth=1.2)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(city_labels)
ax.set_xlabel("Accuracy change in outer city (pp)")
ax.set_title("A  Accuracy", loc="left", weight="semibold")
clean_axis(ax, grid_axis="x")

ax = axes[1]
for i, row in cmp.iterrows():
    ax.plot(
        [row["zoning_sensitivity_change_mean_pp"], row["zoning_sensitivity_change_max_pp"]],
        [i, i],
        color="#BFC5CC",
        linewidth=1.0,
        zorder=1,
    )
ax.scatter(
    cmp["zoning_sensitivity_change_mean_pp"],
    ypos,
    s=48,
    marker="s",
    color=COL_MEAN_RULE,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
ax.scatter(
    cmp["zoning_sensitivity_change_max_pp"],
    ypos,
    s=52,
    marker="o",
    color=COL_POSITIVE,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.set_xlabel("Change in zoning sensitivity in outer city (pp)")
ax.set_title("B  Zoning sensitivity", loc="left", weight="semibold")
clean_axis(ax, grid_axis="x")

fig.suptitle(
    "Comparison of accuracy rules across cities",
    fontsize=12.5,
    weight="semibold",
    y=1.01,
)

fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COL_MEAN_RULE,
            markeredgecolor="white",
            markersize=7,
            label="Mean accuracy rule",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COL_POSITIVE,
            markeredgecolor="white",
            markersize=7,
            label="Maximum city loss rule",
        ),
        Line2D(
            [0],
            [0],
            color=COL_NEGATIVE,
            linestyle="--",
            linewidth=1.2,
            label="−0.5 pp accuracy limit",
        ),
    ],
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
)

fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig5_accuracy_rule_comparison.png", **SAVE)
plt.close(fig)
