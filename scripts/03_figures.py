"""The charts, from the files written by 02_run.py.

Usage:  PYTHONPATH=src python scripts/03_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

INK, MUTED, GRID = "#1d1d1b", "#6b6a66", "#e1e0d9"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#b4b2ab"
COLOUR = {"Model, delay label": BLUE, "Model, long-stay label": ORANGE, "Ward rule": GREEN,
          "Logistic, long-stay label": PURPLE, "Unselected": GREY}
ORDER = ["Model, delay label", "Ward rule", "Model, long-stay label", "Logistic, long-stay label",
         "Unselected"]
HOUR_LABEL = {0: "admission", 6: "first bloods\n6h", 24: "therapy assessment\n24h", 48: "day two\n48h"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 160, "lines.linewidth": 2.0,
})


def save(fig, name):
    fig.savefig(RESULTS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def main():
    accuracy = pd.read_csv(RESULTS / "accuracy.csv")
    value = pd.read_csv(RESULTS / "value.csv")
    window = pd.read_csv(RESULTS / "window.csv")
    sens = pd.read_csv(RESULTS / "sensitivity.csv")

    hours = sorted(value["hour"].unique())
    # The decision points are a sequence, not a continuous axis: evenly spaced so the two that are six
    # hours apart do not collide at one end while a two-day gap stretches out at the other.
    pos = {h: i for i, h in enumerate(hours)}
    ticks = list(pos.values())
    ticklabels = [HOUR_LABEL[h] for h in hours]

    # ------------------------------------------------- 1. accuracy and value point different ways
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a = accuracy[accuracy["label"] == "Model, delay label"].sort_values("hour")
    v = value[value["label"] == "Model, delay label"].sort_values("hour")
    axes[0].plot(a["hour"].map(pos), a["auc"], color=BLUE, marker="o", ms=6)
    axes[0].set(ylabel="AUC", title="What the model knows", xticks=ticks, ylim=(0.6, 0.85))
    axes[0].set_xticklabels(ticklabels, fontsize=8)
    best = v.loc[v["bed_days_saved_per_1000"].idxmax()]
    axes[1].plot(v["hour"].map(pos), v["bed_days_saved_per_1000"], color=BLUE, marker="o", ms=6)
    axes[1].plot([pos[best["hour"]]], [best["bed_days_saved_per_1000"]], "o", ms=13, mfc="none",
                 mec=BLUE, mew=2)
    axes[1].annotate("best", (pos[best["hour"]], best["bed_days_saved_per_1000"]),
                     textcoords="offset points", xytext=(12, 4), color=BLUE, fontsize=9)
    axes[1].set(ylabel="bed days saved per 1,000 admissions", title="What acting on it is worth",
                xticks=ticks)
    axes[1].set_xticklabels(ticklabels, fontsize=8)
    fig.suptitle("Accuracy rises to the end. Value peaks a day in, and then falls.", x=0.01, ha="left",
                 y=1.04)
    save(fig, "accuracy_vs_value.png")

    # ------------------------------------------------- 2. the label is worth more than the model
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for label in ORDER:
        g = value[value["label"] == label].sort_values("hour")
        if g.empty:
            continue
        ax.plot(g["hour"].map(pos), g["bed_days_saved_per_1000"], color=COLOUR[label], marker="o",
                ms=5, label=label)
    ax.set(ylabel="bed days saved per 1,000 admissions", xticks=ticks, ylim=(0, None))
    ax.set_xticklabels(ticklabels, fontsize=8)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.suptitle("Changing the label beats changing the model, and at admission a ward rule beats both",
                 x=0.01, ha="left", y=1.02)
    save(fig, "predictors.png")

    # ------------------------------------------------- 3. the crossover in the decay window
    w = window[window["predictor"] == "delay"].copy()
    finite = w[np.isfinite(w["useful_window_days"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for hour, colour in zip(hours, [GREY, GREEN, BLUE, ORANGE]):
        axes[0].plot(finite["useful_window_days"], finite[f"h{int(hour)}"], color=colour, marker="o",
                     ms=4, label=HOUR_LABEL[hour].replace("\n", " "))
    axes[0].legend(fontsize=8.5, title="referral made at", title_fontsize=8.5)
    axes[0].set(xlabel="useful window L (days)", ylabel="bed days saved per 1,000",
                title="Value of each decision point")
    axes[1].step(finite["useful_window_days"], finite["best_hour"].map(pos), where="post", color=BLUE)
    axes[1].plot(finite["useful_window_days"], finite["best_hour"].map(pos), "o", color=BLUE, ms=5)
    axes[1].set(xlabel="useful window L (days)", ylabel=None, title="When to make the call",
                yticks=ticks, ylim=(-0.4, len(ticks) - 0.6))
    axes[1].set_yticklabels([t.replace("\n", " ") for t in ticklabels], fontsize=8.5)
    fig.suptitle("Unless the window is under three days, wait for the therapy assessment "
                 "— but not beyond it", x=0.01, ha="left", y=1.04)
    save(fig, "window.png")

    # ------------------------------------------------- 4. sensitivity
    s = sens[sens["predictor"] == "delay"].set_index("variation")
    central = s.loc["central", "bed_days_per_1000"]
    s = s.drop(index="central").sort_values("bed_days_per_1000")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(s))
    ax.barh(y, s["bed_days_per_1000"] - central, left=central,
            color=[BLUE if v >= central else ORANGE for v in s["bed_days_per_1000"]], height=0.65)
    ax.axvline(central, color=INK, lw=1.2)
    ax.set_yticks(y, s.index, fontsize=9)
    ax.set(xlabel="bed days saved per 1,000 admissions", title=None)
    ax.grid(axis="y", visible=False)
    ax.annotate(f"central\n{central:.0f}", (central, len(s) - 0.2), textcoords="offset points",
                xytext=(6, 0), fontsize=8.5, color=MUTED)
    for yi, (v, h) in enumerate(zip(s["bed_days_per_1000"], s["best_hour"])):
        ax.text(max(v, central) + 8, yi, f"{int(h)}h", va="center", fontsize=8, color=MUTED)
    fig.suptitle("One assumption at a time, with the best decision point beside each bar", x=0.01,
                 ha="left", y=0.98)
    save(fig, "sensitivity.png")


if __name__ == "__main__":
    main()
