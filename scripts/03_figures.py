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
PRETTY = {"delay": "Model, delay label", "long_stay": "Model, long-stay label",
          "logistic": "Logistic, long-stay label", "simple": "Ward rule", "everyone": "Unselected"}
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
    window = pd.read_csv(RESULTS / "window.csv")
    sens = pd.read_csv(RESULTS / "sensitivity.csv")

    # The value and accuracy charts are built from the 30 simulated cohorts written by 04_stability.py,
    # not from the single cohort in 02_run.py. One cohort is a draw: its best decision point happened to
    # be the most favourable of thirty, and a chart of it would overstate every difference on it.
    seed_values = pd.read_csv(RESULTS / "seed_values.csv")
    seeds = pd.read_csv(RESULTS / "seeds.csv")
    interaction = pd.read_csv(RESULTS / "interaction.csv")
    value = (seed_values.groupby(["predictor", "hour"])["bed_days_saved_per_1000"]
             .agg(["mean", "std"]).reset_index()
             .rename(columns={"mean": "bed_days_saved_per_1000", "std": "sd"}))
    value["label"] = value["predictor"].map(PRETTY).fillna(value["predictor"])
    accuracy = (seed_values.dropna(subset=["auc"]).groupby(["predictor", "hour"])["auc"]
                .agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "auc", "std": "auc_sd"}))
    accuracy["label"] = accuracy["predictor"].map(PRETTY).fillna(accuracy["predictor"])

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
    x = a["hour"].map(pos)
    axes[0].fill_between(x, a["auc"] - a["auc_sd"], a["auc"] + a["auc_sd"], color=BLUE, alpha=0.15,
                         lw=0)
    axes[0].plot(x, a["auc"], color=BLUE, marker="o", ms=6)
    axes[0].set(ylabel="AUC", title="What the model knows", xticks=ticks, ylim=(0.6, 0.85))
    axes[0].set_xticklabels(ticklabels, fontsize=8)
    axes[0].annotate("waiting from 24h to 48h\nbuys +0.001 of AUC", (0.97, 0.10),
                     xycoords="axes fraction", fontsize=8.5, color=MUTED, ha="right")

    xv = v["hour"].map(pos)
    best = v.loc[v["bed_days_saved_per_1000"].idxmax()]
    axes[1].fill_between(xv, v["bed_days_saved_per_1000"] - v["sd"],
                         v["bed_days_saved_per_1000"] + v["sd"], color=BLUE, alpha=0.15, lw=0)
    axes[1].plot(xv, v["bed_days_saved_per_1000"], color=BLUE, marker="o", ms=6)
    axes[1].plot([pos[best["hour"]]], [best["bed_days_saved_per_1000"]], "o", ms=13, mfc="none",
                 mec=BLUE, mew=2)
    axes[1].annotate("best in 93%\nof cohorts", (pos[best["hour"]], best["bed_days_saved_per_1000"]),
                     textcoords="offset points", xytext=(10, 6), color=BLUE, fontsize=8.5)
    axes[1].set(ylabel="bed days saved per 1,000 admissions", title="What acting on it is worth",
                xticks=ticks)
    axes[1].set_xticklabels(ticklabels, fontsize=8)
    fig.suptitle("Accuracy is flat after 24 hours, and still nominally highest at day two. "
                 "Value peaks at 24\nhours and falls. Mean over 30 simulated cohorts, shaded ±1 sd.",
                 x=0.01, ha="left", y=1.07)
    save(fig, "accuracy_vs_value.png")

    # ------------------------------------------------- 2. the label is worth more than the model
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for label in ORDER:
        g = value[value["label"] == label].sort_values("hour")
        if g.empty:
            continue
        gx = g["hour"].map(pos)
        ax.fill_between(gx, g["bed_days_saved_per_1000"] - g["sd"],
                        g["bed_days_saved_per_1000"] + g["sd"], color=COLOUR[label], alpha=0.11, lw=0)
        ax.plot(gx, g["bed_days_saved_per_1000"], color=COLOUR[label], marker="o", ms=5, label=label)
    ax.set(ylabel="bed days saved per 1,000 admissions", xticks=ticks, ylim=(0, None))
    ax.set_xticklabels(ticklabels, fontsize=8)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.suptitle("Changing the label is worth far more than changing the learner. "
                 "At admission the\nward rule and the model are indistinguishable. "
                 "30 cohorts, shaded ±1 sd.", x=0.01, ha="left", y=1.06)
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

    figure_stability(seeds, interaction, pos, ticklabels)


def figure_stability(seeds, interaction, pos, ticklabels):
    """How often 24 hours wins: across cohorts, and across the two assumptions it turns on."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.32})

    hours = sorted(pos)
    share = seeds["best_hour"].value_counts(normalize=True).reindex(hours).fillna(0.0)
    bars = axes[0].bar(range(len(hours)), share.to_numpy(), width=0.6,
                       color=[BLUE if h == 24 else GREY for h in hours])
    axes[0].bar_label(bars, labels=[f"{s:.0%}" if s else "" for s in share], padding=3, fontsize=9,
                      color=INK)
    axes[0].set(xticks=range(len(hours)), ylim=(0, 1.08), ylabel="share of simulated cohorts",
                title=f"Which decision point wins, over {len(seeds)} cohorts")
    axes[0].set_xticklabels(ticklabels, fontsize=8)

    # the two assumptions the answer turns on, varied together rather than one at a time
    grid = interaction.pivot(index="therapy_noise", columns="useful_window_days",
                             values="modal_best_hour")
    axes[1].imshow([[0 if v == 24 else 1 for v in row] for row in grid.to_numpy()],
                   cmap=matplotlib.colors.ListedColormap([BLUE, GREEN]), aspect="auto",
                   vmin=0, vmax=1)
    for i, noise in enumerate(grid.index):
        for j, window in enumerate(grid.columns):
            axes[1].text(j, i, f"{int(grid.loc[noise, window])}h", ha="center", va="center",
                         color="white", fontsize=10, fontweight="bold")
    axes[1].set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns], fontsize=8.5)
    axes[1].set_yticks(range(len(grid.index)), [f"{i:g}" for i in grid.index], fontsize=8.5)
    axes[1].set(xlabel="useful window L (days)  →  slower decay",
                ylabel="therapy noise  →  blunter 24h signal",
                title="Varying both assumptions together")
    axes[1].grid(visible=False)

    fig.suptitle("Waiting for the therapy assessment wins in 93% of cohorts, and in 10 of 12 "
                 "assumption pairs", x=0.01, ha="left", y=1.03)
    save(fig, "stability.png")


if __name__ == "__main__":
    main()
