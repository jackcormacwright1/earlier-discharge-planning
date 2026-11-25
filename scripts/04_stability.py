"""How stable is the chosen decision point?

`02_run.py` picks the best decision point from one simulated cohort, with every assumption at its central
value, and `03_figures.py` varies assumptions one at a time. Neither answers the question that decides
whether "24 hours" is a finding or an artefact of one draw: how often does 24 hours win?

Two things are varied here.

  **Seeds.** The same configuration, resimulated and refitted from scratch. This separates the choice of
  decision point from the particular cohort and the particular train/test split that produced it.

  **The two assumptions the answer actually turns on, together.** How clear the therapy assessment is
  about discharge complexity (`therapy_noise`, the 24-hour signal) and how fast the benefit of planning
  decays (`useful_window_days`). One-at-a-time sensitivity cannot see an interaction between them, and an
  interaction is exactly what is expected: a sharper 24-hour signal is worth more when the window is
  short and less when it is long. The grid is run over seeds too, so each cell reports how often a
  decision point wins rather than which one won once.

Writes:
  results/seeds.csv        the winning decision point and its value, per seed
  results/interaction.csv  the winning decision point over the signal x window grid

Usage:  PYTHONPATH=src python scripts/04_stability.py [--seeds 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eds.economics import (Economics, effect_per_patient,  # noqa: E402
                           top_k_by_risk, value_of_referring)
from eds.predict import fit_at_hour, split  # noqa: E402
from eds.simulate import TRANCHES, Cohort, simulate  # noqa: E402

RESULTS = ROOT / "results"
PREDICTOR = "delay"          # the predictor the headline is about
THERAPY_NOISE = (0.40, 0.70, 1.00)        # sharper, central, blunter 24-hour signal
WINDOWS = (3.0, 5.0, 7.0, 10.0)           # how fast the benefit decays


def values_for_seed(cfg: Cohort, econ: Economics, seed: int) -> pd.DataFrame:
    """Bed days saved per 1,000 for every predictor at every decision point, for one simulated cohort."""
    df = simulate(cfg.with_(seed=seed))
    train, test = split(df, seed=seed)
    per_patient, _ = effect_per_patient(df, econ)
    cap = int(round(econ.capacity_per_1000 * len(test) / 1000))
    rows = []
    for hour in TRANCHES:
        fit = fit_at_hour(train, test, hour, seed=seed)
        auc = (fit["metrics"].set_index("predictor")["auc"].to_dict()
               if len(fit["metrics"]) else {})
        for name, risk in fit["risk"].items():
            referred = top_k_by_risk(risk, fit["eligible"], cap)
            value = value_of_referring(test, referred, hour, econ, per_patient=per_patient)
            rows.append({"seed": seed, "predictor": name, "hour": hour,
                         "bed_days_saved_per_1000": value["bed_days_saved_per_1000"],
                         "auc": auc.get(name, float("nan"))})
    return pd.DataFrame(rows)


def best_hour(values: pd.DataFrame, predictor: str = PREDICTOR) -> dict:
    """The decision point with the highest value for one predictor, from one seed's value table."""
    series = (values[values["predictor"] == predictor]
              .set_index("hour")["bed_days_saved_per_1000"])
    winner = series.idxmax()
    return {"best_hour": winner, "best_value": series[winner],
            **{f"h{int(h)}": series[h] for h in series.index}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ the same world, different draws
    per_seed = [values_for_seed(Cohort(), Economics(), seed) for seed in range(args.seeds)]
    everything = pd.concat(per_seed, ignore_index=True)
    everything.to_csv(RESULTS / "seed_values.csv", index=False)

    # every predictor at every decision point, averaged over cohorts rather than read off one
    summary = (everything.groupby(["predictor", "hour"])["bed_days_saved_per_1000"]
               .agg(["mean", "std", "min", "max"]).reset_index())
    summary.to_csv(RESULTS / "seed_predictors.csv", index=False)
    print("bed days saved per 1,000, mean over seeds (one seed in brackets, as 02_run.py reports it):")
    one = everything[everything["seed"] == 0].set_index(["predictor", "hour"])[
        "bed_days_saved_per_1000"]
    for predictor, group in summary.groupby("predictor"):
        best = group.loc[group["mean"].idxmax()]
        print(f"  {predictor:10s} best at {int(best['hour']):3d}h: "
              f"{best['mean']:6.1f} (sd {best['std']:4.1f})"
              f"   seed 0 said {one[(predictor, best['hour'])]:6.1f}")
    print()

    seeds = pd.DataFrame([{"seed": seed, **best_hour(values)}
                          for seed, values in zip(range(args.seeds), per_seed)])
    seeds.to_csv(RESULTS / "seeds.csv", index=False)
    share = seeds["best_hour"].value_counts(normalize=True).sort_index()
    print(f"over {args.seeds} seeds at the central assumptions, the best decision point was:")
    for hour, fraction in share.items():
        print(f"  {int(hour):3d}h  {fraction:5.0%}")
    spread = seeds[[f"h{int(h)}" for h in TRANCHES]]
    print("\nbed days saved per 1,000, mean (sd) across seeds:")
    for hour in TRANCHES:
        column = spread[f"h{int(hour)}"]
        print(f"  {int(hour):3d}h  {column.mean():6.1f} ({column.std():.1f})")
    gap = spread["h24"] - spread[[c for c in spread.columns if c != "h24"]].max(axis=1)
    print(f"\n24h beat the best alternative in {(gap > 0).mean():.0%} of seeds, "
          f"by {gap.mean():.1f} bed days per 1,000 on average")

    # ----------------------------------------------- the three headline comparisons, paired within seed
    # Paired, because the cohort draw moves every predictor together: comparing two means across seeds
    # throws away that pairing and makes a reliable difference look like noise, or the reverse.
    wide = everything.pivot_table(index="seed", columns=["predictor", "hour"],
                                  values="bed_days_saved_per_1000")
    comparisons = {
        "label: delay vs long-stay, both at 24h":
            wide[("delay", 24.0)] - wide[("long_stay", 24.0)],
        "learner: boosted vs logistic, both on long-stay at 24h":
            wide[("long_stay", 24.0)] - wide[("logistic", 24.0)],
        "ward rule vs the best model, both at admission":
            wide[("simple", 0.0)] - wide[("delay", 0.0)],
        "best model at 24h vs the ward rule at admission":
            wide[("delay", 24.0)] - wide[("simple", 0.0)],
    }
    paired = pd.DataFrame([{"comparison": name, "mean_difference": d.mean(), "sd": d.std(),
                            "share_positive": (d > 0).mean()}
                           for name, d in comparisons.items()])
    paired.to_csv(RESULTS / "paired.csv", index=False)
    print("\npaired differences within seed, bed days saved per 1,000:")
    for _, r in paired.iterrows():
        print(f"  {r['comparison']:-<56s} {r['mean_difference']:+7.1f} "
              f"(sd {r['sd']:5.1f})  positive in {r['share_positive']:.0%} of seeds")

    auc = (everything.dropna(subset=["auc"]).groupby(["predictor", "hour"])["auc"]
           .agg(["mean", "std"]).reset_index())
    auc.to_csv(RESULTS / "seed_auc.csv", index=False)
    d = auc[auc["predictor"] == PREDICTOR].set_index("hour")
    print("\ndiscrimination for the delay model, mean over seeds:")
    print("  " + "   ".join(f"{int(h)}h {r['mean']:.3f}" for h, r in d.iterrows()))

    # ------------------------------------------------- the signal and the decay window, varied together
    grid = []
    for noise in THERAPY_NOISE:
        for window in WINDOWS:
            cfg = Cohort(therapy_noise=noise)
            econ = Economics(useful_window_days=window)
            winners = [best_hour(values_for_seed(cfg, econ, seed))["best_hour"]
                       for seed in range(max(5, args.seeds // 4))]
            counts = pd.Series(winners).value_counts(normalize=True)
            grid.append({"therapy_noise": noise, "useful_window_days": window,
                         "modal_best_hour": counts.idxmax(), "modal_share": counts.max(),
                         **{f"share_h{int(h)}": counts.get(h, 0.0) for h in TRANCHES}})
    interaction = pd.DataFrame(grid)
    interaction.to_csv(RESULTS / "interaction.csv", index=False)
    print("\nwhich decision point wins, varying the 24-hour signal and the decay window together:")
    print(interaction.pivot(index="therapy_noise", columns="useful_window_days",
                            values="modal_best_hour").to_string())
    print("\n(therapy_noise rises = blunter 24-hour signal; useful_window_days rises = slower decay)")
    print("\nwritten to results/")


if __name__ == "__main__":
    main()
