"""Everything the write-up quotes, written to results/.

  results/cohort.csv        the simulated cohort's marginals against the published targets
  results/accuracy.csv      discrimination and calibration by decision point and predictor
  results/value.csv         bed days saved and net benefit by decision point and predictor
  results/window.csv        the same, swept over the useful window L — the crossover analysis
  results/sensitivity.csv   the optimal decision point under each assumption, one at a time
  results/ceiling.csv       how much shortenable delay the cohort contains, against the Cochrane effect

Usage:  PYTHONPATH=src python scripts/02_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eds.economics import (Economics, benefit_ceiling, effect_per_patient,  # noqa: E402
                           top_k_by_risk, value_of_referring)
from eds.predict import fit_at_hour, split  # noqa: E402
from eds.simulate import TRANCHES, Cohort, describe, simulate  # noqa: E402

RESULTS = ROOT / "results"
TARGETS = {"mean_los": 8.1, "share_with_any_delay": 0.142,
           "share_with_delay_7_plus_days": 0.037, "share_of_bed_days_lost_to_delay": 0.147}
PRETTY = {"delay": "Model, delay label", "long_stay": "Model, long-stay label",
          "logistic": "Logistic, long-stay label", "simple": "Ward rule", "everyone": "Unselected"}


def fitted(cfg: Cohort, seed: int = 0):
    """Simulate, split, and fit a model at each decision point. Returns the test set and the fits."""
    df = simulate(cfg.with_(seed=seed))
    train, test = split(df, seed=seed)
    return df, test, {h: fit_at_hour(train, test, h, seed=seed) for h in TRANCHES}


def value_table(df: pd.DataFrame, test: pd.DataFrame, fits: dict, econ: Economics) -> pd.DataFrame:
    """Bed days saved and net benefit for every predictor at every decision point, under the capacity cap."""
    per_patient, _ = effect_per_patient(df, econ)
    cap = int(round(econ.capacity_per_1000 * len(test) / 1000))
    rows = []
    for hour, fit in fits.items():
        for name, risk in fit["risk"].items():
            referred = top_k_by_risk(risk, fit["eligible"], cap)
            v = value_of_referring(test, referred, hour, econ, per_patient=per_patient)
            rows.append({"predictor": name, "label": PRETTY.get(name, name), **v})
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg, econ = Cohort(), Economics()

    # ---------------------------------------------------------------- the cohort
    df, test, fits = fitted(cfg)
    marginals = describe(df, cfg)
    cohort = pd.DataFrame({"realised": marginals,
                           "published_target": pd.Series(TARGETS)}).reindex(marginals.index)
    cohort.to_csv(RESULTS / "cohort.csv")
    print(cohort.round(3).to_string(), "\n")

    # ---------------------------------------------------------------- the ceiling
    ceiling = benefit_ceiling(df)
    _, binds = effect_per_patient(df, econ)
    pd.DataFrame([{"shortenable_delay_days_per_patient": ceiling,
                   "cochrane_effect_days": econ.cochrane_effect_days,
                   "ceiling_binds": binds,
                   "shortenable_fraction_needed": econ.cochrane_effect_days / ceiling
                   * cfg.shortenable_fraction}]).to_csv(RESULTS / "ceiling.csv", index=False)
    print(f"shortenable delay available: {ceiling:.3f} days/patient against a Cochrane effect of "
          f"{econ.cochrane_effect_days}; ceiling binds: {binds}\n")

    # ---------------------------------------------------------------- accuracy and value
    accuracy = pd.concat([f["metrics"] for f in fits.values()], ignore_index=True)
    accuracy["label"] = accuracy["predictor"].map(PRETTY)
    accuracy.to_csv(RESULTS / "accuracy.csv", index=False)

    value = value_table(df, test, fits, econ)
    value.to_csv(RESULTS / "value.csv", index=False)
    print(value.pivot(index="label", columns="hour", values="bed_days_saved_per_1000")
          .round(1).to_string(), "\n")
    print(accuracy.pivot(index="label", columns="hour", values="auc").round(3).to_string(), "\n")

    # ---------------------------------------------------------------- the crossover in L
    # The decay window is the assumption with no evidence behind it, so the headline result is reported
    # as a function of it rather than at one value.
    windows = [2, 3, 4, 5, 7, 10, 14, 21, np.inf]
    rows = []
    for window in windows:
        v = value_table(df, test, fits, econ.__class__(**{**econ.__dict__, "useful_window_days": window}))
        for predictor, g in v.groupby("predictor"):
            best = g.loc[g["bed_days_saved_per_1000"].idxmax()]
            rows.append({"useful_window_days": window, "predictor": predictor,
                         "label": PRETTY.get(predictor, predictor),
                         "best_hour": best["hour"],
                         "best_bed_days_per_1000": best["bed_days_saved_per_1000"]})
            for _, r in g.iterrows():
                rows[-1][f"h{int(r['hour'])}"] = r["bed_days_saved_per_1000"]
    window_table = pd.DataFrame(rows)
    window_table.to_csv(RESULTS / "window.csv", index=False)
    print(window_table[window_table["predictor"] == "delay"]
          [["useful_window_days", "h0", "h6", "h24", "h48", "best_hour"]].round(1).to_string(index=False),
          "\n")

    # ---------------------------------------------------------------- one-at-a-time sensitivity
    variations = {
        "central": ({}, {}),
        "bed day £250": ({}, {"bed_day_value": 250.0}),
        "bed day £800": ({}, {"bed_day_value": 800.0}),
        "Cochrane 0.12 days": ({}, {"cochrane_effect_days": 0.12}),
        "Cochrane 1.33 days": ({}, {"cochrane_effect_days": 1.33}),
        "capacity 100/1000": ({}, {"capacity_per_1000": 100.0}),
        "capacity 400/1000": ({}, {"capacity_per_1000": 400.0}),
        "shortenable 0.25": ({"shortenable_fraction": 0.25}, {}),
        "shortenable 0.75": ({"shortenable_fraction": 0.75}, {}),
        "delay 30% predictable": ({"delay_predictability": 0.30}, {}),
        "delay 80% predictable": ({"delay_predictability": 0.80}, {}),
        "therapy signal noisier": ({"therapy_noise": 1.40}, {}),
    }
    rows = []
    for name, (cohort_changes, econ_changes) in variations.items():
        c = cfg.with_(**cohort_changes) if cohort_changes else cfg
        e = econ.__class__(**{**econ.__dict__, **econ_changes})
        d2, t2, f2 = (df, test, fits) if not cohort_changes else fitted(c)
        v = value_table(d2, t2, f2, e)
        for predictor in ("delay", "simple"):
            g = v[v["predictor"] == predictor]
            best = g.loc[g["bed_days_saved_per_1000"].idxmax()]
            rows.append({"variation": name, "predictor": predictor,
                         "best_hour": best["hour"],
                         "bed_days_per_1000": best["bed_days_saved_per_1000"],
                         "net_benefit": best["net_benefit"]})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(RESULTS / "sensitivity.csv", index=False)
    print(sensitivity.pivot(index="variation", columns="predictor",
                            values=["best_hour", "bed_days_per_1000"]).round(1).to_string())
    print("\nwritten to results/")


if __name__ == "__main__":
    main()
