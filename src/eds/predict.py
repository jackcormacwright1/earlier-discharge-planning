"""One model per decision point, trained only on what a ward would actually have by then.

The rule enforced throughout is that a model built for hour h may use a column only if that column exists
by hour h, and may be evaluated only on patients who are still in hospital at hour h. The second half
matters as much as the first and is easy to forget: by day two a good share of the cohort has gone home,
and the ones left are systematically sicker, so a model that looks better at 48 hours may simply be facing
an easier question on a different population.

Four predictors at each point:

  everyone     refer every patient still in hospital — under a capacity cap this becomes "refer whoever
               turns up first", which is the honest version of the Cochrane trials' unselected referral
  simple       a rule a ward could apply without a computer: age, care home residence, prior admissions
  long_stay    gradient boosting trained on the convenient label: will this stay reach seven days
  delay        gradient boosting trained on the label that matters: will this discharge be delayed

The last two are the point. A long stay is the label that is always to hand, and it is mostly driven by
how ill someone is. The benefit of earlier discharge planning accrues to a different group: the people
whose discharge will be held up by something other than their illness. The two groups overlap but are not
the same, and a model aimed at the first is not aimed at the second. Both are trained here, on identical
features, so the cost of using the convenient label can be read off directly.

The target is binary rather than days because the decision is binary, because days are so right-skewed
that mean error is set by a handful of outliers, and because a threshold is what a capacity-constrained
referral list needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from .simulate import features_at, still_in_hospital

CATEGORICAL = ("specialty",)


def split(df: pd.DataFrame, test_fraction: float = 0.35, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test = rng.random(len(df)) < test_fraction
    return df.loc[~test].reset_index(drop=True), df.loc[test].reset_index(drop=True)


def design(df: pd.DataFrame, columns: list[str], encoder: OneHotEncoder | None = None):
    """Numeric design matrix. Categorical columns are one-hot encoded with the training set's categories."""
    cats = [c for c in columns if c in CATEGORICAL]
    nums = [c for c in columns if c not in CATEGORICAL]
    X = df[nums].astype(float).to_numpy()
    if cats:
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[cats])
        X = np.c_[X, encoder.transform(df[cats])]
    return X, encoder


def simple_rule(df: pd.DataFrame) -> np.ndarray:
    """A ward rule, used as the baseline a model has to beat: older, from a care home, admitted often
    before. Scored rather than binary so it can be ranked under a capacity cap."""
    return (0.03 * (df["age"].to_numpy() - 65)
            + 1.0 * df["care_home"].to_numpy()
            + 0.25 * np.minimum(df["prior_admissions"].to_numpy(), 4)
            + 0.3 * df["lives_alone"].to_numpy())


TARGETS = {"long_stay": "long_stay", "delay": "has_delay"}


def fit_at_hour(train: pd.DataFrame, test: pd.DataFrame, hour: int, seed: int = 0) -> dict:
    """Train on the patients still in hospital at this hour, and predict for the same group in the test
    set. Returns risk scores for every test row (zero for those already discharged) and the usual
    discrimination and calibration measures over the eligible ones."""
    cols = features_at(hour)
    tr = train.loc[still_in_hospital(train, hour)]
    eligible = still_in_hospital(test, hour)
    te = test.loc[eligible]

    out = {"hour": hour, "eligible": eligible, "n_train": len(tr), "n_test": int(eligible.sum()),
           "features": cols}
    if len(tr) < 200 or tr["long_stay"].nunique() < 2 or len(te) == 0:
        return out | {"risk": {}, "metrics": pd.DataFrame()}

    X_tr, enc = design(tr, cols)
    X_te, _ = design(te, cols, enc)
    risks, rows = {}, []
    for name, target in TARGETS.items():
        y_tr, y_te = tr[target].to_numpy(), te[target].to_numpy()
        if len(np.unique(y_tr)) < 2:
            continue
        estimator = HistGradientBoostingClassifier(max_depth=4, max_iter=220, learning_rate=0.07,
                                                   random_state=seed)
        estimator.fit(X_tr, y_tr)
        p = estimator.predict_proba(X_te)[:, 1]
        risks[name] = p
        rows.append(_metrics(name, y_te, p) | {"target": target})

    # the same features, fitted by logistic regression, to show the comparison is not about the learner
    y_tr, y_te = tr["long_stay"].to_numpy(), te["long_stay"].to_numpy()
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    logistic = LogisticRegression(max_iter=4000).fit((X_tr - mu) / sd, y_tr)
    p = logistic.predict_proba((X_te - mu) / sd)[:, 1]
    risks["logistic"] = p
    rows.append(_metrics("logistic", y_te, p) | {"target": "long_stay"})

    rule = simple_rule(te)
    risks["simple"] = rule
    rows.append(_metrics("simple", te["has_delay"].to_numpy(),
                         1 / (1 + np.exp(-(rule - rule.mean()))), calibrated=False)
                | {"target": "has_delay"})
    risks["everyone"] = np.zeros(len(te))      # no ranking: under a cap, whoever comes first
    rows.append({"predictor": "everyone", "target": "none", "auc": 0.5, "brier": np.nan,
                 "calibration_slope": np.nan, "prevalence": float(te["has_delay"].mean())})

    full = {}
    for name, p in risks.items():
        v = np.zeros(len(test))
        v[np.flatnonzero(eligible)] = p
        full[name] = v
    return out | {"risk": full, "metrics": pd.DataFrame(rows).assign(hour=hour)}


def _metrics(name: str, y: np.ndarray, p: np.ndarray, calibrated: bool = True) -> dict:
    row = {"predictor": name, "auc": float(roc_auc_score(y, p)), "prevalence": float(y.mean())}
    if calibrated:
        row["brier"] = float(brier_score_loss(y, p))
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        fit = LogisticRegression(max_iter=2000).fit(logit.reshape(-1, 1), y)
        row["calibration_slope"] = float(fit.coef_[0][0])
    else:
        row["brier"] = np.nan
        row["calibration_slope"] = np.nan
    return row
