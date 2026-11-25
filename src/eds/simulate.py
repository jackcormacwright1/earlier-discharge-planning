"""A synthetic cohort of emergency medical admissions, built so that the thing worth knowing arrives late.

Why simulate at all. The question here is not "how accurate can a length-of-stay model be" — that is a
property of a dataset, and several public datasets would answer it. The question is when enough information
exists to act, which is a property of the *process*: of what is known at admission, what arrives with the
first bloods, and what only becomes apparent when a therapist assesses someone's mobility on day two. No
public dataset lets you vary that schedule, and varying it is the whole point, so the schedule is modelled
explicitly and every conclusion is reported across a range of it.

The structure, which is where all the content is:

  length of stay = clinical days + discharge delay

The clinical part depends on how ill someone is, and illness is measured early: observations and bloods are
back within hours. The delay part depends on whether they can safely go home — whether there is a package
of care, a family member, a stairlift, a placement — and that is assessed late, if at all. A therapy
assessment on day one and a social work referral on day two are the first real signals.

So the two halves of the outcome have opposite information schedules, and the half that drives the long
stays is the half you learn about last. That tension is what the project measures, and it is built in here
rather than discovered: the finding is not that the tension exists but what it costs, and how that cost
moves as the schedule and the shortenable fraction change.

Nothing here is fitted to patient data. The marginal distributions are set to published national figures
(see docs/METHODOLOGY.md) and everything that matters is varied in the sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# Information arrives in tranches. Hours since admission; the names are the decision points a ward round
# would recognise.
TRANCHES = {
    0: "admission",
    6: "first bloods",
    24: "consultant review and therapy assessment",
    48: "day two",
}

SPECIALTIES = ("General medicine", "Respiratory", "Care of the elderly", "Cardiology", "Gastroenterology")
# Median clinical days by specialty, before any discharge delay.
SPECIALTY_DAYS = {"General medicine": 2.6, "Respiratory": 3.4, "Care of the elderly": 5.2,
                  "Cardiology": 2.2, "Gastroenterology": 2.8}
SPECIALTY_SHARE = (0.38, 0.17, 0.25, 0.11, 0.09)


@dataclass(frozen=True)
class Cohort:
    """Everything that defines the simulated population. Defaults are the central case."""

    n: int = 20_000
    seed: int = 0

    # --- the shape of the population
    mean_age: float = 68.0
    sd_age: float = 18.0
    care_home_rate_at_85: float = 0.22     # probability of being a care home resident at age 85
    lives_alone_rate: float = 0.34

    # --- the clinical half
    clinical_sd_log: float = 0.95          # lognormal spread of clinical days
    target_mean_los: float = 8.1           # NAO 2025-26, emergency admissions from A&E

    # --- the discharge-delay half, set to the two published figures it has to reproduce
    delay_rate: float = 0.142              # share of stays involving any delayed discharge (NAO 2025-26)
    delay_median_days: float = 1.712       # solved, with the line below, to hit the two NAO figures
    delay_sd_log: float = 1.457            # (scripts/01_calibrate.py re-derives both from the targets)
    delay_predictability: float = 0.55     # how much of the delay risk is driven by observable complexity
    shortenable_fraction: float = 0.5      # share of delays that earlier planning could actually shorten

    # --- how well each tranche measures what it measures
    bloods_noise: float = 0.75             # noise on the severity signal at 6h
    therapy_noise: float = 0.70            # noise on the complexity signal at 24h
    social_noise: float = 0.40             # noise on the complexity signal at 48h
    admission_complexity_noise: float = 1.25   # how blurred complexity is from admission variables alone

    long_stay_days: int = 7                # the threshold that defines the prediction target

    tranche_hours: tuple = field(default_factory=lambda: tuple(TRANCHES))

    def with_(self, **changes) -> "Cohort":
        """A copy with some parameters changed, for the sensitivity analysis."""
        return replace(self, **changes)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(x: np.ndarray, target_rate: float, slope: float) -> float:
    """The intercept that makes mean(logistic(intercept + slope * x)) equal the target rate. Bisection,
    because the mean of a logistic is not the logistic of the mean and assuming otherwise moves the
    realised rate by several percentage points."""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _logistic(mid + slope * x).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(cfg: Cohort = Cohort()) -> pd.DataFrame:
    """One cohort. Returns a row per patient with the latent truth and every observable, each column
    tagged by the hour it becomes available (see `available_at`)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n

    # ---------------------------------------------------------------- who they are (known at admission)
    age = np.clip(rng.normal(cfg.mean_age, cfg.sd_age, n), 18, 102)
    female = rng.random(n) < 0.52
    specialty = rng.choice(SPECIALTIES, n, p=SPECIALTY_SHARE)
    imd_quintile = rng.integers(1, 6, n)                      # 1 = most deprived
    prior_admissions = rng.poisson(np.exp(-1.6 + 0.022 * (age - 65)), n)
    # care home residence rises steeply with age
    care_home = rng.random(n) < _logistic((age - 85) / 6.0) * cfg.care_home_rate_at_85 / 0.5
    lives_alone = (rng.random(n) < cfg.lives_alone_rate) & ~care_home
    ambulance = rng.random(n) < _logistic((age - 70) / 18.0)

    # ---------------------------------------------------------------- the two latent halves
    # severity: how ill. Drives the clinical stay. Measured early and well.
    severity = rng.normal(0.35 * (age - 68) / 18.0 + 0.25 * prior_admissions.clip(0, 4) / 4.0, 1.0, n)

    # complexity: how hard this person will be to discharge. Drives the delay. Partly foreseeable from
    # admission variables, mostly not: whether a package of care materialises depends on the council's
    # capacity that week, not on anything about the patient.
    foreseeable = (0.9 * (age - 68) / 18.0 + 1.5 * care_home + 0.55 * lives_alone
                   + 0.30 * prior_admissions.clip(0, 4) / 2.0 - 0.12 * (imd_quintile - 3))
    complexity = (cfg.delay_predictability * foreseeable
                  + np.sqrt(max(1e-9, 1 - cfg.delay_predictability ** 2)) * rng.normal(0, 1, n))

    # ---------------------------------------------------------------- length of stay
    median_days = np.array([SPECIALTY_DAYS[s] for s in specialty])
    clinical_days = np.exp(np.log(median_days) + 0.30 * severity
                           + rng.normal(0, cfg.clinical_sd_log, n))

    # A delay happens to a minority, and is much likelier for the complex. The intercept is solved rather
    # than assumed, so the realised rate matches the published one whatever the complexity distribution.
    intercept = _solve_intercept(complexity, cfg.delay_rate, slope=1.35)
    has_delay = rng.random(n) < _logistic(intercept + 1.35 * complexity)
    # Lognormal rather than Gamma: the median delay is a couple of days and the mean is three times that,
    # which is the shape the national figures imply and the shape a ward would recognise.
    delay_days = np.where(
        has_delay,
        np.exp(np.log(cfg.delay_median_days) + 0.35 * complexity
               + rng.normal(0, cfg.delay_sd_log, n)), 0.0)
    # Of the delays, only some could have been shortened by starting sooner; the rest are waiting on
    # capacity that does not exist however early you ask.
    shortenable = has_delay & (rng.random(n) < cfg.shortenable_fraction)

    # The clinical half is scaled so that the cohort's mean length of stay matches the published figure.
    clinical_days *= (cfg.target_mean_los - delay_days.mean()) / clinical_days.mean()

    los = clinical_days + delay_days
    long_stay = los >= cfg.long_stay_days

    # ---------------------------------------------------------------- what is observed, and when
    # 6 hours: bloods and observations. Noisy readings of severity, and nothing about complexity.
    news2 = np.clip(np.round(2.6 + 1.5 * severity + rng.normal(0, cfg.bloods_noise, n)), 0, 18)
    albumin = np.clip(38 - 3.1 * severity + rng.normal(0, 3.4, n), 15, 52)
    crp = np.exp(np.clip(2.6 + 0.75 * severity + rng.normal(0, 0.9, n), 0, 7))
    egfr = np.clip(92 - 0.45 * (age - 65) - 6.5 * severity.clip(0, None) + rng.normal(0, 13, n), 5, 130)
    haemoglobin = np.clip(132 - 7.5 * severity + rng.normal(0, 14, n), 55, 180)

    # 24 hours: consultant review, imaging, and the first therapy assessment. The therapy assessment is
    # the first thing in the record that is about going home rather than about being ill.
    imaging_abnormal = rng.random(n) < _logistic(0.6 * severity - 0.4)
    procedure_planned = rng.random(n) < _logistic(0.5 * severity - 1.3)
    mobility_score = np.clip(np.round(
        6.4 - 1.65 * complexity - 0.022 * (age - 68) + rng.normal(0, cfg.therapy_noise, n)), 0, 10)

    # 48 hours: the discharge-planning signals proper.
    social_work_flag = rng.random(n) < _logistic(1.5 * complexity - 0.9
                                                 + rng.normal(0, cfg.social_noise, n))
    placement_needed = social_work_flag & (rng.random(n) < _logistic(1.1 * complexity - 0.5))
    complication = rng.random(n) < _logistic(0.7 * severity - 2.2)

    return pd.DataFrame({
        # latent truth, never given to a model
        "severity": severity, "complexity": complexity,
        "clinical_days": clinical_days, "delay_days": delay_days,
        "has_delay": has_delay, "shortenable": shortenable,
        "los": los, "long_stay": long_stay,
        # tranche 0
        "age": age, "female": female, "specialty": specialty, "imd_quintile": imd_quintile,
        "prior_admissions": prior_admissions, "care_home": care_home, "lives_alone": lives_alone,
        "ambulance": ambulance,
        # tranche 6h
        "news2": news2, "albumin": albumin, "crp": crp, "egfr": egfr, "haemoglobin": haemoglobin,
        # tranche 24h
        "imaging_abnormal": imaging_abnormal, "procedure_planned": procedure_planned,
        "mobility_score": mobility_score,
        # tranche 48h
        "social_work_flag": social_work_flag, "placement_needed": placement_needed,
        "complication": complication,
    })


# Which hour each observable becomes available. Everything a model may use at hour h is the set of columns
# whose value here is <= h.
AVAILABLE_AT = {
    "age": 0, "female": 0, "specialty": 0, "imd_quintile": 0, "prior_admissions": 0,
    "care_home": 0, "lives_alone": 0, "ambulance": 0,
    "news2": 6, "albumin": 6, "crp": 6, "egfr": 6, "haemoglobin": 6,
    "imaging_abnormal": 24, "procedure_planned": 24, "mobility_score": 24,
    "social_work_flag": 48, "placement_needed": 48, "complication": 48,
}

LATENT = ("severity", "complexity", "clinical_days", "delay_days", "has_delay", "shortenable",
          "los", "long_stay")


def features_at(hour: int) -> list[str]:
    """The columns a model is allowed to use at this many hours after admission."""
    return [c for c, h in AVAILABLE_AT.items() if h <= hour]


def still_in_hospital(df: pd.DataFrame, hour: int) -> np.ndarray:
    """A prediction made at hour h is only useful for patients who are still there. Anyone already
    discharged is neither a candidate for a referral nor a member of the evaluation set."""
    return (df["los"] * 24.0) > hour


def describe(df: pd.DataFrame, cfg: Cohort = Cohort()) -> pd.Series:
    """The realised marginals, for checking the cohort against published figures."""
    delayed = df["delay_days"] > 0
    return pd.Series({
        "n": len(df),
        "median_los": df["los"].median(),
        "mean_los": df["los"].mean(),
        f"share_{cfg.long_stay_days}_plus_days": df["long_stay"].mean(),
        "share_14_plus_days": (df["los"] >= 14).mean(),
        "share_21_plus_days": (df["los"] >= 21).mean(),
        "share_with_any_delay": delayed.mean(),
        "share_of_bed_days_lost_to_delay": df["delay_days"].sum() / df["los"].sum(),
        "share_of_long_stays_that_are_delayed": delayed[df["long_stay"]].mean(),
        "share_of_delayed_who_are_21_plus": (df.loc[delayed, "los"] >= 21).mean(),
        "share_with_delay_7_plus_days": (df["delay_days"] >= 7).mean(),
        "mean_delay_among_delayed": df.loc[delayed, "delay_days"].mean(),
        "share_shortenable": df["shortenable"].mean(),
        "mean_age": df["age"].mean(),
    })
