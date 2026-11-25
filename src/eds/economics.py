"""What a referral is worth, and when.

A prediction is worth something only if acting on it changes an outcome, so the unit here is not accuracy
but bed days saved net of what the referral costs. Three things set that value, and only the first is
uncontroversial.

**What the intervention does.** The Cochrane review of discharge planning (33 trials, 12,242 participants)
finds individualised discharge planning shortens a medical admission by 0.73 days on average, 95% interval
0.12 to 1.33. That is the effect of doing it at all, for unselected patients.

**How the benefit decays with delay.** Nothing in that literature says what happens if the same planning
starts on day three instead of day one, because no trial randomised the timing. It has to be assumed, so it
is assumed explicitly and varied: the benefit falls linearly over a useful window L, and L -> infinity
recovers the case where timing does not matter at all. Every headline result is reported as a function of L
rather than at one value of it, because this is the assumption the answer turns on.

**What the referral costs.** Under a cash constraint, roughly two hours of a band 6 discharge coordinator
against a bed day worth several hundred pounds — which, as it turns out, makes targeting pointless: you
should simply refer everyone. The constraint that bites in a real hospital is not money but the size of the
discharge team, and under a capacity cap a false positive is no longer cheap, because it displaces a true
one. Both are modelled; the capacity version is the one that matters.

A note on the ceiling. The benefit can only come out of delay that earlier planning could actually remove.
At the central assumptions that pool is smaller than the Cochrane effect implies, so the cap binds and the
published effect size cannot be achieved by shortening delays alone. That is a result rather than a bug,
and `benefit_ceiling` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Economics:
    """The central case. Every field is varied in the sensitivity analysis."""

    bed_day_value: float = 400.0        # marginal, not average: the day saved is a medically-fit day
    referral_cost: float = 70.0         # ~2 hours of a band 6 discharge coordinator, fully loaded
    cochrane_effect_days: float = 0.73  # mean reduction when everyone is referred on admission
    useful_window_days: float = 7.0     # L: the benefit is gone by this many days after admission
    capacity_per_1000: float = 200.0    # referrals the discharge team can absorb per 1,000 admissions


def decay(day: float | np.ndarray, window: float) -> np.ndarray:
    """How much of the benefit survives a referral made this many days after admission."""
    if not np.isfinite(window):
        return np.ones_like(np.asarray(day, dtype=float))
    return np.clip(1.0 - np.asarray(day, dtype=float) / window, 0.0, 1.0)


def benefit_ceiling(df: pd.DataFrame) -> float:
    """The largest mean reduction in length of stay that shortening delays could possibly produce: every
    shortenable delay removed in full, for everyone, immediately."""
    return float((df["delay_days"] * df["shortenable"]).mean())


def effect_per_patient(df: pd.DataFrame, econ: Economics) -> tuple[float, bool]:
    """Days saved for a shortenable patient referred immediately, calibrated so that referring the whole
    cohort on admission reproduces the Cochrane effect.

    Returns the per-patient figure and whether the ceiling binds. When it does, the calibration cannot
    reach the Cochrane effect however large the per-patient saving, because there is not that much
    shortenable delay in the cohort; the saving is then set to remove the delays in full.
    """
    ceiling = benefit_ceiling(df)
    if ceiling <= econ.cochrane_effect_days:
        return float("inf"), True
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        achieved = (np.minimum(df["delay_days"], mid) * df["shortenable"]).mean()
        if achieved < econ.cochrane_effect_days:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def days_saved(df: pd.DataFrame, referred: np.ndarray, hour: float, econ: Economics,
               per_patient: float | None = None) -> np.ndarray:
    """Days saved for each patient, given who was referred and when.

    A referral helps only a patient whose delay was shortenable, and only up to the delay they actually
    had. What is left of the effect by hour h is set by the decay above.
    """
    if per_patient is None:
        per_patient, _ = effect_per_patient(df, econ)
    available = np.minimum(df["delay_days"].to_numpy(), per_patient)
    return np.where(referred & df["shortenable"].to_numpy(),
                    available * float(decay(hour / 24.0, econ.useful_window_days)), 0.0)


def value_of_referring(df: pd.DataFrame, referred: np.ndarray, hour: float, econ: Economics,
                       per_patient: float | None = None) -> dict:
    """Bed days saved, what the referrals cost, and the net, per 1,000 admissions."""
    saved = days_saved(df, referred, hour, econ, per_patient)
    n = len(df)
    per_1000 = 1000.0 / n
    return {
        "hour": hour,
        "referrals": int(referred.sum()),
        "referrals_per_1000": referred.sum() * per_1000,
        "bed_days_saved": saved.sum(),
        "bed_days_saved_per_1000": saved.sum() * per_1000,
        "benefit": saved.sum() * econ.bed_day_value * per_1000,
        "cost": referred.sum() * econ.referral_cost * per_1000,
        "net_benefit": (saved.sum() * econ.bed_day_value - referred.sum() * econ.referral_cost) * per_1000,
        "true_positives": int((referred & df["shortenable"].to_numpy()).sum()),
    }


def top_k_by_risk(risk: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    """Refer the k highest-risk patients who are still in hospital. This is the capacity-constrained
    decision: the team has k slots, and spending one on somebody who did not need it is a slot that a
    patient who did need it does not get."""
    referred = np.zeros(len(risk), dtype=bool)
    candidates = np.flatnonzero(eligible)
    if len(candidates) == 0 or k <= 0:
        return referred
    order = candidates[np.argsort(-risk[candidates], kind="stable")]
    referred[order[:k]] = True
    return referred
