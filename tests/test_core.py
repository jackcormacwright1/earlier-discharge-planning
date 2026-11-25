"""Tests for the parts that could be silently wrong: the cohort's calibration against published figures,
the information schedule, and the arithmetic of what a referral is worth."""

import numpy as np
import pytest

from eds.economics import (Economics, benefit_ceiling, decay, effect_per_patient, top_k_by_risk,
                           value_of_referring)
from eds.predict import fit_at_hour, split
from eds.simulate import AVAILABLE_AT, Cohort, _solve_intercept, describe, features_at, simulate, \
    still_in_hospital

# The figures the cohort exists to reproduce: National Audit Office, patient flow, 2025-26.
PUBLISHED = {"mean_los": 8.1, "share_with_any_delay": 0.142,
             "share_with_delay_7_plus_days": 0.037, "share_of_bed_days_lost_to_delay": 0.147}


@pytest.fixture(scope="module")
def cohort():
    return simulate(Cohort(n=40_000))


def test_cohort_reproduces_the_published_figures(cohort):
    got = describe(cohort)
    for name, target in PUBLISHED.items():
        assert got[name] == pytest.approx(target, abs=0.012), name


def test_intercept_solver_hits_the_rate():
    x = np.random.default_rng(0).normal(0, 1.4, 50_000)
    for target in (0.05, 0.142, 0.4):
        b = _solve_intercept(x, target, slope=1.35)
        assert (1 / (1 + np.exp(-(b + 1.35 * x)))).mean() == pytest.approx(target, abs=1e-4)


def test_no_column_from_a_later_tranche_is_available_early():
    for hour in (0, 6, 24, 48):
        for column in features_at(hour):
            assert AVAILABLE_AT[column] <= hour
    assert "mobility_score" not in features_at(6)      # the therapy assessment is a day-one signal
    assert "mobility_score" in features_at(24)
    assert "social_work_flag" not in features_at(24)
    assert set(features_at(0)) < set(features_at(48))


def test_patients_already_discharged_are_not_candidates(cohort):
    for hour in (0, 24, 48):
        still = still_in_hospital(cohort, hour)
        assert cohort.loc[still, "los"].min() * 24 > hour
        assert (cohort.loc[~still, "los"] * 24 <= hour).all()
    assert still_in_hospital(cohort, 0).all()          # nobody has left at the moment of admission


def test_decay_runs_from_one_to_zero_over_the_window():
    assert decay(0, 7) == pytest.approx(1.0)
    assert decay(3.5, 7) == pytest.approx(0.5)
    assert decay(7, 7) == pytest.approx(0.0)
    assert decay(20, 7) == pytest.approx(0.0)          # never negative
    assert decay(100, np.inf) == pytest.approx(1.0)    # an infinite window is "timing does not matter"


def test_effect_is_calibrated_to_the_published_trial_result():
    """With enough shortenable delay in the cohort, referring everyone on admission must reproduce the
    Cochrane effect exactly — that is what the calibration is for."""
    df = simulate(Cohort(n=30_000, shortenable_fraction=1.0))
    econ = Economics()
    per_patient, binds = effect_per_patient(df, econ)
    assert not binds
    everyone = np.ones(len(df), dtype=bool)
    saved = value_of_referring(df, everyone, 0.0, econ, per_patient)["bed_days_saved"] / len(df)
    assert saved == pytest.approx(econ.cochrane_effect_days, rel=1e-3)


def test_the_ceiling_binds_when_there_is_not_enough_shortenable_delay():
    df = simulate(Cohort(n=20_000, shortenable_fraction=0.1))
    econ = Economics()
    assert benefit_ceiling(df) < econ.cochrane_effect_days
    per_patient, binds = effect_per_patient(df, econ)
    assert binds and np.isinf(per_patient)
    # with the ceiling binding, the most that can be saved is every shortenable delay, in full
    saved = value_of_referring(df, np.ones(len(df), bool), 0.0, econ, per_patient)["bed_days_saved"]
    assert saved / len(df) == pytest.approx(benefit_ceiling(df), rel=1e-9)


def test_capacity_cap_is_respected_and_picks_the_highest_risk_eligible():
    risk = np.array([0.9, 0.1, 0.8, 0.7, 0.95])
    eligible = np.array([True, True, True, False, True])
    chosen = top_k_by_risk(risk, eligible, 2)
    assert chosen.sum() == 2
    assert chosen[4] and chosen[0]            # the two highest among the eligible
    assert not chosen[3]                      # the highest ineligible one is never chosen
    assert top_k_by_risk(risk, eligible, 0).sum() == 0
    assert top_k_by_risk(risk, eligible, 99).sum() == eligible.sum()


def test_a_referral_only_helps_a_shortenable_patient():
    df = simulate(Cohort(n=5_000))
    econ = Economics()
    per_patient, _ = effect_per_patient(df, econ)
    from eds.economics import days_saved
    saved = days_saved(df, np.ones(len(df), bool), 0.0, econ, per_patient)
    assert (saved[~df["shortenable"].to_numpy()] == 0).all()
    assert (saved <= df["delay_days"].to_numpy() + 1e-9).all()


def test_training_on_the_delay_label_beats_the_long_stay_label_on_value():
    """The project's central claim, as a regression test: same features, same learner, different label."""
    df = simulate(Cohort(n=25_000))
    train, test = split(df)
    econ = Economics()
    per_patient, _ = effect_per_patient(df, econ)
    fit = fit_at_hour(train, test, 24)
    cap = int(round(econ.capacity_per_1000 * len(test) / 1000))
    value = {}
    for name in ("delay", "long_stay"):
        referred = top_k_by_risk(fit["risk"][name], fit["eligible"], cap)
        value[name] = value_of_referring(test, referred, 24, econ, per_patient)["bed_days_saved"]
    assert value["delay"] > 1.5 * value["long_stay"]


def _value_at(hour, seed, predictors):
    """Bed days saved per 1,000 for each named predictor, at one decision point, in one cohort."""
    df = simulate(Cohort().with_(seed=seed))
    train, test = split(df, seed=seed)
    econ = Economics()
    per_patient, _ = effect_per_patient(df, econ)
    cap = round(econ.capacity_per_1000 * len(test) / 1000)
    fit = fit_at_hour(train, test, hour, seed=seed)
    return {name: value_of_referring(test, top_k_by_risk(fit["risk"][name], fit["eligible"], cap),
                                     hour, econ, per_patient=per_patient)["bed_days_saved_per_1000"]
            for name in predictors}


def test_the_label_is_a_real_effect_and_the_learner_is_not():
    """Finding 2, across cohorts rather than in one. The label swap has to hold every time; the learner
    swap has to be allowed to go either way, because it does. A version of this test that pinned the
    learner swap as positive would be pinning one lucky draw."""
    label, learner = [], []
    for seed in range(5):
        v = _value_at(24, seed, ("delay", "long_stay", "logistic"))
        label.append(v["delay"] - v["long_stay"])
        learner.append(v["long_stay"] - v["logistic"])
    assert all(d > 50 for d in label), "the label swap should help substantially in every cohort"
    assert min(learner) < 0 < max(learner), "the learner swap should not be a reliable improvement"
    assert np.mean(label) > 5 * abs(np.mean(learner))


def test_at_admission_the_ward_rule_and_the_model_are_a_toss_up():
    """Finding 3, after repetition retracted the stronger version of it. An earlier README claimed the
    ward rule beat every model at admission; across cohorts the difference changes sign, so the claim
    this pins is the weaker one that survived."""
    gaps = [(lambda v: v["simple"] - v["delay"])(_value_at(0, seed, ("simple", "delay")))
            for seed in range(6)]
    assert min(gaps) < 0 < max(gaps), "neither the rule nor the model reliably wins at admission"
    assert abs(np.mean(gaps)) < 40, "and the average difference is small either way"


def test_value_falls_to_nothing_once_the_window_has_closed():
    df = simulate(Cohort(n=5_000))
    econ = Economics(useful_window_days=1.0)
    per_patient, _ = effect_per_patient(df, econ)
    late = value_of_referring(df, np.ones(len(df), bool), 48.0, econ, per_patient)
    assert late["bed_days_saved"] == pytest.approx(0.0)
    assert late["net_benefit"] < 0          # the referrals still cost what they cost
