"""Re-derive the delay distribution from the published targets.

The cohort has to reproduce two figures from the National Audit Office's 2026 report on patient flow
(2025-26, emergency admissions from A&E): 14.2% of stays involve a delayed discharge, and patients well
enough to go home used 14.7% of all bed days. Two targets, two free parameters — the median delay and its
log spread — so they are solved rather than assumed. The mean length of stay is held at 8.1 days by the
simulator itself.

Usage:  PYTHONPATH=src python scripts/01_calibrate.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eds.simulate import Cohort, describe, simulate  # noqa: E402

TARGETS = {"share_of_bed_days_lost_to_delay": 0.147, "share_with_delay_7_plus_days": 0.037}


def main():
    def residuals(theta):
        d = describe(simulate(Cohort(n=40_000, delay_median_days=np.exp(theta[0]),
                                     delay_sd_log=np.exp(theta[1]))))
        return [d[k] - v for k, v in TARGETS.items()]

    fit = least_squares(residuals, [np.log(2.6), np.log(1.5)], diff_step=0.08, xtol=1e-4)
    median, sd_log = np.exp(fit.x)
    print(f"delay_median_days = {median:.3f}\ndelay_sd_log      = {sd_log:.3f}\n")
    print(describe(simulate(Cohort(n=60_000, delay_median_days=median, delay_sd_log=sd_log)))
          .round(3).to_string())


if __name__ == "__main__":
    main()
