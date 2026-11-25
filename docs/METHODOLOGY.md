# Methodology

## The question

A hospital that could tell, on day one, which patients will be hard to discharge could start the referrals
early. The usual way to approach that is to build a length-of-stay model and report its accuracy. That is
the wrong shape of answer, because accuracy is not what a bed meeting needs: it needs to know whether to
act now on a worse prediction or wait for a better one, and the answer depends on how fast a referral loses
its usefulness — which no prediction model knows anything about.

So this project measures two curves rather than one. How much the model knows, against hours since
admission. And how much acting on it is worth, against the same axis. They do not peak in the same place.

## The cohort

### Why it is simulated

What is being studied is the **arrival schedule of information**, not the predictability of a particular
dataset. A real extract fixes that schedule at whatever one hospital happened to do; the question needs it
varied. MIMIC-IV's demo (100 patients, open, no credentialing) grounds the timing of labs and assessments
relative to admission, and everything above that is modelled and swept.

The discipline that makes this legitimate rather than circular: the marginals are pinned to published
national figures, and no conclusion rests on a point estimate of anything that was assumed. Every
assumption is in a dataclass, varied one at a time, and reported in `results/sensitivity.csv`.

### Structure

```
length of stay = clinical days + discharge delay
```

Two latent variables drive the two halves. **Severity** drives the clinical stay and is measured early and
well: observations and bloods are back within hours. **Complexity** drives the delay and is measured late:
the first real signal is a therapy assessment of mobility at 24 hours, and the clearest one is a social
work referral at 48.

That is the tension the project measures. The half of the outcome that produces the long stays is the half
you learn about last, and the action that would help has to be taken early.

Complexity is `delay_predictability` correlated with what is observable at admission — age, care home
residence, living alone, prior admissions — and otherwise independent of everything. Whether a package of
care materialises depends on the council's capacity that week, not on the patient.

### Calibration

Four published figures, from the National Audit Office's 2026 report on patient flow (2025-26, emergency
admissions from A&E):

| | |
|---|---|
| Mean length of stay | 8.1 days |
| Share of stays with any delayed discharge | 14.2% |
| Share with a delay of 7+ days | 3.7% |
| Share of all bed days used by patients well enough to go home | 14.7% |

The mean is held by scaling the clinical half. The delay distribution is lognormal — median under two days,
long right tail — with its median and log spread **solved** against the last two targets rather than chosen
(`scripts/01_calibrate.py`). The logit intercept for who gets a delay at all is solved by bisection, since
the mean of a logistic is not the logistic of a mean and assuming otherwise moves the realised rate by
several percentage points.

The share of stays reaching seven days (34%) is an output, not a target. If you have a local figure, it is
one line to re-point the cohort at it.

## The information schedule

| Hour | What arrives | What it measures |
|---|---|---|
| 0 | age, sex, specialty, deprivation, prior admissions, care home, lives alone, ambulance | complexity, blurred |
| 6 | NEWS2, albumin, CRP, eGFR, haemoglobin | severity, well |
| 24 | imaging, planned procedure, **therapy mobility assessment** | severity, and complexity for the first time |
| 48 | complication, **social work flag**, placement needed | complexity, clearly |

Two rules are enforced and tested. A model for hour *h* may use a column only if that column exists by hour
*h*. And it may be evaluated only on patients still in hospital at hour *h* — which matters as much, since
by day two a third of the cohort has gone home and those left are systematically different. A model that
looks better at 48 hours may simply be facing a different population.

## Predictors

- **Unselected** — refer whoever comes first. Under a capacity cap this is what "refer everyone" becomes,
  and it is the honest comparator, since the Cochrane trials referred unselected patients and it worked.
- **Ward rule** — age, care home residence, prior admissions, living alone. No fitting.
- **Model, long-stay label** — gradient boosting predicting a stay of 7+ days.
- **Model, delay label** — gradient boosting, identical features, predicting a delayed discharge.
- **Logistic, long-stay label** — to show the comparison is about the label, not the learner.

Reported with discrimination, Brier score and calibration slope, because a model used at a threshold has to
be calibrated and most published ones are not checked.

## What a referral is worth

```
days saved = min(delay, effect) x decay(day, L)   for a shortenable patient, 0 otherwise
decay(d, L) = max(0, 1 - d/L)
```

`effect` is calibrated so that referring the whole cohort on admission reproduces the Cochrane result of
0.73 days. `L` is the useful window, and it is the assumption with no evidence behind it: no trial
randomised the timing of discharge planning, only whether it happened. Results are therefore reported
across L from 2 days to infinity rather than at one value.

**The ceiling.** The benefit can only come out of delay that earlier planning could remove. At the central
assumptions the cohort holds 0.656 days per patient of shortenable delay against a Cochrane effect of 0.73,
so the cap binds: the published effect is larger than the pool it would have to come from. The calibration
reports this rather than silently exceeding it, and reaching the Cochrane figure needs the shortenable
fraction to be 56% rather than the assumed 50%.

**The cap, not the cost.** At £70 a referral against a £400 bed day, a false positive is nearly free and
the optimal policy is to refer everyone. That is not the situation in a hospital, where the constraint is
the number of discharge coordinators. Under a cap of *k* referrals the problem becomes a ranking one, a
false positive displaces a true one, and the timing trade-off bites. Both versions are computed; the
capacity one is reported.

## What this design cannot tell you

**It cannot tell you the effect size.** Everything scales with the Cochrane estimate, which comes from
trials of unselected medical inpatients in other health systems and other decades. If it does not transfer,
the levels here are wrong — though the *shape* of the result, which is about the relative timing of two
curves, survives.

**It cannot tell you the decay.** L is assumed. The headline is a statement about where the optimum sits as
a function of L, not a claim about L itself. Someone with trust data on referral-to-assessment intervals
could pin it down, and that is the single most valuable thing anyone could add.

**It cannot tell you what a real model would achieve.** The predictability of discharge complexity is a
parameter here. Set it low and no model helps; set it high and the decision moves later. The published
range is wide and the simulation spans it rather than resolving it.

**It is not a trial.** Nothing here establishes that acting earlier works. It establishes what would follow
*if* it worked in the way the trial literature suggests, and which decisions that conclusion is sensitive
to. The honest summary is that the case for an early flag rests on an unmeasured parameter, and that the
most useful next step is measuring it rather than improving the model.
