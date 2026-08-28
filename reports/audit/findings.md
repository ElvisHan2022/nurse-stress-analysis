
### A1 · Inventory and headers

**Ran:** `tasks/audit_a1.py` at 2026-08-28 13:58

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| subjects | 15 | 15 | yes |
| session directories | 609 | 609 | yes |
| distinct rates per signal | ACC=1; BVP=1; EDA=1; HR=1; TEMP=1 | 1 each | yes |
| sessions with nonzero t0 spread | 609 | 0 | no |
| HR t0 offset vs EDA (median s) | 10.0 | 10 (documented) | see notes |

**Figures:** `(none - tabular section)`

**What this changes.** Fixes the native rate of every channel, which sets the ceiling for the analysis-rate choice. Feeds the resampling decision (JC14).

**Surprises.** 609 sessions have nonzero t0 spread


### A2 · Session table, duration, person-days

**Ran:** `tasks/audit_a2.py` at 2026-08-28 13:58

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| sessions | 609 | 609 | yes |
| sensor hours | 1,251.7 | 1,255 | yes |
| median session (h) | 1.167 | 1.17 | yes |
| sessions < 5 min | 92 | 92 | yes |
| person-days | 250 | n.a. (new) | n.a. |
| sessions crossing local midnight | 21 | n.a. (new) | n.a. |

**Figures:** `figures/audit/A2_wear_raster.png`, `figures/audit/A2_duration_ecdf.png`, `figures/audit/A2b_person_days.png`

**What this changes.** Fixes the session inventory and the effective denominator. The person-day count bounds how much the same-day eligibility rule (JC18) can supply.

**Surprises.** none


### A3 · Survey audit

**Ran:** `tasks/audit_a3.py` at 2026-08-28 13:58

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| survey rows | 358 | 358 | yes |
| exact duplicates | 3 | 3 | yes |
| unrated | 113 | 113 | yes |
| level 0 (low) | 46 | 46 | yes |
| level 1 (medium) | 20 | 20 | yes |
| level 2 (high) | 179 | 179 | yes |
| events > 60 min | 19 | 34 | NO |
| max duration (min) | 323 | 323 | yes |
| overlapping pairs | 12 | 12 | yes |

**Figures:** `figures/audit/A3_duration_ecdf_by_level.png`

**What this changes.** Fixes the label inventory and the effective positive count. Feeds JC06 (unrated events), JC07 (level encoding), JC08 (overlaps), JC10 (max duration).

**Surprises.** events > 60 min: 19 vs 34


### A4 · Timezone alignment

**Ran:** `tasks/audit_a4.py` at 2026-08-28 13:58

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| hypothesis 1: survey is UTC | 50 (20.4%) | low | n.a. |
| hypothesis 2: naive Chicago -> UTC | 212 (86.5%) | high | n.a. |
| resolved | America/Chicago (DST-aware) | America/Chicago | yes |
| best fixed offset | UTC-5 | UTC-5 | yes |
| DST-aware advantage (events) | 14 | >0 | yes |
| rated events with no sensor coverage | 33 | n.a. | n.a. |

**Figures:** `figures/audit/A4_timezone_sweep.png`

**What this changes.** Fixes the join used by every later section. Written to reports/audit/timezone_resolution.md.

**Surprises.** 33 rated events fall outside every session even under the winning hypothesis - they have no sensor coverage at all.


### A10 · Event-triggered average

**Ran:** `tasks/audit_a10.py` at 2026-08-28 14:00

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| level-2 events with usable coverage | 141 / 178 | n.a. | n.a. |
| EDA per-event shift (0-10 min vs baseline) | +9.7498 +/- 3.1465 SE | visible rise | yes |
| HR per-event shift (0-10 min vs baseline) | +0.0062 +/- 0.0613 SE | visible rise | NO |
| EDA events with positive shift | 64.3% | >50% | yes |
| HR events with positive shift | 48.6% | >50% | NO |

**Figures:** `figures/audit/A10_event_triggered_average.png`, `figures/audit/A10_event_triggered_by_subject.png`

**What this changes.** Decides whether the labels mark a physiologically distinguishable state. The shape also indicates where the informative part of the interval sits, which is the empirical input to boundary trimming (JC09).

**Surprises.** none


### A10b · Robustness of the event-triggered average

**Ran:** `tasks/audit_a10b.py` at 2026-08-28 14:02

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| EDA mean dz (outlier-sensitive) | 9.7498 | n.a. | n.a. |
| HR mean dz (outlier-sensitive) | 0.0062 | n.a. | n.a. |
| EDA MEDIAN dz (robust) | 0.5678 | n.a. | n.a. |
| HR MEDIAN dz (robust) | -0.0665 | n.a. | n.a. |
| EDA events positive | 64.3% | >50% | yes |
| HR events positive | 48.6% | >50% | NO |
| EDA sign-test p | 0.000913 | <0.05 | yes |
| HR sign-test p | 0.8 | <0.05 | NO |

**Figures:** `figures/audit/A10b_event_triggered_median.png`, `figures/audit/A10b_per_event_shift.png`

**What this changes.** Determines whether the A10 result survives outlier-resistant statistics. The pooled mean is not a defensible summary when the causal-z denominator can approach zero; the median and sign test are.

**Surprises.** the A10 pooled EDA mean of +9.75 is dominated by a few events; the median is +0.5678


### A5 · IBI structure and HRV feasibility

**Ran:** `tasks/audit_a5.py` at 2026-08-28 14:03

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| time-convention agreement | 0.8829 | high (>0.9) | NO |
| sessions with >=1 usable run | 294/609 | 75/105 sampled | n.a. (full archive here) |
| median usable runs per session | 0 | 3 | NO |
| WINDOW-level coverage (120s) | 5.94% | unknown - this is the new number | n.a. |
| median session-level coverage | 0.81% | n.a. | n.a. |

**Figures:** `figures/audit/A5_hrv_window_coverage.png`, `figures/audit/A5_poincare.png`

**What this changes.** Feeds decision point 7 (HRV in the baseline feature set or demoted to an ablation) and JC15/JC17.

**Surprises.** session-level coverage (48% of sessions have a run) is far more flattering than window-level (5.9%).


### A6 · Missingness and informativeness

**Ran:** `tasks/audit_a6.py` at 2026-08-28 14:07

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| windows | 37,252 | n.a. | n.a. |
| level-2 windows | 2,203 (5.91%) | n.a. | n.a. |
| HRV missing overall | 94.06% | n.a. | n.a. |
| LOSO AUC from missingness alone | 0.4679 | >0.5 means informative | no |
| median per-subject AUC | 0.4996 | n.a. | n.a. |

**Figures:** `figures/audit/A6_hrv_availability_by_activity.png`

**What this changes.** If informative, HRV availability is a label proxy and tree models will exploit it. Feeds decision point 7 alongside A5's coverage number.

**Surprises.** not meaningfully informative


### A7 · Labelling propensity and SCAR

**Ran:** `tasks/audit_a7.py` at 2026-08-28 14:08

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| propensity AUC (cross-validated) | 0.7067 | 0.728 | yes |
| p_labelled min | 0.1111 | 0.125 | yes |
| p_labelled max | 0.5957 | 0.724 | NO |
| spread ratio | 5.36x | 5.8x | yes |

**Figures:** `figures/audit/A7_propensity_by_subject.png`, `figures/audit/A7_time_budget_by_subject.png`

**What this changes.** Determines whether naive PU's ranking guarantee holds. If SCAR fails, nnPU must be subject-stratified and a pooled threshold is not defensible.

**Surprises.** computed here for the first time; PLAN.md quoted these as 'derived' with no source in the repository


### A8 · Effective sample size and negative pool

**Ran:** `tasks/audit_a8.py` at 2026-08-28 14:09

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| windows | 37,252 | ~4,869 (post-exclusion) | n.a. |
| level-2 events | 178 | 178 | yes |
| subjects | 15 | 14 | 15 pre-exclusion |
| aggregate ratio | 4.83:1 | 3.6:1 | NO |
| subjects below 3:1 | 5 of 15 | 5 of 14 | yes |
| minimum achievable ratio | 0.00:1 | 0.9:1 | n.a. |

**Figures:** `figures/audit/A8_achievable_ratio.png`, `figures/audit/A8_events_vs_minutes.png`

**What this changes.** Determines the negative ratio setting (JC21) and must be run before it is fixed. Feeds decision point 8.

**Surprises.** aggregate 4.83:1 conceals 5 subjects below 3:1 and 1 below 1:1


### A9 · Attrition and rule order

**Ran:** `tasks/audit_a9.py` at 2026-08-28 14:11

**Numbers**

| Quantity | Observed | Expected | Match |
|---|---|---|---|
| windows entering | 37,252 | n.a. | n.a. |
| windows surviving (order A) | 36,535 | n.a. | n.a. |
| windows surviving (order B) | 36,535 | n.a. | n.a. |
| window-level order effect | 0 (rules commute) | unknown | n.a. |
| sessions kept, length-first | 517 | n.a. | n.a. |
| sessions kept, non-wear-first | 504 | n.a. | n.a. |
| session-level order effect | 13 sessions | unknown | n.a. |
| final level-2 events | 176 | 178 | NO |

**Figures:** `figures/audit/A9_attrition.png`

**What this changes.** Records the exclusion sequence and whether order is itself a judgment call. Window-level masks commute; the session-length rule does not.

**Surprises.** 13 sessions change status depending on whether non-wear is removed before or after the length cutoff

