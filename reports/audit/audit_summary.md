# Audit summary — merged source of truth

Reconciles two independent lines of work on the same archive:

- **The audit** — `AUDIT.md` sections A1–A10 plus A5b, A5c, A5d and A11, over all
  609 sessions. Scripts in `tasks/audit_a*.py`, detail in `findings.md`, figures
  in `figures/audit/`.
- **Plan v2** — `PLAN_v2.md`, whose Phase 0 facts come from Eric's exploration
  notebook and a 45-session stratified diagnostic sample.

Where they agree, the fact is settled. Where they disagree, the disagreement is
resolved here with a number rather than left open. **This file supersedes the
Phase 0 tables in both `PLAN.md` and `PLAN_v2.md`.**

Last updated 2026-08-29 against `Eric/nurse_stress_exploration.ipynb` at `96387c2`.

---

## 1. Settled facts

Confirmed independently by both lines of work. Carry these forward without
re-deriving.

| Fact | Value | Source |
|---|---|---|
| Subjects / sessions | 15 / 609 | both |
| Sensor time | 1,251.7 h (v2 quotes 1,255) | both |
| Median session length | 1.17 h | both |
| Sessions under 5 min | 92 | both |
| Person-days | **250** (8 for 6D to 26 for CE) | audit A2b |
| Survey rows | 358 — 245 rated, 113 unrated | both |
| Stress levels | 46 low / 20 medium / 179 high | both |
| Subjects with zero medium events | 9 of 15 | both |
| Exact duplicate survey rows | 3 | both |
| Overlapping event pairs | 12 | both |
| Longest event | 323 min | both |
| **Level-2 events** | **178** | both |
| Timezone | survey `America/Chicago` DST-aware, sensors UTC | both |
| Native rates | ACC 32, BVP 64, EDA 4, HR 1, TEMP 4 Hz | both |
| ICC(subject) at n=7 | temp 0.522, eda 0.132, acc 0.160, hr 0.127 | both, to 3 d.p. |
| Drop subject 6D | 2 events, **0 eligible negatives** | both, independently |

**Timezone evidence** (audit A4, 245 rated events): survey-as-UTC places 50
(20.4%) inside a same-subject session; naive Chicago places **212 (86.5%)**. Best
fixed offset is UTC−5 at 198; the DST-aware named zone beats it by 14 because the
archive spans April to December. **Use the named zone, never a fixed offset.**
33 rated events (13.5%) have no sensor coverage at all under the winning
hypothesis; four of those sit within one minute of a session edge.

**Window length is settled at 120 s / 60 s hop, twice over.** v2's argument is the
stronger one: no labelled event is shorter than 120 s, and 300 s would discard 15
events outright. Independently, the ultra-short HRV literature finds 120 s
sufficient for RMSSD and SDNN (Muñoz 2015, near-perfect agreement against a
240–300 s reference; Orini 2023 validates RMSSD from ≤15 s ECGs). The classical
300 s figure is a Task Force recording convention, not a requirement here.

---

## 2. Where the two lines disagreed

### 2.0 HRV — the units were never the same

**`min_run` means beats in Eric's notebook and seconds in the audit.** His own
comment states it: *"min_run=30 beats is LENIENT — conventional short-term HRV
uses ~5 min (~300 beats)."* At the measured median valid interval of 0.75 s:

| `min_run` (beats) | ≈ seconds | Audit equivalent |
|---|---|---|
| 30 | 22 s | `MIN_RUN_S = 22` |
| 100 | 75 s | `MIN_RUN_S = 75` |
| **120** | **90 s** | `MIN_RUN_S = 90` |
| 300 | 225 s | `MIN_RUN_S = 225` |

Commit `96387c2` is titled *"Updated nurse exploration with 120s run"*, but
`min_run=120` is **~90 s, not 120 s**. Worth confirming which was intended.

Reimplementing Eric's exact validity rule — intervals from `np.diff(t)`,
`iv_range` (0.4, 1.5), Malik 0.20, plus his run-level HR and RMSSD plausibility
gates — and applying the audit's denominator over all 609 sessions (A5d):

| `min_run` | ≈ s | Subjects with a run in **best session** | Sessions ≥1 run | **120 s windows covered** | Subjects ≥100 windows |
|---|---|---|---|---|---|
| **30** | 22 s | 15/15 | 341/609 (56%) | **5.70%** | **7 of 15** |
| 100 | 75 s | 14/15 | 110/609 (18%) | 1.27% | 0 of 15 |
| **120** | **90 s** | 14/15 | 83/609 (14%) | **0.87%** | **0 of 15** |
| 300 | 225 s | 4/15 | 8/609 (1%) | 0.14% | 0 of 15 |

Two conclusions, and they point opposite ways:

1. **At `min_run=30` beats, HRV is genuinely usable for about half the cohort** —
   7 of 15 subjects clear 100 covered windows (DF 375, 83 339, BG 337, CE 253,
   5C 250, E4 134, F5 133). This is a real concession to v2: HRV is not dead.
2. **Commit `96387c2` tightens to `min_run=120` and that destroys it** — 0.87%
   coverage, zero subjects clearing the threshold. Nothing in the ultra-short HRV
   literature requires 90 s; Muñoz (2015) finds 120 s sufficient and even a single
   10 s window valid for RMSSD, and Orini (2023) validates RMSSD from ≤15 s.

**Recommendation: go looser, not tighter.** `min_run=30` beats, RMSSD only, as an
ablation restricted to the seven subjects with real coverage. Report which
subjects those are rather than carrying an availability flag across a cohort where
half of it is structurally empty.

**A tension this creates with §3.2.** DF and CE are two of the five best HRV
subjects (375 and 253 covered windows) *and* two of the four flagged for flat EDA.
Excluding them on EDA grounds removes the best HRV coverage in the cohort. The two
exclusion criteria disagree about the same people, and that has to be decided
explicitly rather than by whichever rule runs first.

### 2.1 HRV at the audit's original thresholds — still resolved against v2

v2 upgrades HRV from "probably unusable" to "viable for most subjects" and states
that v1 was wrong. **The audit does not support the upgrade**, and the reason is a
denominator, not a bug.

v2's evidence is session-level: *33/45 sessions have ≥1 plausible run*, and
*min_run=100 → 11/15 subjects qualify **in their best session***. Both are
maxima-flavoured statistics. The modelling substrate is the 120 s window.

Measured over all 609 sessions (A5c):

| min_run | Sessions with ≥1 usable run | **120 s windows covered** | Flattery ratio |
|---|---|---|---|
| 30 s | 294/609 = 48.3% | **5.88%** | 8.2× |
| 100 s (v2's primary) | 72/609 = 11.8% | **0.27%** | 44× |
| 300 s | 5/609 = 0.8% | 0.00% | — |

At v2's own `min_run=100`, **100 windows out of 37,252 carry HRV**. Per subject,
the best is 5C with 35 covered windows, then BG with 33; **zero of fifteen
subjects reach v2's own Phase 2 falsification threshold of ≥100 windows.**

Three things were checked to make sure this is not an artifact of the audit's
method:

1. **Ectopic handling is not the cause.** The audit splits a run at every Malik
   violation, which is more conservative than the conventional drop-the-beat-and-
   continue. Rerunning both ways: 5.94% vs 5.88%. Immaterial.
2. **The time convention is correct.** `t[i] − t[i−1] == ibi[i]` within 50 ms for
   88.3% of 812,794 consecutive pairs across 482 sessions. An off-by-one would
   show near zero.
3. **v2's per-subject run counts do not reproduce on the full archive.** They
   correlate ρ=0.31 with the per-subject median but ρ=0.59 with the per-subject
   **maximum** — the signature of a diagnostic sample stratified toward better
   sessions.

**Resolution: HRV is an ablation, not a baseline feature.** Restrict to RMSSD
(which outperforms SDNN at short durations in every ultra-short study), drop
frequency-domain measures entirely, and use `min_run=30` rather than 100 — still
conservative against the literature and it is the difference between 5.9% and
0.27% coverage. Total usable beat data in the archive is **39.5 h of 1,251.7 h
(3.16%)** at 30 s/20 beats, or 57.3 h (4.58%) at a lenient 20 s/15 beats. Window
choice only redistributes this; it cannot create more.

v2's segmentation-bug fix may well be real. It does not change the conclusion.

### 2.2 Smaller divergences

| Quantity | Plan v2 | Audit | Reading |
|---|---|---|---|
| Propensity AUC | 0.728 | **0.707 CV**, 0.751 in-sample | v2's figure is in-sample; use the CV one |
| P(labelled) spread | 5.8× | **5.36×** | conclusion unchanged |
| Overlapping *session* pairs | 7 | **1** | unresolved, low stakes |
| Eligible negative hours | 293.3 | **354.5** | different eligibility implementations |
| Aggregate negative ratio | 3.6 : 1 | **4.83 : 1** | — |
| Events over 60 min | 34 | **19 rated** | 34 is the all-events count; both `AUDIT.md` A3 and v2 §6.4 repeat this conflation |
| Per-subject median EDA range | 26× | **15×** | v2 used the 45-session sample |

**The SCAR refinement that matters.** An ablation on the propensity model shows
session **duration** predicts labelling better (AUC 0.719) than subject identity
(AUC 0.630). A meaningful share of the "5.8× SCAR violation" is the mechanical
fact that longer recordings contain more events. Subject-stratified nnPU remains
the right call; the justification should cite 0.63 for the subject component, not
0.728 overall.

---

## 3. Facts the audit added

### 3.1 The event-triggered average — the result that decides the project

Absent from both plan versions. **This is the check that asks whether the labels
mark anything physiological at all**, and it should be Phase 0's first row.

140 level-2 events with usable coverage, aligned at onset, 0–10 min post-onset
against the preceding 30 min:

| Channel | Mean Δz | **Median Δz** | Events positive | Sign test | Median raw shift |
|---|---|---|---|---|---|
| **EDA** | +9.75 | **+0.568** | **90/140 = 64.3%** | **p = 0.0009** | **+0.64 µS** on a 0.68 µS baseline |
| HR | +0.006 | −0.067 | 68/140 = 48.6% | p = 0.80 | +0.47 bpm |

EDA survives every robust test (Wilcoxon p = 3.3×10⁻⁶) — roughly a doubling of
skin conductance. **HR is a coin flip.** The problem is well posed, and it is well
posed *in EDA*. Any result driven by heart rate deserves suspicion.

**Do not use the pooled mean.** It reports +9.75 because five events out of 140
contribute 63% of the total, the largest at +318 causal-z units. The cause is a
live bug — see §5.

### 3.2 Near-floor EDA subjects — v2 was right, and it is now decisive

The audit missed this; v2 caught it. Checking v2's four flagged subjects against
the full archive rather than the 45-session sample confirms every one:

| Subject | v2 median | Full archive | **IQR / median** |
|---|---|---|---|
| 7E | 0.09 µS | 0.074 | **0.60** |
| DF | 0.07 µS | 0.092 | **0.44** |
| EG | 0.10 µS | 0.099 | **0.72** |
| CE | 0.10 µS | 0.106 | **0.48** |
| *(all eleven others)* | — | 0.18 – 1.08 | **1.04 – 5.14** |

The last column settles it. A low median alone is not disqualifying — a small but
*varying* signal still carries information after causal normalisation. These four
are the **only** subjects whose relative variation falls below 1.0, and there is
no overlap with the rest of the cohort. Their EDA is not merely small, it is flat.

Combined with §3.1 — EDA is the only channel that responds at onset — a subject
with flat EDA cannot contribute signal. **Recommend dropping all four.**

Phase 2 budget under each option (A11):

| Option | Subjects | Level-2 windows | Neg. hours | Aggregate | Min achievable | Below 3:1 |
|---|---|---|---|---|---|---|
| Keep all 15 | 15 | 2,203 | 354.5 | 4.83:1 | 0.00 | 5 |
| Drop 6D only | 14 | 2,181 | 354.5 | 4.88:1 | 1.87 | 4 |
| **Drop 6D + all four** | **10** | **1,579** | **265.2** | **5.04:1** | **2.14** | **2** |
| Drop 6D + DF, 7E | 12 | 1,927 | 290.8 | 4.53:1 | 1.87 | 4 |

Dropping all four costs **exactly 20 of 178 events (11%)**, matching v2's estimate.
It also *improves* the negative budget: minimum achievable ratio rises 1.87 → 2.14
and subjects below 3:1 halve. **Recommended `RATIO` = 2.14** on the ten-subject
cohort, applied uniformly.

### 3.3 Other additions

- **Sample-rate feasibility.** 1 Hz is the only common-grid rate at which all five
  channels are real measurements. At 4 Hz, HR is upsampled ×4; at 8 Hz and above,
  three of five channels are fabricated. This does not contradict v2's three-layer
  model — v2 is right that rate-sensitive features must be extracted natively and
  then aggregated — it constrains the *common grid* layer only.
- **Missingness is not informative.** A classifier on missingness indicators alone
  reaches LOSO AUC **0.468**, median per-subject 0.50. HRV availability falls with
  motion (10.7% in the lowest accelerometer decile to 3.6% in the highest) but does
  not act as a label proxy. One less confound than feared.
- **Exclusion order does not commute.** Window-level filters do (36,535 surviving
  either way). At session level, measuring length *before* removing non-wear keeps
  517 sessions; *after* keeps 504. **Thirteen sessions change status on rule order
  alone.** Declare the order and perturb it.
- **The three sample sizes.** 37,252 windows / **178 events** / **15 subjects**
  (10 after the recommended exclusions). 2,483 rows per person — any interval
  computed from the row count is wrong by more than an order of magnitude.

---

## 4. Problems in the source documents

1. **`AUDIT.md` A1's stop condition is mis-specified.** It requires zero `t0`
   spread across signals. Observed spread is exactly 10.000 s in all 609 sessions
   with zero variance, entirely the documented HR warm-up. ACC/BVP/EDA/TEMP share
   an identical `t0` everywhere. Amend to exempt HR.
2. **`AUDIT.md` A3 expects 34 events over 60 min from the rated set.** Rated-only
   is 19; 34 counts unrated rows. This *weakens* the published event-duration
   objection — 7.8% of rated events, not 9.5%.
3. **v2 §6.4 threat 8 repeats the same 34.**
4. **v2's Phase 0 propensity AUC of 0.728 appears to be in-sample.**
5. **v2's signal-quality table is measured on 45 stratified sessions** and reads
   optimistic against the full archive wherever the two can be compared.

---

## 5. Open items, in priority order

1. **The causal-z denominator floor.** Not a preference — a numerical failure. The
   trailing IQR approaches zero on flat stretches and produces z-scores above 300.
   It already inverted the headline of §3.1 once. Every feature built on `causal_z`
   inherits it. **Fix before any modelling.** Needs a minimum-IQR floor or a clip,
   and the choice is itself a judgment call that needs recording.
2. **Robust vs mean aggregation, fixed a priori.** §3.1 gives opposite headlines
   from identical data depending on this. Commit to the median and sign test.
3. **`judgment_calls.yaml` does not exist.** `PLAN.md` references `JC02`–`JC28`
   throughout and the plan calls it the preregistration, to be git-tagged before
   Phase 3. Six further calls surfaced during the audit that the 29-entry registry
   does not cover: analysis sample rate; HR `t0` offset handling; session-length
   rule position; event-coverage tolerance for the 33 uncovered events; the
   causal-z floor; and robust-vs-mean aggregation.
4. **`CLAUDE.md` and `nurse_stress_analysis_plan.md` do not exist in the repo**
   despite `PLAN.md` referencing both as authoritative.
5. **v2's open questions 1, 2, 4, 5, 7 remain open** — ICC on the log scale, the
   `eda_med`/`eda_skew` coupling, E4's 1 °C temperature range, Malik truncation of
   real RMSSD, and whether `tags.csv` corroborates the timezone conclusion.

---

## 6. What this changes for the plan

- **Phase 0** gains the event-triggered average as its first row. EDA responds at
  onset; HR does not.
- **HRV leaves the baseline feature set** and becomes an ablation at
  `min_run=30` **beats** (~22 s), RMSSD only, no frequency-domain measures, and
  restricted to the seven subjects with real coverage. Do not tighten to
  `min_run=120`: that is 0.87% coverage and zero usable subjects.
- **Drop 6D plus DF, 7E, CE and EG** — ten subjects, 158 events. The four have
  flat EDA and EDA is the only responsive channel. **Unresolved:** DF and CE are
  also two of the five best HRV subjects, so the EDA and HRV criteria disagree
  about the same people. Decide explicitly rather than by rule order.
- **`RATIO` = 2.14**, the minimum achievable on the ten-subject cohort, applied
  uniformly rather than letting balance vary seventeenfold across folds.
- **1 Hz for the common grid**, native rates for rate-sensitive features, 120 s
  windows. All three now have evidence behind them rather than convention.
- **Soften the SCAR justification** to the subject-specific component (0.63), not
  the pooled 0.728.
- **Weight the feature set toward EDA.** HR contributed nothing measurable at
  onset.
