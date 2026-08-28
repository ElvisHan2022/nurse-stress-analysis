# Audit summary

Full run of `AUDIT.md` sections A1–A10 against the Dryad archive
(`Eric/Stress_dataset`, 609 sessions, 3.55 GB) and `Eric/SurveyResults.xlsx`.
Scripts in `tasks/audit_a*.py`, per-section detail in `findings.md`, figures in
`figures/audit/`.

Two additions beyond `AUDIT.md`: a **sample-rate feasibility analysis** (A1b) and
a **person-day census** (A2b). One addition beyond the plan: **A10b**, a
robustness re-run of the event-triggered average, which changed the headline.

---

## 1. Expected against observed

| Quantity | Expected | Observed | Match |
|---|---|---|---|
| Subjects | 15 | 15 | yes |
| Session directories | 609 | 609 | yes |
| Sensor hours | 1,255 | 1,251.7 | yes (0.3%) |
| Median session length | 1.17 h | 1.167 h | yes |
| Sessions < 5 min | 92 | 92 | yes |
| `tags.csv` empty | 92% | 92.4% | yes |
| Survey rows | 358 | 358 | yes |
| Exact duplicate rows | 3 | 3 | yes |
| Unrated events | 113 | 113 | yes |
| Level 0 / 1 / 2 | 46 / 20 / 179 | 46 / 20 / 179 | yes |
| Max event duration | 323 min | 323 min | yes |
| Overlapping event pairs | 12 | 12 | yes |
| Events > 60 min | 34 | **19 rated** (34 all events) | **no — see below** |
| Level-2 events | 178 | 178 | yes |
| Subjects below 3:1 negatives | 5 of 14 | 5 of 15 | yes |
| ICC(subject), temperature | 0.522 | 0.522 (notebook) | yes |
| Propensity AUC | 0.728 | **0.707 CV / 0.751 in-sample** | close |
| P(labelled) spread | 5.8× | **5.36×** | close |

**The one real mismatch.** `AUDIT.md` A3 asks for the over-60-minute count on
*rated* events and expects 34. Only **19 of 245 rated** events exceed 60 minutes;
34 is the count across all 358 rows including unrated. This weakens the published
event-duration objection: it applies to 7.8% of the rated set, not 9.5%.

**On the SCAR numbers.** `PLAN.md` quoted these as "derived" with nothing in the
repository computing them. A7 now does. The conclusion holds — SCAR is violated —
but the exact figures are slightly softer, and an ablation shows **session
duration (AUC 0.719) carries more of the propensity signal than subject identity
(AUC 0.630)**. Part of what reads as subject-driven reporting bias is the
mechanical fact that longer recordings contain more events. The subject component
is real but smaller than 5.8× implies.

---

## 2. Stop conditions

| Section | Triggered | Resolution |
|---|---|---|
| A1 | **yes** | Spurious. See below. |
| A2 | no | — |
| A4 | no | — |
| A10 | no | EDA responds at onset |

**A1's stop condition is mis-specified and should be amended.** It requires zero
`t0` spread across signals within a session. Observed spread is **exactly 10.000 s
in all 609 sessions, with zero variance, and entirely attributable to `HR.csv`** —
the E4's documented algorithm warm-up. ACC, BVP, EDA and TEMP share an identical
`t0` in every session. This is a device characteristic, not a broken extraction.
Amend the rule to *"nonzero `t0` spread among ACC/BVP/EDA/TEMP, or an HR offset
other than 10 s."*

---

## 3. The three sample sizes

| Unit | Count | What it is the honest denominator for |
|---|---|---|
| 120 s windows | 37,252 | nothing — a compute statistic |
| Level-2 events | **178** | "can we detect an episode" |
| Subjects | **15** (14 usable) | "does this work on a new nurse" |

Rows-to-subject ratio is **2,483**. The judgment-calls guide flags anything above
~100 as a case where standard errors computed from the row count are wrong by more
than an order of magnitude.

Person-days: **250** with any sensor data, from 8 (subject 6D) to 26 (CE), median
5.0 sensor-hours per person-day.

---

## 4. Timezone

**Resolved: survey timestamps are naive `America/Chicago`, DST-aware.** Sensors
are UTC unix epochs.

| Hypothesis | Rated events starting inside a same-subject session |
|---|---|
| Survey is UTC | 50 / 245 (20.4%) |
| Survey naive Chicago → UTC | **212 / 245 (86.5%)** |

Best fixed offset is UTC−5 at 198; the DST-aware named zone beats it by 14 events,
because the archive spans April to December. **Use the named zone, not an offset.**

33 rated events (13.5%) fall outside every session even under the winning
hypothesis — they have no sensor coverage at all. Four of those sit within one
minute of a session boundary and could be recovered with a small tolerance; the
rest are a median of 6.6 hours away and are genuinely uncovered.

Written to `reports/audit/timezone_resolution.md`.

---

## 5. Sample rates and analysis-rate feasibility

Native rates, read from row 2 of each file: **ACC 32 Hz, BVP 64 Hz, EDA 4 Hz,
HR 1 Hz, TEMP 4 Hz** — one distinct rate per signal across all 609 sessions.
Published descriptions giving BVP 72 Hz and TEMP 10 Hz are wrong for this
extraction.

| Target | ACC | BVP | EDA | HR | TEMP | Channels backed by measurement |
|---|---|---|---|---|---|---|
| **1 Hz** | down | down | down | native | down | **5 of 5** |
| **4 Hz** | down | down | native | ×4 up | native | 4 of 5 |
| 8 Hz | down | down | ×2 up | ×8 up | ×2 up | 2 of 5 |
| 16 Hz | down | down | ×4 up | ×16 up | ×4 up | 2 of 5 |
| 32 Hz | native | down | ×8 up | ×32 up | ×8 up | 2 of 5 |

**Recommendation: 1 Hz.** It is the only rate at which every channel is backed by
real measurement. 4 Hz is defensible if HR is explicitly documented as
step-interpolated. At 8 Hz and above, three of five channels are fabricated —
within-window variance in EDA, HR and TEMP would be an artifact of the fill, which
is precisely the flaw that made the merged Kaggle CSV unusable. Anything above
4 Hz is only honest for an ACC/BVP-only model, and costs 8–32× the storage.

---

## 6. HRV feasibility

Time convention verified: `t[i] − t[i−1] == ibi[i]` within 50 ms for **88.3%** of
812,794 consecutive pairs across 482 sessions. High enough to confirm the
convention, so violations mean dropped beats rather than an indexing error.

| Level | Coverage |
|---|---|
| Sessions with ≥1 usable run | 294 / 609 (48.3%) |
| **120 s windows with a usable run** | **2,214 / 37,252 = 5.94%** |

**Recommendation: HRV does not enter the baseline feature set.** Demote to an
ablation. The session-level figure is eight times more flattering than the
window-level one, exactly as `AUDIT.md` predicted. Per-subject availability ranges
from 0.06% (EG) to 15.4% (BG).

**A6's informative-missingness test does not fire.** A classifier using only
missingness indicators and run length reaches **LOSO AUC 0.468** — below chance,
median per-subject 0.50. HRV availability does fall with motion (10.7% in the
lowest accelerometer decile to 3.6% in the highest), but it does not differ by
class enough to act as a label proxy. One less confound than feared.

---

## 7. Negative pool and the ratio setting

Eligibility: session ≥ 30 min, on a subject-day carrying a reported event, ≥ 30 min
from any event boundary rated or not.

1,242 sensor hours → **354 hours of eligible negatives** (10,635 windows against
2,203 positive).

| | |
|---|---|
| Aggregate ratio | **4.83 : 1** |
| Subjects below 3:1 | **5 of 15** — 6D, EG, 8B, 6B, CE |
| Subjects below 1:1 | 1 — **6D, which has zero eligible negatives** |

**Recommended setting: `RATIO = 1.87`** — the minimum achievable across subjects
retaining any negatives — applied uniformly, after dropping 6D. Taking whatever
each subject can supply, silently, would let class balance vary from 1.87:1 to
32:1 across folds. Since balance moves the decision threshold and threshold
transfer is already compromised by the A7 propensity spread, the two failures
compound.

This confirms `PLAN.md` amendment 2.1b in direction. The per-subject ratios differ
from its table (it has CE at 0.9 and E4 at 2.3; observed here 2.45 and 3.73), but
the qualitative claim — the aggregate conceals several subjects that cannot reach
3:1 — holds, and the identity of four of the five is the same.

---

## 8. Does anything happen at event onset?

**Yes for EDA. No for HR.** This is the result that decides the project is well
posed, and it required a robustness pass to state honestly.

The pooled mean (A10) showed an EDA rise of **+9.75 causal-z units**, peaking at
+25. That is not a plausible z-score. The causal-z denominator is a trailing IQR,
so any stretch of near-constant EDA drives it toward zero and the ratio toward
infinity. **Five events contribute 63% of the total shift**, the largest being
+318. The mean is a statement about those five events, not the population.

Outlier-resistant re-run (A10b), 140 level-2 events with usable coverage:

| Channel | Mean Δz | **Median Δz** | Events positive | Sign test | Median raw shift |
|---|---|---|---|---|---|
| **EDA** | +9.75 | **+0.568** | **90/140 = 64.3%** | **p = 0.0009** | **+0.64 µS** |
| HR | +0.006 | −0.067 | 68/140 = 48.6% | p = 0.80 | +0.47 bpm |

EDA survives every robust test — sign test p = 0.0009, Wilcoxon p = 3.3×10⁻⁶ — and
the raw median shift of +0.64 µS sits on a baseline of 0.68 µS, roughly a doubling
of skin conductance. **HR shows nothing**: 48.6% of events move in the positive
direction, indistinguishable from a coin flip.

The labels mark a physiologically distinguishable state, and they mark it in EDA.
This is consistent with the original paper's SHAP finding that EDA dominates, and
it is now established before any model was fitted.

---

## 9. Rule order

Window-level exclusions commute exactly — 36,535 surviving windows under both the
`PLAN.md` order and the swapped order, difference zero.

**Session-level exclusions do not.** Computing session length before removing
non-wear keeps **517** sessions; removing non-wear first keeps **504**. Thirteen
sessions change status. Both are defensible; only one is what you did, and the
write-up must say which.

Survey-side attrition: 358 rows → 355 (drop duplicates) → 242 (drop unrated) →
178 (levels 0 and 1 excluded) → **176** (drop 6D).

---

## 10. Judgment calls needing registry entries

Encountered during the audit and not covered by the existing 29:

1. **Analysis sample rate.** Not in the registry at all. Options 1 / 4 Hz, with
   8+ Hz ruled out by A1b. Default 1 Hz. This is upstream of `JC14` (channel set)
   and should precede it.
2. **HR `t0` offset handling.** HR starts exactly 10 s after its siblings in every
   session. Options: shift, truncate all channels to the common span, or carry the
   offset into the resample index. Currently implicit.
3. **Session-length rule position.** A9 shows this does not commute with non-wear
   removal. Needs an explicit order declaration, and perturbing.
4. **Event-coverage tolerance.** 33 rated events fall outside every session; 4 are
   within one minute of a boundary. A tolerance parameter decides whether they are
   recovered. Currently zero by omission rather than by choice.
5. **Causal-z denominator floor.** The A10 artifact is a live failure mode. Any
   feature built on `causal_z` needs a minimum-IQR floor or a clip, and the choice
   changes results.
6. **Robust vs mean aggregation for event statistics.** A10 and A10b give opposite
   headlines from identical data. Fix this a priori.

Item 5 is the one to act on first: it is not a reporting preference, it is a
numerical failure that will propagate silently into every feature.

---

## What this changes for the plan

- **HRV out of the baseline.** 5.94% window coverage. Ablation only.
- **1 Hz confirmed as the analysis rate**, on measurement grounds rather than
  convenience.
- **Drop 6D** — it has zero eligible negatives, which A8 establishes independently
  of its event count.
- **`RATIO = 1.87`**, not 3.
- **The project is well posed, on EDA.** HR contributes nothing at onset, which is
  a reason to weight the feature set toward electrodermal channels and to treat any
  HR-driven result with suspicion.
- **SCAR is violated but less dramatically than stated**, and more of it is session
  duration than subject identity. Subject-stratified nnPU is still the right call;
  the justification should cite AUC 0.63 for the subject component, not 0.728
  overall.
