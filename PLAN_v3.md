# Nurse Stress Detection — Analysis Plan (v3)

Dryad `doi:10.5061/dryad.5hqbzkh6f`. Supersedes v1 and v2.

Merges the exploration notebook, the independent audit (A1–A11), and the HRV
reconciliation. Every number below is measured over all 609 sessions unless marked
otherwise.

**Changes from v2**

- **HRV demoted** from baseline feature to ablation. v2 upgraded it on a session-level
  statistic; the window-level number is 5.94% and the definition-free number is 3.16%.
  v2 was wrong and the audit was right.
- **Event-triggered average added as Phase 0's first row.** EDA responds at onset; HR does not.
  Neither v1 nor v2 contained this check, and it reorders the feature priorities.
- **Cohort cut to 10 subjects, 156 level-2 events.** Four subjects have flat, not merely small, EDA.
- **`RATIO` set to 2.14**, from the audit's Phase 2 budget on the ten-subject cohort.
- **SCAR justification softened** to the subject-specific component (AUC 0.63), not the pooled figure.
- **One blocking bug**: the `causal_z` denominator floor. Fix before any modelling.

---

# PART 1 — HIGH-LEVEL PLAN

**The task.** Detect high-stress periods (survey level 2) from wrist physiology, evaluated
leave-one-subject-out at the event level.

**The two problems, which are independent.**

| Problem | Size | Addressed by |
|---|---|---|
| No negative class exists in the labels | 92% of sensor time is unlabelled | Phase 2 (curated negatives) + Phase 3 (PU) |
| Very few positives | 156 level-2 events across 10 subjects | Phases 4–5 (pretrain, then fine-tune) |

Self-supervised pretraining does **not** manufacture negatives. Solve the negative problem
first; a well-pretrained model fitted against a biased target is still biased.

**Six phases.**

0. **Fix the causal-z floor, then confirm the labels mark something physiological.** Both are
   prerequisites, not preliminaries.
1. **Define units and exclusions.** Window, label rule, cohort. Produce the effective-N table.
2. **Construct negatives by matching, not filtering.** Activity-matched within subject.
   Falsify with an ACC-only model.
3. **Baseline + PU.** Gradient boosting on hand-crafted features first, weighted toward EDA.
   Then nnPU, subject-stratified, with a sensitivity grid over π.
4. **Self-supervised pretraining.** Masked reconstruction over all 1,251.7 h, inside each fold.
5. **Fine-tuning and model comparison** against a pre-committed adoption rule.
6. **Severity, evaluation, reporting.**

**The four checks that decide whether any of it is real.**

- **Event-triggered average** — do the labels mark a physiological change at all? *(EDA yes, HR no.)*
- **ACC-only ablation** — is it an activity detector?
- **Subject-ID probe on the embedding** — did the encoder learn identity?
- **Naive vs. curated negatives, side by side** — the gap is a result either way.

---

# PART 2 — DETAILED PLAN

## Phase 0 — Prerequisites and established facts

### 0.1 BLOCKING: the causal-z denominator floor

`causal_z` divides by a trailing IQR. `iqr.replace(0, np.nan)` catches exact zeros but not
near-zeros, so a flat stretch with IQR ≈ 0.001 yields z-scores above 300. Five events out of
140 contributed 63% of the pooled mean in the event-triggered analysis, the largest at +318 z.
It already inverted one headline.

Every feature built on `causal_z` inherits this. **Fix first.** Needs a minimum-IQR floor or a
clip, and that choice is itself a judgment call to record.

Alongside it, fix a second choice *a priori*: **median and sign test, never the pooled mean.**
The same data gives opposite headlines depending on this.

### 0.2 The event-triggered average — the result that frames everything

140 level-2 events with usable coverage, aligned at onset, 0–10 min post-onset against the
preceding 30 min:

| Channel | Median Δz | Events positive | Sign test | Median raw shift |
|---|---|---|---|---|
| **EDA** | **+0.568** | **90/140 = 64.3%** | **p = 0.0009** | **+0.64 µS** on a 0.68 µS baseline |
| HR | −0.067 | 68/140 = 48.6% | p = 0.80 | +0.47 bpm |

EDA survives every robust test (Wilcoxon p = 3.3×10⁻⁶) — roughly a doubling of skin
conductance. **HR is a coin flip.**

**Consequences that propagate through the whole plan:** the problem is well posed, and well
posed *in EDA*. Weight the feature set accordingly. Treat any result driven by heart rate as
suspect until explained. And a subject whose EDA is flat cannot contribute signal — which is
what settles §1.5.

### 0.3 Structure and labels

| Fact | Value |
|---|---|
| Subjects / sessions | 15 / 609 |
| Sensor time | 1,251.7 h |
| Person-days | 250 (8 for 6D to 26 for CE) |
| Median session length | 1.17 h; 92 sessions < 5 min |
| Survey events | 358 rows — 245 rated, 113 unrated |
| Stress levels | 46 low / 20 medium / 179 high |
| Level-2 events after dedup | 178 (156 after the cohort cut) |
| Rated events with **no sensor coverage** | 33 (13.5%); four within one minute of a session edge |
| Exact duplicate survey rows / overlapping event pairs | 3 / 12 |
| Rated events over 60 min | **19** (not 34 — that figure counts unrated rows) |
| Timezone | survey `America/Chicago` DST-aware, sensors UTC |
| The three sample sizes | **37,252 windows / 156 events / 10 subjects** |

**Timezone evidence:** survey-as-UTC places 50/245 rated events (20.4%) inside a same-subject
session; naive Chicago places **212 (86.5%)**. Best fixed offset is UTC−5 at 198; the DST-aware
named zone beats it by 14 because the archive spans April to December. **Use the named zone.**

### 0.4 Sampling rates

| Layer | Rate | Job |
|---|---|---|
| Native | EDA 4, TEMP 4, HR 1, ACC 32, BVP 64 Hz; IBI irregular | rate-sensitive feature extraction |
| Common grid | **1 Hz** | label join, non-wear masking, plotting, coarse features |
| Window | 120 s / 60 s hop | the modelling substrate |

1 Hz is the only common-grid rate at which all five channels are real measurements — at 4 Hz HR
is upsampled ×4; at 8 Hz and above, three of five channels are fabricated. This constrains the
common-grid layer only; rate-sensitive features are still extracted natively and then aggregated.

Measured cost of getting that wrong: SCR detection on 1 Hz EDA finds **110 vs 184** SCRs on
`5C_1587297777` and **829 vs 1,400** on `7A_1587643240` — a consistent ~40% loss.

### 0.5 HRV — demoted, and why

The reconciliation is tight. Three independent routes to usable beat data agree within 0.2%:
`a5_hrv_runs.usable_s` 39.53 h, `a5b_usable_totals` 39.53 h, `mean_covered_s × n_windows`
39.46 h. Window coverage matches exactly between A5 and A5b (2,214/37,252 = 5.94%).

| Measure | Value |
|---|---|
| Usable beat data, definition-free | **39.5 h of 1,251.7 h = 3.16%** |
| 120 s windows with a usable run (Definition A) | 5.94% |
| 120 s windows with ≥30 s summed across runs (Definition B) | **6.01%** — the single-run requirement costs 24 windows out of 37,252 |
| At the lenient 20s/15b rule | 57.3 h (4.58%); 7.04% of windows |
| At the strict 60s/40b rule | 16.1 h (1.29%); 1.98% of windows |

**Window length gives no HRV argument either way.** Mean covered fraction is flat at 3.17–3.18%
at 60, 90, 120, 180 and 300 s. Definition A appears to improve with window length and Definition
B appears to peak at 120 s, but both are threshold-scaling artifacts. Window choice
redistributes usable beat data; it cannot create more. Keep 120 s on the grounds in §1.1.

**The correct framing is that HRV is concentrated, not absent.** At the recommended
`min_run=30 s`, 7 of 15 subjects have ≥100 covered windows (DF 401, 83 348, BG 345, CE 262,
5C 257, E4 142, F5 142). The "zero of fifteen subjects" claim in the audit summary holds only at
`min_run=100 s`, a threshold the audit itself rejects.

**Decision: HRV is an ablation, not a baseline feature.** RMSSD only (it outperforms SDNN at
short durations in every ultra-short study), no frequency-domain measures, `min_run=30 s / 20
beats`, with an `hrv_available` indicator and `valid_frac` covariate. Never impute.

### 0.6 Two measurement caveats from our own tooling

1. **The Malik criterion biases RMSSD low at the high end.** Simulated: true 20 ms → 19.9,
   40 ms → 40.2, **80 ms → 68.7 (−14%)**. Because RMSSD falls under stress, this compresses the
   relaxed end toward the stressed end. Use `malik_gate=0.20` to decide usability and
   `malik_metric=0.30` to compute the reported value.
2. **`plausible_runs` is redundant** — it sits downstream of Malik and rejects almost nothing.
   Keep as an assertion that should stay near zero; `valid_frac` is the real quality metric.

### 0.7 SCAR — softened but intact

Propensity to be labelled is predictable from observables: **AUC 0.707 cross-validated**
(0.751 in-sample; v2's 0.728 was the in-sample figure). Per-subject P(labelled) spans 5.36×.

**But the decomposition matters:** session **duration** predicts labelling better (AUC 0.719)
than subject identity (**AUC 0.630**). Much of the apparent SCAR violation is the mechanical
fact that longer recordings contain more events. Subject-stratified nnPU remains the right call;
justify it with 0.63 for the subject component, not the pooled number.

### 0.8 One confound that isn't

A classifier on missingness indicators alone reaches **LOSO AUC 0.468** (median per-subject
0.50). HRV availability falls with motion (10.7% in the lowest accelerometer decile to 3.6% in
the highest) but does not act as a label proxy. One less thing to control for.

---

## Phase 1 — Units, labels, cohort

### 1.1 Windowing — settled

**120 s windows, 60 s hop.** Two independent arguments:

- No labelled event is shorter than 120 s; 300 s would discard 15 events outright, and only
  6 of 179 level-2 events fall under 4 minutes.
- The ultra-short HRV literature finds 120 s sufficient for RMSSD and SDNN (Muñoz 2015; Orini
  2023 validates RMSSD from ≤15 s ECG). The classical 300 s figure is a recording convention.

HRV coverage gives **no** argument either way (§0.5).

Windows lie entirely within one session. Overlapping windows are near-duplicates: fine for
training, never split across train/test.

### 1.2 Label rule

Positive if ≥50% of the window falls inside a level-2 event. Negative only if it survives
Phase 2. Everything else **excluded**, not zero.

### 1.3 Exclusions — declare the order, then perturb it

**Exclusion order does not commute.** Window-level filters do (36,535 survive either way). At
session level, measuring length *before* removing non-wear keeps 517 sessions; *after* keeps 504.
**Thirteen sessions change status on rule order alone.** Fix the order below, record it as a
judgment call, and run the reverse order as a robustness check.

| # | Rule |
|---|---|
| 1 | Drop sessions < 5 min (measured **before** non-wear removal) |
| 2 | Drop non-wear seconds |
| 3 | Drop sessions with `eda_med` < 0.05 µS or `eda_floor_pct` > 50 |
| 4 | Drop the 113 unrated (`na`) events from *both* classes |
| 5 | Drop the 3 exact-duplicate survey rows |
| 6 | Resolve the 12 overlapping event pairs — keep the outer span or drop the pair; decide once |
| 7 | Exclude level-0 and level-1 events from the binary task |
| 8 | Drop subject **6D** — 2 events, 0 eligible negatives |
| 9 | Drop **7E, DF, EG, CE** — flat EDA (§1.5) |
| 10 | Set aside the **33 rated events with no sensor coverage**; record the tolerance used for the four within one minute of a session edge |

**On rule 7:** do not fold level 0 into the negative class. Those are windows a nurse actively
flagged — the cleanest hard negatives available. Excluding them is not the same as calling them
negative. They return in Phase 6.

### 1.4 Effective-N table — the ten-subject cohort

| Subject | Level-2 events | Eligible neg. hours | HRV-covered windows |
|---|---|---|---|
| 5C | 4 | 9.02 | 257 |
| 94 | 5 | 15.40 | 67 |
| 6B | 11 | 12.34 | 30 |
| 8B | 13 | 5.39 | 38 |
| BG | 14 | 25.48 | 345 |
| 15 | 16 | 15.08 | 15 |
| 83 | 16 | 44.54 | 348 |
| F5 | 24 | 37.11 | 142 |
| 7A | 25 | 35.55 | 81 |
| E4 | 28 | 31.12 | 142 |
| **Total** | **156** | **231.0** (audit: 265.2 — see open items) | **1,465** |

**Two subjects have ≤5 events** (5C, 94). Report those folds separately; do not average them in.

**Arithmetic note.** 178 − 20 (the four flat-EDA subjects) − 2 (6D) = **156**. The audit summary
states 158, which subtracts the four but not 6D. Use 156.

### 1.5 The cohort cut — settled by the IQR/median criterion

A low median alone is not disqualifying; a small but *varying* signal still carries information
after causal normalisation. Relative variation separates the two cases cleanly:

| Subject | Median EDA | **IQR / median** |
|---|---|---|
| 7E | 0.074 | **0.60** |
| DF | 0.092 | **0.44** |
| EG | 0.099 | **0.72** |
| CE | 0.106 | **0.48** |
| *(all eleven others)* | 0.18 – 1.08 | **1.04 – 5.14** |

These four are the **only** subjects below 1.0, with no overlap with the rest of the cohort.
Their EDA is not merely small, it is flat. Combined with §0.2 — EDA is the only channel that
responds at onset — a subject with flat EDA cannot contribute signal.

**Cost:** 20 of 178 events (11%). **Benefit:** the negative budget improves (minimum achievable
ratio 1.87 → 2.14, subjects below 3:1 halve from 4 to 2).

**The interaction nobody planned for.** DF and CE are also the first and fourth highest for HRV
coverage. The four flat-EDA subjects hold **717 of 2,214 HRV-covered windows (32%)**. After the
cut, 1,465 windows remain across 10 subjects, with 5 of them above 100 windows (5C, 83, BG, E4,
F5). Still enough for an ablation — but the two recommendations were reached independently and
overlap on a third of the HRV data. State this rather than discovering it later.

**EG is unambiguous:** flat EDA *and* 2 HRV-covered windows out of 3,223 (0.06%). Both channels
dead. Drop regardless of how the other three resolve.

---

## Phase 2 — Negative construction

### 2.1 Eligibility

All of:

1. Session ≥30 min.
2. **On a subject-day containing at least one reported event** — the key rule. Restricts
   negatives to days when the nurse was demonstrably completing surveys, so absence of a report
   carries information. 250 person-days total, so this is a meaningful restriction.
3. ≥30 min from the boundary of *any* event, including the 113 unrated ones.
4. Worn, in a session that passed the dead-EDA check.

### 2.2 Activity matching — the part that matters

Do not filter negatives to low-motion periods. If positives contain motion and negatives don't,
the model learns to detect movement.

```python
POS  = windows[windows.label == 1]
CAND = windows[windows.eligible_negative]

edges = POS.acc_mag_mean.quantile(np.linspace(0, 1, 11)).values
edges[0], edges[-1] = -np.inf, np.inf
POS['abin']  = pd.cut(POS.acc_mag_mean,  edges, labels=False)
CAND['abin'] = pd.cut(CAND.acc_mag_mean, edges, labels=False)

RATIO = 2.14                        # audit A11, uniform across the ten-subject cohort
neg = []
for sub, gp in POS.groupby('subject'):          # match WITHIN subject
    want = gp.abin.value_counts() * RATIO
    pool = CAND[CAND.subject == sub]
    for b, n in want.items():
        avail = pool[pool.abin == b]
        neg.append(avail.sample(min(int(round(n)), len(avail)), random_state=0))
NEG = pd.concat(neg)
print(NEG.groupby('subject').size())
```

**`RATIO` = 2.14** is the minimum achievable on the ten-subject cohort. Applying it uniformly
matters: letting each subject take its maximum lets class balance vary seventeenfold across
folds, which makes per-fold metrics incomparable.

Match **within subject**. Global matching draws negatives disproportionately from the
high-coverage subjects and turns subject identity into a label proxy.

Repeat under ≥3 seeds; report variance.

### 2.3 Falsification checks (before any modelling)

| Check | Pass condition | If it fails |
|---|---|---|
| **ACC-only model** | well below full model | re-stratify; add `acc_sd`, `acc_p2p` to matching |
| **Time-of-day-only model** | near chance | add hour to the matching |
| Subject-ID from features | — | record as the ceiling Phase 4's probe must beat |
| Negative count per subject | ≥100 windows | drop or flag the subject |

Note that the missingness-as-label-proxy check has already passed (§0.8, AUC 0.468).

### 2.4 Deliverable

A frozen, versioned table: `subject, session, window_start, label ∈ {0,1}, abin, seed`.

---

## Phase 3 — Baseline, then PU

### 3.1 Feature set — weighted toward EDA

**EDA (primary).** From the native 4 Hz file via `eda_features_native()`, never the 1 Hz cache.
Tonic mean and slope, phasic SD, SCR count, SCR mean amplitude. `log1p` the level-like features.
This is the channel that responds at onset (§0.2); everything else is supporting.

**ACC.** Mean/SD/p2p of magnitude, fraction of seconds above a movement threshold. Needed as an
exertion covariate whether or not it helps directly.

**HR (demoted).** Mean, SD, slope, max; delta vs. trailing 30-min median. Include, but §0.2
found no measurable onset response — treat any HR-driven result as suspect until explained.

**TEMP.** Slope and delta-from-rolling-median only. **Never absolute temperature** (ICC 0.52–0.62
makes it a subject fingerprint).

**HRV — ablation only, not in the baseline.** RMSSD at `min_run=30 s / 20 beats`,
`malik_metric=0.30`, with `hrv_available` and `valid_frac`. Covers ~6% of windows. Never impute.

**Context.** Hour of day, minutes into session.

### 3.2 Normalisation

`causal_z` within session, **after the §0.1 floor fix**. Settle the ICC-on-log question (open
item 1) before finalising: if ICC on `log1p(eda)` is high, add per-subject centring on the log
scale; if it stays low, the causal rolling baseline alone is sufficient.

### 3.3 Baseline model

Gradient boosting, balanced class weights, LOSO. This is the number everything else must beat.
There is a real chance it wins outright.

### 3.4 nnPU, subject-stratified

- Positives = level-2 windows. Unlabelled = the *full* eligible pool, not the matched subset.
- Fit subject-stratified or with a per-subject labelling-propensity offset (§0.7).
- **π: bracket, don't estimate.** Fit at π ∈ {0.05, 0.10, 0.20, 0.30}; report how conclusions move.
- Non-negative risk correction (Kiryo et al. 2017), not plain uPU.

### 3.5 Naive comparator

Same model, all unlabelled time as 0. Report **beside** the curated result. Do not use it to seed
or select curated negatives.

---

## Phase 4 — Self-supervised pretraining (optional)

Attempt only if Phase 3 clears a floor worth improving on.

**Data.** All 1,251.7 h, worn seconds only, causal-z normalised, absolute temperature dropped.
Channels `eda, hr, acc_mag, acc_sd, temp_delta` at 1 Hz. Include the excluded subjects and
sessions — pretraining has no label requirement.

**Pretext: masked reconstruction.** Mask random 30–120 s spans; reconstruct with MSE. Chosen over
contrastive learning deliberately: standard augmentations include amplitude scaling, and **EDA
amplitude is the signal**.

**Architecture.** 1D dilated CNN (4–6 blocks) or a small transformer over 120 s windows. ~4.5M
timesteps is modest; keep it small.

**Subject-shortcut probe (mandatory gate).** Freeze the encoder, train a linear classifier to
predict subject ID. Near chance → proceed. High → gradient-reversal on a subject head, stronger
per-session normalisation, or within-session-only reconstruction targets. Re-probe.

**Leakage trap.** Pretraining on all subjects then evaluating LOSO is **transductive**. Either
re-run pretraining inside each fold excluding the test subject, or pretrain once and label the
result transductive. Pick one, state it.

---

## Phase 5 — Fine-tuning and model comparison

One question: *does a pretrained representation beat hand-crafted features on 156 events?*
Answer it cleanly or not at all.

### 5.1 Two variants, both required

| Variant | What | Why |
|---|---|---|
| **Linear probe** | freeze the encoder, fit a logistic head on the embeddings | tests representation quality directly; likely the winner at this data volume |
| **Full fine-tune** | unfreeze with discriminative learning rates, encoder at ~1/10 the head's | tests whether task adaptation helps or overfits |

If full fine-tuning loses to the linear probe, that is a data-volume verdict, not a bug.

### 5.2 Nested validation — the trap specific to this phase

Early stopping needs a validation set, and **a random split of training windows is invalid** —
adjacent windows are near-duplicates, so an inner random split leaks and stopping fires at the
wrong epoch. Use a nested **group** split:

```python
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit

outer = LeaveOneGroupOut()
for tr_idx, te_idx in outer.split(X, y, groups=subject):
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    fit_idx, val_idx = next(inner.split(X[tr_idx], y[tr_idx], groups=subject[tr_idx]))
    # fit on tr_idx[fit_idx], early-stop on tr_idx[val_idx], score on te_idx
```

With 10 subjects an inner split costs 2 per fold. Expensive but unavoidable; the alternative is
a leaked stopping criterion.

### 5.3 Class imbalance

Class weights and threshold tuning. **No SMOTE** — interpolating between adjacent autocorrelated
windows manufactures copies of real samples, and applied before the split it leaks outright.

### 5.4 Comparison protocol

All models on **identical folds and identical windows**: Phase 3 gradient boosting, Phase 3 nnPU,
linear probe, full fine-tune, ACC-only.

Compare **per fold**, not by mean. With 10 subjects, a paired Wilcoxon signed-rank test across
folds is the appropriate comparison and is still underpowered — a difference smaller than the
between-fold spread is not a difference.

### 5.5 Pre-commit to an adoption rule

> Adopt the pretrained model only if it beats the Phase 3 baseline on event-level recall at the
> fixed false-alarm rate in at least *k* of *n* folds, with no fold degrading by more than *X*.

Without this, four variants across ten folds will always yield a configuration that "wins."

### 5.6 Expected failure modes

| Symptom | Reading |
|---|---|
| Full FT ≪ linear probe | too little labelled data to adapt the encoder |
| Both ≪ baseline | the pretext task didn't capture stress-relevant structure |
| Both ≫ baseline, but subject probe is high | you are measuring subject identity — return to Phase 4 |
| Large fold-to-fold variance | expected; report per-fold, don't average the low-count subjects |

---

## Phase 6 — Severity, evaluation, reporting

### 6.1 Severity

Detection uses level 2 vs. constructed baseline. The 0/1/2 scale is a separate task on the rated
events, requiring no negatives.

**The scales are not comparable across subjects.** F5 rated 1/0/25; 94 rated 11/4/5. Therefore
ordinal (ordered logit), not 3-class softmax; mixed-effects with a per-subject random intercept,
or within-subject rank normalisation. With 20 level-1 events study-wide and 9 subjects having
none, don't expect the middle class to be learnable — consider collapsing to {0,1} vs {2}.

### 6.2 Evaluation protocol

- **LeaveOneGroupOut on subject.** Report per-fold counts alongside per-fold metrics.
- **Event-level scoring**: detected if ≥*k*% of its windows fire (fix *k* = 50 in advance).
- **Primary metric**: event-level recall at a fixed **false alarms per worn hour**. More
  informative than AUC given a constructed negative class.
- Report the low-count folds (5C, 94) separately.
- **No SMOTE.**

### 6.3 Required ablations

| Ablation | Answers |
|---|---|
| ACC-only | is it an activity detector? |
| Time-of-day-only | is it a shift-schedule detector? |
| Naive vs. curated negatives | how much does negative construction matter? |
| π ∈ {0.05, 0.10, 0.20, 0.30} | how much does the unknowable prior matter? |
| EDA-only / HR-only | does anything but EDA contribute? (§0.2 predicts not) |
| **+HRV on the ~6% of windows that have it** | does HRV add anything where available? |
| With vs. without the four flat-EDA subjects | is the result driven by data quality? |
| Exclusion order reversed | do the 13 boundary sessions matter? |
| Subject-ID probe | is the representation contaminated? |
| ≥3 negative-sampling seeds | how much is sampling noise? |

### 6.4 Threats to validity

1. **SCAR is violated**, though less than v2 claimed — subject component AUC 0.63, duration 0.72.
2. **π is unidentifiable** from this data.
3. **Negatives are constructed, not observed.** Every number is conditional on Phase 2's recipe.
4. **RMSSD is conservative** by ~14% at the high end, from Malik filtering.
5. **Four subjects excluded for flat EDA**; report with and without.
6. **33 rated events (13.5%) have no sensor coverage** and are silently absent from every metric.
7. **Cohort**: 10 of 15 female nurses, one hospital, COVID-era. External validity is narrow.
8. **Documentation discrepancy**: sessions run 2020-04-14 → 2020-12-13; the source paper
   documents only Apr–May and Nov–Dec.
9. **Long events**: 19 rated events exceed 60 min, max 323 min — 7.8% of rated events, which is
   weaker than the published objection implies. Event 21 (94 min) showed sustained EDA elevation
   across its whole span, so these may be genuine.

### 6.5 Preregister

`judgment_calls.yaml` **does not yet exist** despite being referenced throughout as the
preregistration, to be git-tagged before Phase 3. Create it. Beyond the 29 entries already
drafted, six calls surfaced during the audit that it does not cover:

1. analysis sample rate
2. HR `t0` offset handling
3. session-length rule position (§1.3 — 13 sessions turn on it)
4. event-coverage tolerance for the 33 uncovered events
5. **the causal-z floor** (§0.1)
6. **robust vs mean aggregation** (§0.1)

With 156 positives across 10 subjects, undisclosed flexibility will produce whatever result you
go looking for.

---

## Open items

### Blocking

| # | Item | Why it blocks |
|---|---|---|
| 1 | **Causal-z denominator floor** | every downstream feature inherits it; already inverted one headline |
| 2 | **Robust vs mean aggregation, fixed a priori** | same data, opposite headlines |
| 3 | **Create `judgment_calls.yaml`** | it is the preregistration |

### Needed before Phase 2

| # | Question | How |
|---|---|---|
| 4 | **HRV coverage stratified by label** — is coverage inside level-2 windows lower than the 5.94% overall? Coverage falls with motion and events involve motion. If it is ~2%, the ablation is a formality. | needs raw IBI intervals (`export_ibi.py`) |
| 5 | **Corrected A5c rerun.** The "drop beat, continue" variant does not continue — it splits and discards the beat, which is why coverage *fell* 5.94→5.88 when it should have risen. Session counts are identical at all three thresholds. | rerun with `segment_true_continue` |
| 6 | **Eligible negative hours: 231.0 (notebook) vs 265.2 (audit) on the ten-subject cohort**, 293.3 vs 354.5 on all fifteen — a 21% gap from different eligibility implementations | diff the two implementations |
| 7 | ICC(eda) on the **log** scale, `N_PER_SUBJECT=7` | recompute `icc1` |
| 8 | `eda_med` / `eda_skew` coupling (ρ = −0.43, p = 0.11 across subjects) | session-level correlation on the 45-point sample |
| 9 | Overlapping *session* pairs: 7 (notebook) vs 1 (audit) | low stakes, still unresolved |

### Lower priority

| # | Question |
|---|---|
| 10 | E4's temperature range is ~1 °C (33.91–34.87) vs 6D's 7 °C |
| 11 | How much Malik truncates real RMSSD — sweep malik ∈ {0.20, 0.30, 0.50} on 5C, DF, BG |
| 12 | Do the 46 sessions with button presses corroborate the timezone conclusion? |
| 13 | Is the Jun–Aug data a third undocumented collection phase? (ask the authors) |
| 14 | `AUDIT.md` A1's stop condition requires zero `t0` spread; observed spread is exactly 10.000 s in all 609 sessions, entirely the documented HR warm-up. Amend to exempt HR. |

---

## Decision points

| # | Decision | Default | Revisit if |
|---|---|---|---|
| 1 | Binary level-2 vs. 3-class | binary | a different label rule grows the level-1 count |
| 2 | Level 0 → negative? | **no**, exclude | — |
| 3 | Window length | **120 s, settled** | — |
| 4 | Guard band | 30 min | negative pool < 100 windows/subject |
| 5 | Same-day rule | on | pool too small |
| 6 | HRV | **ablation, `min_run=30 s`, RMSSD only** | open item 4 shows label-stratified coverage is adequate |
| 7 | Cohort | **10 subjects** | — |
| 8 | `RATIO` | **2.14, uniform** | — |
| 9 | Pretrain per fold | yes if compute allows | otherwise report transductive |

---

## Sequencing

**Week 1** — Blocking items 1–3, then open items 4–8. Three of these change Phase 1–3 decisions.
**Weeks 2–3** — Phases 1 and 2. Frozen window table, negative pool, falsification checks.
**Week 4** — Phase 3.1–3.3 baseline. LOSO, per-fold counts, ablations.
**Week 5** — Phase 3.4–3.5 nnPU and naive comparator, π grid.
**Weeks 6–8** — Phases 4 and 5, only if warranted.
**Week 9** — Phase 6 severity model and write-up.

Phases 1–3 stand alone. "Curated-negative detection reaches X% event recall at Y false alarms
per hour, and the naive construction inflates this to Z" is a complete, honest, publishable
result whether or not X is impressive.
