# Nurse Stress Detection — Analysis Plan (v2)

Dryad `doi:10.5061/dryad.5hqbzkh6f`. Follows on from the structural exploration notebook.
Every budget figure below is measured from the archive, not assumed.

**Changes from v1:** HRV upgraded from "probably unusable" to viable for most subjects (an
interval-segmentation bug was masking it). Four subjects flagged for chronically near-floor EDA.
Sampling-rate policy made explicit. Window length confirmed empirically. Fine-tuning promoted
from a subsection to **Phase 5**, so the phases renumber from there.

---

# PART 1 — HIGH-LEVEL PLAN

**The task.** Detect high-stress periods (survey level 2) from wrist physiology, evaluated
leave-one-subject-out at the event level.

**The two problems, which are independent.**

| Problem | Size | Addressed by |
|---|---|---|
| No negative class exists in the labels | 87.7% of sensor time is unlabelled | Phase 2 (curated negatives) + Phase 3 (PU) |
| Very few positives | 178 level-2 events across ≤14 usable subjects | Phases 4–5 (pretrain, then fine-tune) |

Self-supervised pretraining does **not** manufacture negatives. Solve the negative problem
first; a well-pretrained model fitted against a biased target is still biased.

**Six phases.**

1. **Define units and exclusions.** Fix the window, the label rule, and what gets dropped.
   Produce an effective-N table before writing any model code.
2. **Construct negatives by matching, not filtering.** Eligible unlabelled time, sampled to
   match the positive class's activity distribution. Falsify with an ACC-only model.
3. **Baseline + PU.** Hand-crafted features and gradient boosting first. Then nnPU with a
   sensitivity grid over the class prior π, fitted subject-stratified because labelling
   propensity varies 5.8× between subjects.
4. **Self-supervised pretraining.** Masked reconstruction over all 1,255 h. Pretraining must
   happen *inside* each LOSO fold or the comparison is transductive.
5. **Fine-tuning and model comparison.** Linear probe and full fine-tune against the Phase 3
   baseline, on identical folds and identical windows, with a pre-committed adoption rule.
6. **Severity, evaluation, reporting.** Ordinal model on events only. Event-level recall
   against false-alarms-per-hour. Report per-fold, report the ablations, report π sensitivity.

**The three checks that decide whether any of it is real.**

- **ACC-only ablation.** If activity features alone approach full-model performance, you
  built an actigraph.
- **Subject-ID probe on the embedding.** If subject is linearly decodable, the encoder learned
  identity, not stress, and LOSO will collapse.
- **Naive vs. curated negatives, reported side by side.** The gap is a result either way.

**Rough effort.** Phases 1–3 are 2–3 weeks and produce a publishable result on their own.
Phases 4–5 are the expensive part and are optional — do them only if Phase 3's baseline clears
a floor worth improving on.

---

# PART 2 — DETAILED PLAN

## Phase 0 — Established facts (carry these forward)

### Structure and labels

| Fact | Value |
|---|---|
| Sessions / sensor time | 609 / 1,255 h |
| Median session length | 1.17 h; 92 sessions < 5 min; 7 overlapping pairs |
| Survey events | 358 rows, 245 labelled, 113 unrated |
| Stress levels | 46 low / 20 medium / 179 high |
| Sensor time inside a labelled event | 99.9 h (8.0%) |
| Timezone | survey = `America/Chicago`, sensors = UTC |
| **SCAR violated** | AUC 0.728 predicting labelling from subject/shift/duration; per-subject P(labelled) spans 0.125–0.724 (**5.8×**) |

### Sampling rates — three layers, not one

| Layer | Rate | Job |
|---|---|---|
| Native | EDA 4 Hz, TEMP 4 Hz, HR 1 Hz, ACC 32 Hz, BVP 64 Hz, IBI irregular | rate-sensitive feature extraction |
| Common grid | 1 Hz | label join, non-wear masking, plotting, coarse features |
| Window | 120 s / 60 s hop | the modelling substrate |

**Extract rate-sensitive features at the native rate, then aggregate to the window.** Measured
cost of getting this wrong: SCR detection on downsampled 1 Hz EDA finds **110 vs 184** SCRs on
`5C_1587297777` and **829 vs 1,400** on `7A_1587643240` — a consistent ~40% loss. ACC is
already handled correctly (within-second SD computed at 32 Hz before averaging).

### Signal quality (measured over 45 stratified sessions)

| Diagnostic | Result | Consequence |
|---|---|---|
| **HRV plausible runs** | 33/45 sessions have ≥1; median RMSSD **34.8 ms**; `valid_frac` median 0.74 | HRV is **viable** — v1 said otherwise, and v1 was wrong |
| HRV at conventional length | `min_run=300` → **1/15** subjects even in their best session; `min_run=100` → 11/15 | use min_run=100 primary, 30 as sensitivity; never claim 5-min HRV |
| EDA skew | 31/45 right-skewed (>1) | `log1p` justified; verify per subject |
| **EDA near-floor subjects** | median EDA: DF 0.07, 7E 0.09, CE 0.10, EG 0.10 µS | four subjects at risk of being unusable — see Phase 1 |
| EDA dead-signal sessions | 3/45 median < 0.05 µS; 5/45 with >20% floor time; exclusion rate 0.33 for 7E, DF, EG | subject-level problem, not a stray-session filter |
| Per-subject median EDA range | 0.07 → 1.82 µS (**26×**) | between-subject differences are multiplicative |
| Non-wear detected | median 0%, p90 0% | expected: off-wrist time falls *between* sessions |
| ICC(subject), raw scale, n=3/subject | temp 0.616, eda 0.174, acc_mag 0.136, hr 0.106 | **unstable and transform-dependent — see open questions** |
| ICC(subject), raw scale, n=7/subject | temp 0.522, eda 0.132, acc_mag 0.160, hr 0.127 | temp moved 0.62 → 0.52; use n=7 |

### Two measurement caveats introduced by our own tooling

1. **The Malik criterion biases RMSSD low at the high end.** Rejecting intervals differing >20%
   from their predecessor removes exactly the population RMSSD measures. Simulated: true 20 ms →
   19.9 (no bias), 40 ms → 40.2 (no bias), **80 ms → 68.7 (−14%)**. Observed `ibi_rmssd_med` has
   p90 = 53.5 and max = 76.0, so the top decile sits in the biased region. Because RMSSD *falls*
   under stress, this compresses the relaxed end toward the stressed end and reduces separability.
   **Use a split threshold:** `malik_gate=0.20` to decide which runs are usable, `malik_metric=0.30`
   to compute the reported value.
2. **`plausible_runs` is now redundant.** It sits downstream of Malik and rejects almost nothing
   (5C 161/161, DF 111/111, 83 68/68; only BG differs by one). Keep it as an assertion that should
   stay near zero, but `valid_frac` is the real quality metric.

### Implication of SCAR, stated once because everything depends on it

The naive-PU guarantee that ranking is preserved up to a constant factor requires
`P(labelled | positive)` to be constant. It varies 5.8× **by subject** — and subject is what LOSO
holds out. Any pooled model's decision threshold will not transfer between folds.

---

## Phase 1 — Units, labels, exclusions

### 1.1 Windowing — confirmed, not provisional

**120 s windows, 60 s hop.** Verified against the event-duration distribution:

| Window | Labelled events shorter than the window | Shorter than 2× the window |
|---|---|---|
| 60 s | 0 | 0 |
| **120 s** | **0** | **10** |
| 300 s | 15 | 71 |

No labelled event is shorter than 120 s, and only 6 of 179 level-2 events fall under 4 minutes.
300 s would discard 15 events outright. 120 s is the right choice and is now settled.

Windows must lie entirely within one session. Overlapping windows are near-duplicates: fine for
training, never split across train/test.

### 1.2 Label rule

Positive if ≥50% of the window falls inside a level-2 event. Negative only if it survives
Phase 2. Everything else is **excluded**, not zero.

### 1.3 Exclusions (in order)

| # | Rule | Rationale |
|---|---|---|
| 1 | Drop sessions < 5 min | 92 sessions; no usable window |
| 2 | Drop non-wear seconds (`flag_nonwear`) | notebook Step 7 |
| 3 | Drop sessions with `eda_med` < 0.05 µS or `eda_floor_pct` > 50 | dead sensor |
| 4 | Drop the 113 unrated (`na`) events from *both* classes | flagged but unrated |
| 5 | Drop the 3 exact-duplicate survey rows | Step 5 |
| 6 | Resolve the 12 overlapping event pairs — keep the outer span or drop the pair; decide once, document it | Step 5 |
| 7 | Exclude level-0 and level-1 events from the binary task | see below |
| 8 | Drop subject **6D** | 2 level-2 events, 0 h eligible negatives |
| 9 | **Conditionally drop DF, 7E, CE, EG** | chronically near-floor EDA — see 1.5 |

**On rule 7:** do *not* fold level 0 ("low stress") into the negative class. Those are windows a
nurse actively flagged. Putting them with unreported time merges your cleanest hard negatives
into your noisiest bucket. Exclude from the binary task; they return in Phase 6.

### 1.4 Effective-N table (measured)

Level-2 events after deduplication:

| Subject | Events | Minutes | ≈Windows | Eligible neg. hours | EDA median (µS) | HRV runs (median session) |
|---|---|---|---|---|---|---|
| 6D | 2 | 46 | 46 | 0.00 | 0.23 | 5 | ← **drop (rule 8)** |
| 7E | 3 | 71 | 71 | 19.62 | **0.09** | 0 | ← at risk |
| 5C | 4 | 136 | 136 | 9.02 | 0.40 | 38 |
| 94 | 5 | 69 | 69 | 15.40 | 0.52 | 0 |
| EG | 5 | 316 | 316 | 6.32 | **0.10** | 1 | ← at risk |
| DF | 6 | 439 | 439 | 30.40 | **0.07** | 4 | ← at risk |
| CE | 6 | 414 | 414 | 5.97 | **0.10** | 1 | ← at risk |
| 6B | 11 | 386 | 386 | 12.34 | 1.82 | 0 |
| 8B | 13 | 156 | 156 | 5.39 | 0.58 | 1 |
| BG | 14 | 323 | 323 | 25.48 | 0.37 | 44 |
| 15 | 16 | 277 | 277 | 15.08 | 0.20 | 1 |
| 83 | 16 | 662 | 662 | 44.54 | 1.30 | 16 |
| F5 | 24 | 236 | 236 | 37.11 | 0.18 | 12 |
| 7A | 25 | 521 | 521 | 35.55 | 0.27 | 6 |
| E4 | 28 | 817 | 817 | 31.12 | 1.17 | 2 |
| **Total** | **178** | **81.2 h** | **~4,869** | **293.3 h** | | |

Negative-to-positive ratio **3.6 : 1** before wear-masking and activity matching. Events, not
hours, are the binding constraint.

**Four subjects have ≤5 level-2 events** (7E, 5C, 94, EG). Report those folds separately.

### 1.5 The EDA-quality decision (new, and consequential)

DF, 7E, CE, and EG have median skin conductance of 0.07–0.10 µS. Resting EDA is normally
≥0.1–0.2 µS *with variability*; these are at or below the floor. Before Phase 2, run:

```python
for sub in ['DF', '7E', 'CE', 'EG']:
    g = DIAG[DIAG.subject == sub]
    print(sub, 'sessions', len(g),
          '| median eda_med %.3f' % g.eda_med.median(),
          '| median floor%% %.1f' % g.eda_floor_pct.median(),
          '| skew %.2f' % g.eda_skew.median())
```

Decide per subject: usable, EDA-excluded (keep HR/ACC features only), or dropped entirely.
Dropping all four takes you from 14 subjects to 10 and removes 20 of 178 level-2 events —
material, so make the call explicitly and document it rather than letting rule 3 silently
thin them.

**Re-run the Phase 2 negative-pool budget after this decision.** The 293.3 h figure predates it.

---

## Phase 2 — Negative construction

### 2.1 Eligibility

All of:

1. Session ≥30 min.
2. **On a subject-day containing at least one reported event.** The key rule — it restricts
   negatives to days when the nurse was demonstrably completing surveys, so absence of a report
   carries information. It is what reduces 1,255 h to 293 h.
3. ≥30 min from the boundary of *any* event, including the 113 unrated ones.
4. Worn, in a session that passed the dead-EDA check.

### 2.2 Activity matching — the part that matters

Do **not** filter negatives to low-motion periods. If positives contain motion and negatives
don't, the model learns to detect movement.

```python
POS  = windows[windows.label == 1]
CAND = windows[windows.eligible_negative]

edges = POS.acc_mag_mean.quantile(np.linspace(0, 1, 11)).values
edges[0], edges[-1] = -np.inf, np.inf
POS['abin']  = pd.cut(POS.acc_mag_mean,  edges, labels=False)
CAND['abin'] = pd.cut(CAND.acc_mag_mean, edges, labels=False)

RATIO = 3
neg = []
for sub, gp in POS.groupby('subject'):          # match WITHIN subject
    want = gp.abin.value_counts() * RATIO
    pool = CAND[CAND.subject == sub]
    for b, n in want.items():
        avail = pool[pool.abin == b]
        neg.append(avail.sample(min(int(n), len(avail)), random_state=0))
NEG = pd.concat(neg)
print(NEG.groupby('subject').size())            # report; some subjects will fall short
```

Match **within subject**. Global matching draws negatives disproportionately from the
high-coverage subjects (83, 7A, F5 hold 117 of the 293 hours), making subject identity a label
proxy — the same failure the SCAR test already exposed.

Repeat under ≥3 seeds and report variance.

### 2.3 Falsification checks (before any modelling)

| Check | Pass condition | If it fails |
|---|---|---|
| **ACC-only model** | well below full model | re-stratify; add `acc_sd`, `acc_p2p` to matching |
| **Time-of-day-only model** | near chance | add hour to the matching |
| Subject-ID from features | — | record as the ceiling Phase 4's probe must beat |
| Negative count per subject | ≥100 windows | drop or flag the subject |

### 2.4 Deliverable

A frozen table: `subject, session, window_start, label ∈ {0,1}, abin, seed`. Version it.

---

## Phase 3 — Baseline, then PU

### 3.1 Feature set

**EDA — from the native 4 Hz file, via `eda_features_native()` (notebook Step 6b), never the
1 Hz cache.** Tonic mean and slope, phasic SD, SCR count, SCR mean amplitude. Apply `log1p` to
level-like features (31/45 sessions right-skewed).

**HR** — mean, SD, slope, max; delta vs. trailing 30-min median.

**HRV — now included, which reverses v1.** RMSSD, SDNN, pNN50 computed on runs with
`min_run=100` (11/15 subjects qualify in their best session) and `malik_metric=0.30`. Carry an
`hrv_available` flag and a `valid_frac` covariate. **Never impute.** Subjects 6B, 7E, and 94 have
a median of 0 usable runs — for them HRV is structurally missing, not missing at random, so a
missing-indicator is required and mean-imputation would be actively misleading.

**ACC** — mean/SD/p2p of magnitude, fraction of seconds above a movement threshold.

**TEMP** — slope and delta-from-rolling-median only. **Never absolute temperature** (ICC 0.52–0.62
makes it a subject fingerprint). Note also that E4's session-wide range is ~1 °C against 6D's
7 °C — see open questions.

**Context** — hour of day, minutes into session.

### 3.2 Normalisation

`causal_z` (trailing 60-min robust z) within session. Leakage-safe by construction.

**Before finalising this, settle the ICC-on-log question (open question 1).** If ICC on
`log1p(eda)` is high, add per-subject centring on the log scale; if it stays low, the causal
rolling baseline alone is sufficient.

### 3.3 Baseline model

Gradient boosting, balanced class weights, LOSO. This is the number everything else must beat.
There is a real chance it wins outright.

### 3.4 nnPU, subject-stratified

Only after the baseline exists.

- Positives = level-2 windows. Unlabelled = the *full* eligible pool, not the matched subset.
- **Fit subject-stratified or with a per-subject labelling-propensity offset**, because `c`
  varies 5.8× by subject. Within a subject the Elkan–Noto ranking argument is defensible; pooled
  it is not.
- **π:** bracket, don't estimate. Fit at π ∈ {0.05, 0.10, 0.20, 0.30}. Anchor the range with the
  observed 8.0% labelled coverage; true prevalence is above that.
- Use the non-negative risk correction (Kiryo et al. 2017), not plain uPU.

### 3.5 Naive comparator

Same model, all unlabelled time as 0. Report **beside** the curated result. Do not use it to seed
or select curated negatives — its errors are systematically placed.

---

## Phase 4 — Self-supervised pretraining (optional)

Attempt only if Phase 3 clears a floor worth improving on.

### 4.1 Data and preprocessing

All 1,255 h, worn seconds only, `causal_z` normalised, absolute temperature dropped. Channels:
`eda, hr, acc_mag, acc_sd, temp_delta` at 1 Hz. Include unlabelled sessions and the subjects
excluded in Phase 1 — pretraining has no label requirement, and 6D's 24 h still carries signal.

### 4.2 Pretext task: masked reconstruction

Mask random 30–120 s spans across channels; reconstruct with MSE.

Chosen over contrastive learning deliberately: standard time-series augmentations include
amplitude scaling and jitter, and **EDA amplitude is the signal**. A scale-invariance objective
would train the encoder to discard exactly what you need.

Architecture: a 1D dilated CNN encoder (4–6 blocks) or a small transformer over 120 s windows.
Keep it small — ~4.5M timesteps is modest.

### 4.3 The subject-shortcut probe (mandatory gate)

Freeze the encoder, train a linear classifier on the embeddings to predict subject ID.

- Near chance (~1/n_subjects) → proceed.
- High accuracy → apply a gradient-reversal layer on a subject head, strengthen per-session
  normalisation, or sample reconstruction targets within-session only. Re-probe.

Do not skip. With ICC(temp) at 0.52–0.62 the shortcut is available, and unsupervised objectives
find shortcuts.

### 4.4 The leakage trap

Pretraining on all subjects then evaluating LOSO is **transductive**. Either **(a)** re-run
pretraining inside each fold excluding the test subject (n× compute, clean), or **(b)** pretrain
once and label the result transductive, reporting it as an upper bound. Pick one, state it.

---

## Phase 5 — Fine-tuning and model comparison (NEW)

The point of Phases 4–5 is a single question: *does a pretrained representation beat hand-crafted
features on 178 events?* Answer it cleanly or not at all.

### 5.1 Two variants, both required

| Variant | What | Why |
|---|---|---|
| **Linear probe** | freeze the encoder, fit logistic regression / a single dense layer on the embeddings | tests representation quality directly; with ~4,869 positive windows this may well be the winner |
| **Full fine-tune** | unfreeze with discriminative learning rates — encoder at ~1/10 the head's | tests whether task-specific adaptation helps or just overfits |

If full fine-tuning loses to the linear probe, that is a data-volume verdict, not a bug. Report it.

### 5.2 Nested validation — the trap specific to this phase

Fine-tuning needs a validation set for early stopping and learning-rate selection. **A random
split of the training windows is invalid** — adjacent windows are near-duplicates, so an inner
random split leaks and early stopping will fire at the wrong epoch.

Use a **nested group split**: hold out one or two *subjects* from the training folds as inner
validation.

```python
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit

outer = LeaveOneGroupOut()
for tr_idx, te_idx in outer.split(X, y, groups=subject):
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    fit_idx, val_idx = next(inner.split(X[tr_idx], y[tr_idx], groups=subject[tr_idx]))
    # fit on tr_idx[fit_idx], early-stop on tr_idx[val_idx], score on te_idx
```

With ~10–14 subjects, an inner split costs you 2 subjects per fold. That is expensive but
unavoidable; the alternative is a leaked stopping criterion.

### 5.3 Class imbalance in the head

Class weights and threshold tuning. **No SMOTE** — interpolating between adjacent autocorrelated
windows manufactures copies of real samples, and applied before the split it leaks outright.

### 5.4 Comparison protocol

All models scored on **identical folds and identical windows**:

| Model | Source |
|---|---|
| Gradient boosting on hand-crafted features | Phase 3.3 |
| nnPU on hand-crafted features | Phase 3.4 |
| Linear probe on pretrained embeddings | 5.1 |
| Full fine-tune | 5.1 |
| ACC-only | falsification |

Compare **per fold**, not by mean. With ~10–14 subjects, a paired Wilcoxon signed-rank test
across folds is the appropriate comparison, and even then it is underpowered — a difference
smaller than the between-fold spread is not a difference.

### 5.5 Pre-commit to an adoption rule

Write this down before running anything:

> Adopt the pretrained model only if it beats the Phase 3 baseline on event-level recall at the
> fixed false-alarm rate in **at least k of n folds**, with no fold degrading by more than X.

Without a pre-committed rule, four model variants across a dozen folds will always yield a
configuration that "wins."

### 5.6 Expected failure modes

| Symptom | Reading |
|---|---|
| Full FT ≪ linear probe | too little labelled data to adapt the encoder; report the probe |
| Both ≪ baseline | the pretext task didn't capture stress-relevant structure |
| Both ≫ baseline, but subject probe is high | you are measuring subject identity — go back to 4.3 |
| Large fold-to-fold variance | expected; report per-fold and do not average over the low-count subjects |

---

## Phase 6 — Severity, evaluation, reporting

### 6.1 Severity

Detection uses level 2 vs. constructed baseline. The 0/1/2 scale is a separate, easier task on
the 245 labelled events, requiring no negatives.

**The scales are not comparable across subjects.** F5 rated 1/0/25; 94 rated 11/4/5. Therefore:

- Ordinal model (ordered logit), **not** 3-class softmax.
- Mixed-effects with a per-subject random intercept, or normalise ratings to within-subject ranks.
- With 20 level-1 events across the study and 9 subjects having none, don't expect the middle
  class to be learnable. Consider collapsing to {0,1} vs {2}.

### 6.2 Evaluation protocol

- **LeaveOneGroupOut on subject.** Report per-fold counts alongside per-fold metrics.
- **Event-level scoring**: detected if ≥*k*% of its windows fire (fix *k* = 50 in advance;
  report sensitivity to *k* as secondary).
- **Primary metric**: event-level recall at a fixed **false alarms per worn hour** — the
  clinically meaningful operating point, and more informative than AUC given a constructed
  negative class.
- Report the low-count folds (7E, 5C, 94, EG) separately.
- **No SMOTE.**

### 6.3 Required ablations

| Ablation | Answers |
|---|---|
| ACC-only | is it an activity detector? |
| Time-of-day-only | is it a shift-schedule detector? |
| Naive vs. curated negatives | how much does negative construction matter? |
| π ∈ {0.05, 0.10, 0.20, 0.30} | how much does the unknowable prior matter? |
| EDA-only / HR-only / no-HRV | which channel carries it? |
| With vs. without the four near-floor-EDA subjects | is the result driven by data quality? |
| Subject-ID probe | is the representation contaminated? |
| ≥3 negative-sampling seeds | how much is sampling noise? |
| `min_run` ∈ {30, 100} for HRV | does the HRV threshold change conclusions? |

### 6.4 Threats to validity to state explicitly

1. **SCAR is violated** (AUC 0.728). Naive PU ranking guarantees don't hold across subjects.
2. **π is unidentifiable** from this data.
3. **Negatives are constructed, not observed.** Every number is conditional on Phase 2's recipe.
4. **RMSSD is conservative** by ~14% at the high end, from Malik filtering.
5. **Four subjects have marginal EDA**; results should be reported with and without them.
6. **Cohort**: 15 female nurses, one hospital, COVID-era. External validity is narrow.
7. **Documentation discrepancy**: sessions run 2020-04-14 → 2020-12-13; the source paper
   documents only Apr–May and Nov–Dec.
8. **Long events**: 34 exceed 60 min, max 323 min. Event 21 (94 min) showed sustained EDA
   elevation across its whole span, so these may be genuine — say so rather than assuming.

### 6.5 Preregister

Before fitting anything in Phase 3, write down the window, label rule, exclusion list, primary
metric and operating point, the Phase 5 adoption rule, and the ablation list. With 178 positives
across ≤14 subjects, undisclosed flexibility will produce whatever result you go looking for.

---

## Open questions — verify before Phase 2

| # | Question | How | Blocks |
|---|---|---|---|
| 1 | Is ICC(eda) high on the **log** scale? Raw ICC is 0.13–0.17 despite a 26× range in per-subject medians, because raw-scale within-subject variance is dominated by high-baseline subjects. | recompute `icc1` on `log1p(eda)` with `N_PER_SUBJECT=7` | Phase 3.2 normalisation choice |
| 2 | Are `eda_med` and `eda_skew` coupled? Across 15 subjects Spearman ρ = −0.43 (p = 0.11) — suggestive that high skew is partly a quality signal, not a shape question. | `DIAG[['eda_med','eda_skew','eda_floor_pct']].corr(method='spearman')` on the 45 session-level points | interpretation of rule 3 |
| 3 | Are DF, 7E, CE, EG usable at all? | §1.5 | Phase 1 exclusions, Phase 2 budget |
| 4 | Why is E4's temperature range ~1 °C (33.91–34.87) vs 6D's 7 °C? | plot one E4 session end to end | whether TEMP features are trustworthy |
| 5 | How much does Malik truncate real RMSSD here? | recompute RMSSD at malik ∈ {0.20, 0.30, 0.50} on the cleanest sessions (5C, DF, BG) | Phase 3.1 HRV features |
| 6 | Does the negative pool survive the Phase 1.5 exclusions? | re-run the 293.3 h budget after deciding on the four subjects | Phase 2 feasibility |
| 7 | Do the 46 sessions with button presses corroborate the timezone conclusion? | compare `tags.csv` timestamps to survey event onsets | independent validation of Step 4 |
| 8 | Is the Jun–Aug data a third undocumented collection phase? | ask the authors | anything citing the collection period |

---

## Decision points

| # | Decision | Default | Revisit if |
|---|---|---|---|
| 1 | Binary level-2 vs. 3-class | binary | a different label rule grows the level-1 count |
| 2 | Level 0 → negative? | **no**, exclude | — |
| 3 | Window length | **120 s, settled** | — |
| 4 | Guard band | 30 min | negative pool < 100 windows/subject |
| 5 | Same-day rule | on | pool too small; relaxing it weakens every negative |
| 6 | HRV `min_run` | 100 | report 30 as sensitivity |
| 7 | Near-floor-EDA subjects | decide in §1.5 | — |
| 8 | Pretrain per fold | (a) if compute allows | otherwise report transductive |

---

## Sequencing

**Week 1** — Open questions 1–6. Cheap, and three of them change Phase 1–3 decisions.
**Weeks 2–3** — Phases 1 and 2. Frozen window table, negative pool, falsification checks.
**Week 4** — Phase 3.1–3.3 baseline. LOSO, per-fold counts, ablations.
**Week 5** — Phase 3.4–3.5 nnPU and naive comparator, π grid.
**Weeks 6–8** — Phases 4 and 5, only if warranted.
**Week 9** — Phase 6 severity model and write-up.

Phases 1–3 stand alone. "Curated-negative detection reaches X% event recall at Y false alarms
per hour, and the naive construction inflates this to Z" is a complete, honest, publishable
result whether or not X is impressive.

---

**Superseded in part.** The Phase 0 facts in this file and in `PLAN.md` are
reconciled against the full-archive audit in `reports/audit/audit_summary.md`,
which is the merged source of truth. Notable: the HRV viability upgrade in this
version does not hold at the window level (0.27% coverage at `min_run=100`), and
the propensity AUC of 0.728 appears to be in-sample against a cross-validated
0.707. The near-floor EDA finding in this file is confirmed and strengthened.
