# PLAN.md

Executable research plan. **This document follows `nurse_stress_analysis_plan.md`.**
Its five phases, section numbering, exclusion order, effective-N table, decision
points and week sequencing are carried over and are authoritative.

Judgment calls referenced as `JCnn` are defined in `judgment_calls.yaml`.

### Notation for changes to the source plan

| Mark | Meaning |
|---|---|
| *(unmarked)* | Carried from the source plan without modification |
| **[+]** | Addition. Fills a gap the source plan does not cover |
| **[Δ]** | Amendment. Changes something the source plan specifies, with reason given |

Nothing in the source plan has been dropped. Two items are amended and both are
argued for at the point of amendment.

---

# PART 1 — HIGH-LEVEL PLAN

**The task.** Detect high-stress periods (survey level 2) from wrist physiology,
evaluated leave-one-subject-out at the event level.

**The two problems, which are independent.**

| Problem | Size | Addressed by |
|---|---|---|
| No negative class exists in the labels | 87.7% of sensor time is unlabelled | Phase 2 (curated negatives) + Phase 3 (PU) |
| Very few positives | 178 level-2 events across 14 usable subjects | Phase 4 (self-supervised pretraining) |

Self-supervised pretraining does not manufacture negatives. Solve the negative
problem first; a well-pretrained model fitted against a biased target is still
biased.

**Five phases.** As specified in the source plan.

**The three checks that decide whether any of it is real.** ACC-only ablation.
Subject-ID probe on the embedding. Naive against curated negatives, reported side
by side.

**[+] The PCS framing, layered on top rather than replacing anything.**

The source plan already treats the negative construction as the object of study
("the gap is a result either way"). That is a stability claim in the sense of Yu
and Kumbier. Making it explicit costs nothing and gives the write-up a frame:

| Principle | Where it already lives in the source plan |
|---|---|
| **Predictability** | LOSO at the event level, §5.2 |
| **Computability** | Phase 3.1 baseline is cheap by design, which is what makes the ablation grid in §5.3 affordable |
| **Stability** | §5.3 ablations, the π grid, the ≥3 negative-sampling seeds |

The one addition is to run the ablations of §5.3 as a **perturbation study** rather
than as a checklist, and to compare their spread against the spread from resampling
subjects. See §5.4 **[+]**.

---

# PART 2 — DETAILED PLAN

## Phase 0 — Established facts (carry these forward)

| Fact | Value | Source |
|---|---|---|
| Sessions / sensor time | 609 / 1,255 h | Step 2 |
| Median session length | 1.17 h; 92 sessions < 5 min | Step 2 |
| Survey events | 358 rows, 245 labelled, 113 unrated | Step 5 |
| Stress levels | 46 low / 20 medium / 179 high | Step 5 |
| Sensor time inside a labelled event | 99.9 h (8.0%) | derived |
| Timezone | survey `America/Chicago`, sensors UTC | Step 4 |
| **SCAR violated** | AUC 0.728; per-subject P(labelled) spans 0.125–0.724 (**5.8×**) | derived |
| ICC(subject) for temperature | 0.522 | Step 11 |
| HRV availability | 75/105 sampled sessions have ≥1 usable run; median 3 | Step 8b |
| EDA skew | right-skewed (>1) in 72/105 sessions | Step 8b |

**Implication of the SCAR result.** The naive-PU guarantee that ranking is
preserved up to a constant factor requires `P(labelled | positive)` to be constant.
Here it varies 5.8× by subject, and subject is exactly what LOSO holds out. Any
pooled model's decision threshold will not transfer between folds.

### T0.1 Verify the facts table
Run `tasks/t00_audit.py` against the archive. Confirm every row above. A mismatch
means the extraction differs from the reference notebook, and everything downstream
needs rechecking before proceeding.

### T0.2 **[+]** Resolve the documentation discrepancy cheaply
Threat 5.4.5 proposes contacting the authors about session dates running
2020-04-14 to 2020-12-13 against documented Apr–May and Nov–Dec. This may not
require correspondence. Published accounts of this dataset disagree with each
other; at least one gives 15 April to 6 August and 8 October to 11 December, which
is consistent with the observed range. Read the Scientific Data article itself
before writing to anyone.

---

## Phase 1 — Units, labels, exclusions

### 1.1 Windowing
120 s windows, 60 s hop (`JC11`, `JC12`). Long enough for a skin conductance
response to develop, short enough that a 5-minute event yields several windows.
Windows lie entirely within one session and never span a session boundary.
Overlapping windows are fine for training and must never be split across
train/test.

**[+]** Evaluation uses non-overlapping windows regardless of the training hop.
Overlapping test windows are correlated and narrow any reported spread artificially.

### 1.2 Label rule
A window is **positive** if ≥50% of its span falls inside a level-2 event
(`JC07`). A window is **negative** only if it survives Phase 2. Everything else is
**excluded**, not zero.

### 1.3 Exclusions (apply in this order)

| # | Rule | JC |
|---|---|---|
| 1 | Drop sessions < 5 min | `JC02` |
| 2 | Drop non-wear seconds (`flag_nonwear`) | `JC03` |
| 3 | Drop sessions with median EDA < 0.05 µS or `eda_floor_pct` > 50 | `JC04` |
| 4 | Drop the 113 unrated (`na`) events from *both* classes | `JC06` |
| 5 | Drop the 3 exact-duplicate survey rows | `JC05` |
| 6 | Resolve the 12 overlapping event pairs | `JC08` |
| 7 | Exclude level-0 and level-1 events from the binary task | `JC07` |
| 8 | Drop subject **6D** (2 events, 0 h eligible negatives) | — |

**On rule 7.** Level-0 windows are windows a nurse actively flagged. Putting them
with unreported time merges the cleanest hard negatives into the noisiest bucket.
They return in Phase 5.

**[+]** Rules 4, 6 and 7 are genuine forks rather than obvious cleanups, so each
carries alternatives in the registry and each is perturbed in §5.4. The source
plan's reasoning for the defaults is recorded there verbatim as the narrative
justification, which is what PCS documentation requires.

### 1.4 Effective-N table (measured)

Carried unchanged. 178 level-2 events, 81.2 h, ~4,869 windows, 293.3 h eligible
negatives across 14 retained subjects.

Four subjects have ≤5 level-2 events (7E, 5C, 94, EG). Their folds cannot support a
confident metric and are reported separately rather than averaged in.

### **[+]** 1.5 Structural EDA

Six figures, all safe to compute on every subject because none touches the
feature-label relationship. That distinction governs the whole EDA programme:

> **Structural** exploration describes the data's shape and is safe globally.
> **Relational** exploration describes how features relate to labels, can influence
> a judgment call, and must run on training folds only. Looking at a feature-label
> relationship in a held-out subject and then choosing a preprocessing rule is the
> same leak as tuning a threshold on the test set, and it is harder to notice
> because no model was fitted.

**F01 · Wear raster.** One row per subject, calendar time horizontal, filled where
a session exists, coloured by session length. Shows the two collection periods and
who contributed when. *Pathological:* one subject with a solid block while others
are sparse, meaning folds are not comparable.

**F02 · Session duration ECDF.** Log horizontal axis, `JC02` candidate cutoffs
marked. An ECDF rather than a histogram, because you want to read "what fraction is
below 5 minutes" directly instead of estimating it from bar areas.

**F03 · Signal presence matrix.** Sessions against signal files. *Pathological:*
`IBI.csv` missing in a block belonging to one subject, which means HRV availability
is confounded with subject before anything has been fitted.

**F04 · Sample rate consistency.** One bar per signal showing distinct rates
observed. Every bar should be at height one. Read the rate from row 2 of each CSV;
published descriptions of this dataset conflict with the device specification.

**F05 · Time budget, global.** One stacked bar over all sensor time: non-wear and
artifact, level-2 positive, other labelled events, eligible unlabelled, ineligible
unlabelled. Annotate hours and percentages. The visual sliver of the positive
segment is worth more than reading "8.0%".

**F06 · Time budget per subject.** Same decomposition, one bar per subject, sorted
by positive fraction. This is the 5.8× SCAR spread made visible, and it is the
figure that justifies subject-stratified fitting in §3.3.

**F07 · Event duration ECDF by level.** One curve per level, vertical line at 60
minutes. This figure is the response to threat 5.4.6 and to the published
reanalysis that excluded this dataset over event-duration implausibility. It
belongs in the paper.

**F08 · Events against minutes, per subject.** Scatter with iso-duration lines.
Reveals that EG, CE and DF are few-long-event subjects while 7E and 94 are
few-short-event subjects. The upper-left group is the same group whose eligible
negative pool collapses, because a long event plus a 30-minute guard band consumes
the day. Threat 5.4.6 and the negative-scarcity problem are one problem, and this
figure shows it.

### 1.6 Deliverable
Frozen `derived/windows.parquet`. Later phases read it and never rewrite it.
Unfreezing requires an entry in `provenance/unfreeze_log.md`.

---

## Phase 2 — Negative construction

### 2.1 Eligibility

A candidate negative window satisfies all of:

1. Within a session ≥30 min long.
2. **On a subject-day containing at least one reported event** (`JC18`). The key
   rule. It restricts negatives to days when the nurse was demonstrably completing
   surveys, so absence of a report carries information. It is what reduces 1,255 h
   to 293 h.
3. ≥30 min from the boundary of *any* event, including the 113 unrated ones
   (`JC19`).
4. Worn, in a session that passed the dead-EDA check.

### **[Δ]** 2.1b Audit achievable ratios before fixing `RATIO`

The source plan sets `RATIO = 3` and states that a 3.6:1 pool "is enough. The
binding constraint is events, not hours." The aggregate figure is right and the
per-subject picture is not. Converting the effective-N table to common units:

| Subject | Positive min | Eligible negative min | Achievable ratio |
|---|---|---|---|
| CE | 414 | 358 | **0.9 : 1** |
| EG | 316 | 379 | **1.2 : 1** |
| 6B | 386 | 740 | **1.9 : 1** |
| 8B | 156 | 323 | **2.1 : 1** |
| E4 | 817 | 1,867 | **2.3 : 1** |
| 15 | 277 | 905 | 3.3 : 1 |
| 5C | 136 | 541 | 4.0 : 1 |
| 83 | 662 | 2,672 | 4.0 : 1 |
| 7A | 521 | 2,133 | 4.1 : 1 |
| DF | 439 | 1,824 | 4.2 : 1 |
| BG | 323 | 1,529 | 4.7 : 1 |
| F5 | 236 | 2,227 | 9.4 : 1 |
| 94 | 69 | 924 | 13.4 : 1 |
| 7E | 71 | 1,177 | 16.6 : 1 |

Five of fourteen subjects cannot reach 3:1, and CE has fewer eligible negative
minutes than positive minutes. Decile matching reduces these further, since a
subject can only draw from bins populated in her own eligible pool.

The source plan's falsification check 4 (`≥100 windows per subject`) and decision
point 4 both catch this downstream. The amendment is to compute the table
**before** fixing `RATIO`, not after, because the consequence is not simply that
some subjects fall short. Class balance would then vary by an order of magnitude
across folds, and since balance moves the decision threshold and threshold transfer
is already compromised by the 5.8× SCAR spread, the two failures compound.

**Amended default (`JC21`):** set `RATIO` to the minimum achievable across retained
subjects, applied uniformly. Alternative: record it per subject as a reported
covariate. Either is defensible. Taking whatever each subject can supply, silently,
is not.

**[+]** Note the connection to F08: the scarce-negative subjects are the long-event
subjects. If `JC09` boundary trimming or `JC19` guard-band reduction is used to
relieve the shortage, say which problem is being traded against which.

### 2.2 Activity matching

Do not filter negatives to low-motion periods. If positives contain motion and
negatives do not, the model learns to detect movement.

Decile-stratify on activity, then sample negatives to mirror the positive
histogram, **matching within subject**, not globally. Global matching draws the
negative class disproportionately from high-coverage subjects (83, 7A, F5) and
makes subject identity a label proxy. Code as given in the source plan §2.2.

Repeat the draw under ≥3 seeds (`JC22`) and report variance.

**[+] F10 · Achievable negative ratio per subject.** Sorted horizontal bars with the
requested ratio marked. The figure form of table 2.1b.

**[+] F11 · Activity distribution, positives against candidate negatives**
`[relational]`. Overlaid densities of `acc_mag_mean`, drawn twice, before and after
matching, faceted by the four highest-event subjects. Healthy after matching means
the densities superimpose. Residual separation means the pool had no mass in the
deciles the positives occupy, which is the failure mode falsification check 1 is
designed to catch, seen directly.

**[+] F12 · Clock hour, positives against matched negatives** `[relational]`.
Matching on activity does not match on circadian phase and a model can learn
either. Two confounds, two figures.

### 2.3 Falsification checks (run before any modelling)

| Check | Pass condition | If it fails |
|---|---|---|
| ACC-only model | well below full model | matching failed; re-stratify, add `acc_sd` and `acc_p2p` to the matching variables |
| Time-of-day-only model | near chance | events cluster in shift hours; add hour to the matching |
| Subject-ID from features | — | expect high; record as the ceiling Phase 4's probe must beat |
| Negative count per subject | ≥100 windows | drop or flag the subject |

**[+]** These fit models and therefore use the fold structure from §3.0. Running
them without subject-level splitting would let overfitting masquerade as a pass.

### 2.4 Deliverable
Frozen `subject, session, window_start, label ∈ {0,1}, abin, seed`. Versioned.

---

## Phase 3 — Baseline, then PU

### **[+]** 3.0 Fold structure, defined once and reused by every later phase

The source plan specifies LeaveOneGroupOut on subject (§5.2) and an operating point
in false alarms per worn hour (§5.2), but does not say where the threshold
achieving that operating point comes from. If it is read off the test subject, the
reported recall is selected rather than estimated.

The plan's own SCAR finding sharpens this: a pooled threshold will not transfer. So
the rule must be stated. Two defensible options (`JC26`):

**(a) Inner validation subjects.** Within each of the 14 folds, hold out 2 of the
remaining 13 to set the threshold, train on 11, apply unchanged to the test
subject. Estimates performance for a new subject with no calibration data. Harder
claim, and the one a reviewer assumes unless told otherwise.

**(b) Per-subject calibration.** Set the threshold on the test subject's earliest
eligible sessions, evaluate on the remainder, excluding the calibration hours.
Deployable, since a real system calibrates on arrival. More favourable claim.

Report (a) as primary and (b) as the calibrated variant. Option (b) is the better
match to the SCAR finding and should be described as such rather than as a
concession.

### 3.1 Baseline first (do not skip)

Hand-crafted features per 120 s window, as specified in the source plan: EDA tonic
mean/SD/slope plus SCR count, amplitude, rise time with `log1p`; HR
mean/SD/slope/max and delta against trailing 30 min median; HRV RMSSD/SDNN/pNN50
where `ibi_usable_runs > 0` with an `hrv_available` indicator and never imputed;
ACC mean/SD/p2p of magnitude and fraction above threshold; TEMP slope and
delta-from-rolling-median only, never absolute; hour of day and minutes into
session.

Gradient boosting, balanced class weight, LOSO. This is the number everything else
must beat and there is a real chance it wins outright.

**[+] Report the majority-class baseline beside it.** Under heavy imbalance a model
can post a respectable F1 while an always-negative classifier posts a higher one.

**[+] F13–F16 · Relational EDA, training folds only.**

- **F13 · Per-channel densities by class**, small multiples. The common wrong
  conclusion from overlapping densities is that there is no signal; they may only
  mean the pooled view hides within-subject structure. Read with F14.
- **F14 · EDA density by class, faceted by subject.** If each panel shows a small
  consistent shift while the panels sit at different locations, per-subject
  normalisation is doing essential work and F13 was misleading. This is the visual
  form of the ICC(EDA) = 0.174 finding.
- **F15 · Standardised mean difference heatmap**, subjects against features,
  diverging scale at zero. Look for vertical consistency. A feature whose sign flips
  between subjects is a subject-specific artifact and will not transfer.
- **F16 · Event-triggered average.** Align every training-fold event at onset, plot
  mean EDA and HR from −30 to +60 min with a standard-error band. **This is the
  figure to run first.** It asks whether anything physiological happens when a nurse
  says something happened. A visible rise means the labels mark a real state and the
  problem is well posed. A flat trace means they do not, at this resolution, and no
  modelling repairs it. It also shows where the informative part of the interval
  sits, which is the empirical input to `JC09`.

**[+] F17–F18 · HRV usability.** Decision point 7 includes HRV with an availability
flag. Two figures decide whether that default holds.

- **F17 · Window-level HRV coverage.** The Step 8b figure of 75/105 sessions is
  session-level and flatters the situation, because a session with three short runs
  still leaves most of its windows uncovered. Compute the fraction of admissible
  120 s windows carrying a usable run. That is the number that matters.
- **F18 · HRV availability by ACC decile and class.** Beat rejection is
  motion-driven, motion tracks being on the ward, and events happen on the ward. If
  availability differs by class within a decile, missingness carries label
  information. Note that LightGBM routes missing values natively, so a tree can
  split on missingness whether or not you supply the indicator. "Never impute" does
  not close this by itself. If F18 fires, run the baseline without HRV and bring it
  back as the `no-HRV` ablation already listed in §5.3.

### 3.2 Normalisation
`causal_z`, trailing 60 min robust z on all continuous channels, computed within
session (`JC13`). Leakage-safe by construction and addresses within-subject drift
dominating between-subject variance for EDA (ICC 0.174) and HR (0.106).

### 3.3 nnPU, subject-stratified
Only after the baseline exists.

Positives are level-2 windows. Unlabelled is the *full* eligible pool, not the
matched subset. Fit subject-stratified or with a per-subject labelling-propensity
offset, because `c` varies 5.8× by subject and a single pooled `c` is not
defensible. Within a subject the Elkan–Noto ranking argument is much more
defensible, and that is the salvage.

Bracket π rather than estimating it: π ∈ {0.05, 0.10, 0.20, 0.30} (`JC25`),
anchored by the observed 8.0% coverage with under-reporting implying the truth sits
above it. Use the non-negative risk correction, not plain uPU.

**[+] Implementation note.** nnPU is a loss, not a model. LightGBM custom objectives
are awkward, so the nnPU arm runs on penalised logistic regression or a small
PyTorch network. Record which pairing produced which number, since the baseline and
the nnPU arm then differ in two respects rather than one.

### 3.4 Naive comparator
Same model, all unlabelled time as 0. Report beside the curated result. Do not use
it to select or seed the curated negatives; its errors are systematically placed and
bootstrapping from it launders the bias.

### **[+]** 3.5 Comparison procedure
Every condition is evaluated on identical folds, so comparisons are paired. Take the
per-fold differences and test with a Wilcoxon signed-rank test, excluding the four
low-count folds and reporting them separately as §5.2 already requires. Report the
median paired difference and the full difference distribution, not significance
alone.

State in the methods that LOSO fold estimates are correlated, since any two folds
share 12 of 13 training subjects, and that no unbiased estimator of the variance of
k-fold cross-validation exists. Report the spread descriptively; do not divide it by
the square root of the fold count.

---

## Phase 4 — Self-supervised pretraining (optional)

Attempt only if Phase 3 clears a floor worth improving on.

### 4.1 Data and preprocessing
All 1,255 h, worn seconds only, `causal_z` normalised, absolute temperature dropped.
Channels `eda, hr, acc_mag, acc_sd, temp_delta` at 1 Hz (`JC14`). Include unlabelled
sessions and subject 6D; pretraining has no label requirement.

### 4.2 Pretext task: masked reconstruction
Mask random 30–120 s spans across channels, reconstruct with MSE (`JC28`).

Chosen over contrastive learning deliberately: standard time-series augmentations
include amplitude scaling and jitter, and EDA amplitude is the signal. A
scale-invariant objective would train the encoder to discard exactly what is needed.
**[+]** This reasoning belongs in the paper, since reviewers will ask why contrastive
methods were not attempted.

Architecture: 1D dilated CNN encoder of roughly 4–6 blocks, or a small transformer
over 120 s windows. Keep it small; ~4.5M timesteps is modest.

### 4.3 The subject-shortcut probe (mandatory gate)
Freeze the encoder, train a linear classifier on the embeddings to predict subject
ID. Near chance (~1/14) proceeds. High accuracy means the encoder learned identity;
mitigate with a gradient-reversal layer on a subject head, stronger per-session
normalisation, or within-session reconstruction targets, then re-probe.

Do not skip. With ICC(temp) = 0.522 the shortcut is available and unsupervised
objectives find shortcuts.

### 4.4 The leakage trap
Pretraining on all subjects and evaluating LOSO is transductive. Either **(a)**
re-run pretraining inside each fold excluding the test subject, or **(b)** pretrain
once and label the result transductive, reporting it as an upper bound. Pick one,
state it. Most published wearable-SSL results quietly do (b).

**[+] One detail if (a) is chosen.** Exclude the two inner validation subjects from
pretraining as well, not only the test subject. They supply the threshold, and a
threshold set on subjects whose physiology shaped the encoder is fitted to them
rather than estimated for a stranger. So pretraining sees 11 subjects, and
convergence is monitored on held-out windows drawn from those same 11.

**[+] Second detail.** Fix mask span and pretraining duration a priori. Selecting
them by observing downstream fold performance leaks through the selection process
even when each individual run excluded its test subject.

### 4.5 Fine-tuning
Replace the reconstruction head with a classification head. Two variants: frozen
encoder plus linear probe, and full fine-tune with a low encoder learning rate. With
~4,869 positive windows the linear probe may well win.

### **[+]** 4.6 Label-efficiency sweep — the primary Phase 4 deliverable

Frozen-against-fine-tuned answers which head is better. It does not answer the
question that justifies pretraining at all, which is whether it reduces the labels
required.

Subsample the events supplied to fine-tuning at 25, 50, 100 and all 178, holding
folds and negatives fixed, for both the pretrained and the randomly initialised
encoder. Plot event-level recall against event count.

Separation at low label counts that closes as labels increase is the result, and the
crossover point is directly quotable. Absence of separation is a reportable negative
finding about self-supervision at this scale, which is more useful than a marginal
improvement. This is what makes Phase 4 worth the compute rather than an appendix.

**[+] A consequence of §4.1 worth stating in the write-up.** At 1 Hz with this
channel set the encoder has no access to cardiac timing, since Empatica's HR output
is already smoothed and a 1 Hz grid cannot carry a waveform. Self-supervision as
scoped removes HRV rather than rescuing it. If F18 shows HRV carries signal, either
add a 64 Hz BVP branch or concatenate hand-crafted HRV onto the embedding. Decide
after the ablation, not before.

---

## Phase 5 — Severity, evaluation, reporting

### 5.1 Severity
Ordinal model on the 245 labelled events, requiring no negatives. Ordered logit or
ordinal regression, not 3-class softmax. Mixed effects with a per-subject random
intercept, or within-subject rank normalisation, because the scales are not
comparable across subjects (F5 rated 1/0/25; 94 rated 11/4/5). With 20 level-1
events and 9 subjects having none, do not expect the middle class to be learnable;
consider collapsing to {0,1} against {2}.

### 5.2 Evaluation protocol
LeaveOneGroupOut on subject, per-fold counts always reported beside per-fold
metrics. Event-level scoring with an event counted as detected if ≥50% of its
windows fire, *k* fixed in advance with sensitivity to *k* secondary. Primary metric
is event-level recall at a fixed false-alarms-per-worn-hour operating point. Report
the four low-count folds separately. No SMOTE.

**[+]** Conventional precision and specificity are not reported as performance
claims, because the held-out subject's unlabelled windows are themselves
contaminated and a flagged unlabelled window cannot be distinguished between a false
alarm and an unreported episode. State this in the metrics subsection rather than
deferring it to limitations.

### 5.3 Required ablations
ACC-only. Time-of-day-only. Naive against curated negatives. π ∈ {0.05, 0.10, 0.20,
0.30}. EDA-only, HR-only, no-HRV. Subject-ID probe. ≥3 negative-sampling seeds.

### **[+]** 5.4 Run §5.3 as a perturbation study

The ablation list is already a stability analysis. Two changes turn it into the
paper's headline at little extra cost.

**5.4a Add a data perturbation arm.** Bootstrap over subjects, 200 replicates,
recomputing the primary metric. Resample subjects, not windows.

**5.4b Plot the two spreads on one axis.** The distribution of the primary metric
under subject resampling, beside its distribution across the §5.3 ablation grid.

Yu and Barter report these as comparable in magnitude in a housing-price setting.
Whether that holds in a physiological positive-unlabeled setting is unknown, and
either answer is a result. If judgment-call spread is comparable to or wider than
sampling spread, then intervals reported in this literature capture a fraction of
the real uncertainty. That claim, with this figure under it, is a stronger paper
than a recall number.

Supporting figure: the metric distribution per perturbed decision, sorted by
influence, showing which choices matter and which do not.

### **[+]** 5.5 Heterogeneity

The 14 per-fold numbers will have substantial spread. Reporting it is necessary;
explaining it is the contribution.

Regress per-fold performance on per-subject covariates already computed: event
count, mean event duration, labelling propensity, achievable negative ratio, HRV
coverage, EDA skew. Descriptive at 14 subjects and labelled as such. It answers who
the model fails for and whether failure is predictable from measurable properties of
the person, which is where the accountability argument lives.

### 5.6 Threats to validity to state explicitly
As enumerated in the source plan: SCAR violated; π unidentifiable; negatives
constructed rather than observed; narrow cohort; documentation discrepancy (see
T0.2); long events.

**[+]** Add: a published reanalysis excluded this dataset over event-duration
implausibility and absent cooling-down periods. Address it directly rather than
waiting for a reviewer. The source plan's inspection of event 21 is evidence but is
a single case; strengthen it into a trimming grid, `JC09` ∈ {0, 2, 5, 10} minutes,
and report whether conclusions move. Then state the inversion: a study of what label
assumptions cost has an obvious reason to retain a dataset whose labels are
ambiguous.

### 5.7 Preregister
Before fitting anything in Phase 3, write down the window, the label rule, the
exclusion list, the primary metric and operating point, and the ablation list. With
178 positives across 14 subjects the space of defensible analyses is large enough
that undisclosed flexibility will produce whatever result you go looking for.

**[+]** `judgment_calls.yaml` is that preregistration. Freeze it with a git tag
before Phase 3 begins and reference the tag in the paper.

### **[+]** 5.8 Final artifact
Cross-validation estimates the procedure, not any surviving model. After results are
locked, retrain once on all 14 subjects under the frozen recipe and release that
checkpoint, stating that it carries no held-out score of its own.

Never average weights across folds. Independently initialised networks share no
ordering of hidden units, so an average of their weights corresponds to no meaningful
function. For an ensemble, average predicted probabilities and note that the CV
estimate does not describe the ensemble.

---

## Decision points

Source plan's seven, plus three added.

| # | Decision | Default | Revisit if |
|---|---|---|---|
| 1 | Binary level-2 vs 3-class | binary | level-1 count grows via a different label rule |
| 2 | Level 0 → negative? | **no**, exclude | baseline cannot separate 0 from constructed negatives anyway |
| 3 | Window length | 120 s | events mostly < 5 min → shorten to 60 s |
| 4 | Guard band | 30 min | negative pool falls below 100 windows/subject |
| 5 | Same-day rule | on | pool too small; relaxing weakens every negative |
| 6 | Pretrain per fold | (a) if compute allows | otherwise report transductive |
| 7 | HRV features | include with availability flag | **[Δ]** F18 shows availability differs by class within an ACC decile → demote to ablation |
| **[+] 8** | Negative ratio | min achievable across subjects | per-subject ratios recorded as a covariate instead |
| **[+] 9** | Threshold source | inner validation subjects | report per-subject calibration as the deployable variant |
| **[+] 10** | Event boundary trim | 0 min | trimming grid changes conclusions → report the grid, not a point |

---

## Sequencing

Source plan's eight weeks, with insertions marked.

| Week | Work | Insertions |
|---|---|---|
| 1–2 | Phases 1 and 2. Frozen window table, negative pool, falsification checks | **[+]** F01–F08 structural EDA; **[Δ]** achievable-ratio audit before fixing `RATIO`; **[+]** fold structure including inner split |
| 3 | Phase 3.1–3.2 baseline, LOSO, per-fold counts, ablations | **[+]** F13–F18 relational EDA, F16 first; **[+]** majority-class baseline; **[+]** paired signed-rank procedure |
| 4 | Phase 3.3–3.4 nnPU and naive comparator, π grid | **[+]** event-boundary trimming grid |
| 5–7 | Phase 4, only if warranted | **[+]** label-efficiency sweep as the primary deliverable; **[+]** exclude validation subjects from pretraining |
| 8 | Phase 5 severity model and write-up | **[+]** data-perturbation arm and the two-spread figure; **[+]** heterogeneity regression; **[+]** final full-cohort retrain |

Phases 1–3 stand alone. "Curated-negative detection reaches X% event recall at Y
false alarms per hour, and the naive construction inflates this to Z" is a complete,
honest, publishable result whether or not X is impressive.

**[+]** With §5.4 added, a second complete result exists even if X is poor: "the
choice of negative construction moves event recall by as much as resampling the
cohort does." That one does not depend on the detector working.

---

## Task index

| Task | Section | Freezes |
|---|---|---|
| T0.1 | Phase 0 verification | — |
| T1.1–T1.3 | Windowing, label rule, exclusions | — |
| T1.4 | Effective-N table | — |
| T1.5 | Structural EDA F01–F08 | — |
| T1.6 | Window table | `derived/windows.parquet` |
| T2.1 | Eligibility + **[Δ]** ratio audit + F10 | — |
| T2.2 | Activity matching, F11–F12 | — |
| T2.3 | Falsification checks | — |
| T2.4 | Label table | `derived/labels_<seed>.parquet` |
| T3.0 | Fold structure | `derived/folds.json` |
| T3.1 | Baseline + F13–F18 | `reports/floor.json` |
| T3.2–T3.5 | Normalisation, nnPU, naive comparator, paired tests | — |
| T4.1–T4.6 | Pretraining, gate, fine-tune, label-efficiency | `derived/encoder_fold<k>.pt` |
| T5.1–T5.8 | Severity, evaluation, perturbation, heterogeneity, artifact | `reports/` |

Invocation: `Read PLAN.md, execute task T2.1, and write provenance.`
