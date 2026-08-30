# AUDIT.md

Executable specification for the initial data audit. Run this before any
modeling and before `PLAN.md` Phase 1 freezes anything.

**Invocation:** `Read AUDIT.md and run the full audit.`
Or one section at a time: `Read AUDIT.md and run section A4.`

---

## Setup

```
ARCHIVE = <path to unzipped Stress_dataset>
SURVEY  = <path to SurveyResults.xlsx>
OUT     = reports/audit/
FIG     = figures/audit/
```

Create `OUT` and `FIG`. Everything here is **read-only** with respect to the
archive. Write nothing back into it.

Dependencies: `pandas`, `numpy`, `matplotlib`, `openpyxl`, `scikit-learn`,
`pyarrow`. Install what is missing before starting.

---

## How to run this

Work through the sections in order. Sections build on each other and a failure in
an early one invalidates the later ones.

For each section:

1. Write a script at `tasks/audit_<section>.py`. One section per file. Do not put
   the whole audit in one script.
2. Run it and **print the actual output**, not a summary of it. I want to see
   `df.head()`, `df.dtypes`, `value_counts()`, and shapes. Truncate long tables to
   the first 15 rows rather than describing them in prose.
3. Save any figure to `FIG` as `<section>_<slug>.png`, 150 dpi, with axis labels
   and units.
4. Append your findings to `OUT/findings.md` using the template at the bottom of
   this file.
5. If a section's **stop condition** is met, stop and tell me. Do not continue and
   do not work around it.

Do not tune any threshold in this document to make a check pass. If a check fails,
that is the finding.

---

## A1 · Inventory and headers

**Question.** Do I have what I think I have, and at what rates?

Walk the archive. Count subject folders and session directories. For each session
record which signal CSVs are present.

Every sampled E4 file has this header structure: row 1 is the session start as a
UTC unix timestamp, row 2 is the sampling rate in Hz, data begins at row 3.
`IBI.csv` and `tags.csv` do not follow it.

**Read the sampling rate from row 2. Do not use any documented value.** Published
descriptions of this dataset give blood volume pulse at 72 Hz and temperature at
10 Hz, which contradicts the device specification of 64 Hz and 4 Hz. The file
header settles it for this extraction.

**Print:**
- subject list and count
- session directory count
- fraction of sessions containing each of `ACC BVP EDA HR TEMP IBI tags`
- observed distinct sampling rates per signal
- spread of `t0` across signals within a session (should be exactly zero)

**Stop condition.** More than one distinct rate for any signal, or nonzero `t0`
spread. Either means the archive mixes device configurations or the extraction is
broken.

---

## A2 · Session table and duration

Build one row per session: `subject, session, t0, rate, n_samples, dur_s,
start_utc, end_utc`. Compute duration as `n_samples / rate`, never from metadata.

**Print:** `head(10)`, `dtypes`, total sensor hours, date range, and
`dur_min.describe()` with the 10th, 25th, 50th, 75th and 90th percentiles.

**Figures:**
- `A2_wear_raster.png` — one row per subject, calendar time horizontal, a filled
  bar per session, coloured by duration. Shows collection periods and who
  contributed when.
- `A2_duration_ecdf.png` — empirical CDF of session length on a log x-axis, with
  vertical lines at 2, 5 and 10 minutes. ECDF rather than histogram so the fraction
  below a cutoff is readable directly.

**Expected values, from the reference notebook.** 609 sessions, 1,255 h, median
1.17 h, 92 sessions under 5 minutes.

**Stop condition.** Any of those four differs by more than a few percent. The
extraction then differs from the reference and every established fact needs
rechecking.

---

## A3 · Survey audit

Load the workbook. Print `columns`, `head(10)`, `dtypes`, and row count before and
after dropping exact duplicates.

Classify the stress level column into rated and unrated. Print the full
`value_counts()` including the unrated category.

For rated events compute duration in minutes and print `describe()` grouped by
level. Count events over 60 minutes and report the maximum.

Detect overlapping events within subject: sort by start, flag any event beginning
before the previous one ends. Print every overlapping pair.

**Figure:** `A3_duration_ecdf_by_level.png` — one ECDF curve per level, vertical
line at 60 min. This figure is the response to a published objection that marked
sections in this dataset exceeded plausible event duration, so it needs to be
publication quality.

**Expected values.** 358 rows, 3 exact duplicates, 113 unrated, 46 low / 20 medium
/ 179 high, 34 events over 60 min with a 323 min maximum, 12 overlapping pairs.

---

## A4 · Timezone alignment

**This is the highest-risk section and it fails silently.**

The survey timestamps are believed naive `America/Chicago`; sensor `t0` values are
UTC unix timestamps.

Compute, under two hypotheses:
1. Survey timestamps interpreted as UTC
2. Survey timestamps localized to `America/Chicago` then converted to UTC

For each, count how many rated events have a start falling inside a session
belonging to the same subject. Print both counts side by side as a fraction of all
rated events.

**Interpretation.** The correct hypothesis is the one where nearly all events land
inside a session. A low count is not a puzzle, it is the signature of an unhandled
offset. If the pipeline had joined under the wrong hypothesis it would have
returned almost no positives and raised nothing.

**Print** the resolved hypothesis explicitly and write it to
`OUT/timezone_resolution.md`. Every later section uses it.

**Stop condition.** Neither hypothesis places most events inside a session. Then
something other than timezone is wrong with the join.

---

## A5 · IBI structure and HRV feasibility

`IBI.csv` is the only event-indexed file. Row 1 is `<session start unix>, IBI`.
Each later row is `<elapsed seconds since start>, <interval duration in seconds>`.
There is no sampling rate.

**First, verify the time convention.** The elapsed column conventionally marks the
second beat of each pair, so for consecutive accepted beats `t[i] - t[i-1]` should
equal `ibi[i]`. Compute the fraction of rows satisfying this within 50 ms, across a
sample of at least 60 sessions. Print it.

A high fraction confirms the convention and confirms that a violation indicates a
dropped beat rather than an indexing error. **Report the number rather than
assuming it.** An off-by-one here silently corrupts every HRV feature.

**Then segment into usable runs.** A break occurs when any of these holds:
- `abs((t[i] - t[i-1]) - ibi[i]) > 0.05` seconds — a beat was dropped
- `ibi[i]` outside 0.33 to 1.50 seconds — non-physiological, 40 to 180 bpm
- `abs(ibi[i] - ibi[i-1]) > 0.20 * ibi[i-1]` — Malik criterion, ectopic beat

Rules 2 and 3 remove a beat and therefore also split the run.

A run is usable if it spans at least 30 seconds **and** contains at least 20 beats.
Both conditions, not either.

**Print:** runs per session, usable runs per session, longest run duration, and the
distribution of the fraction of session time inside a usable run.

**Then compute the number that actually matters.** Tile each session into 120 s
windows and report the fraction of windows containing a usable run of at least
30 s. The existing figure of 75 of 105 sessions having at least one usable run is
session-level and flatters the situation, because a session with three short runs
still leaves most of its windows uncovered.

**Figures:**
- `A5_hrv_window_coverage.png` — histogram of per-window coverage
- `A5_poincare.png` — successive intervals plotted against one another for six
  sampled runs. A healthy plot is a comet along the diagonal. Scatter off-diagonal
  means the segmentation is admitting artifacts

**Decision this feeds.** If window-level coverage is low, HRV is demoted from the
baseline feature set to an ablation. Do not decide here; report the number.

---

## A6 · Missingness, and whether it is informative

Build a per-window table with an `hrv_available` indicator and `hrv_run_s`.

**Print:**
- overall missingness rate per feature
- missingness rate per subject, sorted
- for HRV specifically, availability broken down by accelerometer decile

**Then run the decisive test.** Fit a classifier whose only inputs are the
missingness indicators and run length, predicting whether a window falls inside a
level-2 event, under leave-one-subject-out. Report the area under the ROC curve.

**Interpretation.** Anything meaningfully above 0.5 means missingness carries label
information. The mechanism is physical: beat rejection is motion-driven, motion
tracks being on the ward, and events happen on the ward.

Note explicitly in the findings that LightGBM and XGBoost route missing values
natively, so a tree can split on missingness whether or not an indicator column is
supplied. A policy of never imputing does not close this by itself.

**Figure:** `A6_hrv_availability_by_activity.png` — grouped bars, accelerometer
decile on x, availability rate on y, one series per class.

---

## A7 · Labeling propensity and the SCAR assumption

Standard positive-unlabeled methods require the probability of a positive being
labeled to be constant across the data. Test it.

**Print:** per-subject probability that a session carries a label, sorted, and the
ratio between the largest and smallest nonzero value.

**Then fit a propensity model:** predict whether a session carries a label, from
subject, session duration and start hour. Report the area under the ROC curve.

**Expected values.** AUC around 0.728, per-subject probability spanning 0.125 to
0.724, a ratio near 5.8.

**Figures:**
- `A7_propensity_by_subject.png` — sorted bars with the AUC annotated
- `A7_time_budget_by_subject.png` — one stacked bar per subject over their sensor
  time, segmented into non-wear, level-2, other labeled, eligible unlabeled,
  ineligible unlabeled

**Interpretation to write down.** Labeling propensity varying by subject means the
assumption fails along exactly the axis the evaluation split uses. Any pooled
decision threshold will not transfer between folds.

---

## A8 · Effective sample size and the negative pool

Compute three counts and print them together: windows, level-2 events, subjects.
State plainly that the honest denominator for a claim about a new nurse is the
subject count, and for a claim about detecting an episode is the event count.

Build the per-subject table: events, positive minutes, eligible negative minutes
under the same-day rule and a 30 minute guard band, and the **achievable
negative-to-positive ratio**.

**Print the table sorted by ratio ascending.**

**Figures:**
- `A8_achievable_ratio.png` — sorted horizontal bars per subject with a vertical
  line at the requested ratio of 3
- `A8_events_vs_minutes.png` — scatter, one point per subject, event count against
  positive minutes, with iso-duration reference lines

**What to look for.** An aggregate ratio near 3.6:1 that conceals several subjects
below 3:1 and at least one below 1:1. The scatter should show that the
scarce-negative subjects are the same ones with few long events, because a long
event plus a guard band consumes the day.

**This determines the negative ratio setting and must be run before it is fixed.**

---

## A9 · Attrition, and whether rule order matters

Build the attrition table for the Phase 1 exclusion sequence: drop sessions under
5 min; drop non-wear seconds; drop dead-EDA sessions; drop unrated events; drop
duplicate survey rows; resolve overlaps; exclude levels 0 and 1; drop subject 6D.

**Print** one row per rule with hours entering, hours removed, hours surviving.

**Then rerun with rules 1 and 2 swapped** and print the final count both ways.

**Interpretation.** If the totals differ, exclusion order is itself a judgment call
and must be recorded and perturbed like any other. Report the difference whether or
not it is large.

---

## A10 · Does anything actually happen at event onset?

**Run this even if earlier sections are incomplete. It is the cheapest way to learn
whether the project is well posed.**

Align every level-2 event at its onset. For each of electrodermal activity and
heart rate, extract the causally normalized signal from 30 minutes before to 60
minutes after onset, average across events, and plot with a standard error band and
a vertical line at zero.

Produce two versions: pooled across all subjects, and faceted by subject for the
six subjects with the most events.

**Figure:** `A10_event_triggered_average.png`

**Interpretation.**
- A visible rise near onset means the labels mark a physiologically distinguishable
  state and the modeling problem is well posed. The shape also indicates where the
  informative portion of the interval sits, which is the empirical input to the
  boundary trimming decision.
- A flat trace means the labels do not mark a distinguishable state at this
  resolution, and no modeling repairs that.

**Stop condition.** A flat pooled trace with flat per-subject facets. Report it and
stop. That result reframes the project and is worth knowing in week one rather than
month three.

---

## Findings template

Append one block per section to `OUT/findings.md`:

```markdown
### <Section ID> · <name>

**Ran:** <script path> at <timestamp>

**Numbers**
| Quantity | Observed | Expected | Match |
|---|---|---|---|
| ... | ... | ... | yes / no / n.a. |

**Figures:** <paths>

**What this changes.** <which judgment call or plan decision this feeds, and in
which direction. If nothing, say "nothing".>

**Surprises.** <anything that did not match expectation, stated plainly. If none,
say "none".>
```

---

## Closing summary

After all sections, write `OUT/audit_summary.md` containing:

1. A table of every expected value against every observed value, with a match column.
2. The list of stop conditions triggered, if any.
3. The three sample sizes: windows, events, subjects.
4. The resolved timezone hypothesis.
5. Window-level HRV coverage, and a recommendation on whether HRV enters the
   baseline feature set.
6. The achievable negative ratio table and a recommended ratio setting.
7. Whether the event-triggered average shows a response at onset.
8. Anything encountered that is not covered by an existing judgment call and
   therefore needs adding to the registry.

Item 8 matters most. A decision made during the audit and not recorded is exactly
the kind of thing this whole apparatus exists to prevent.
