"""PHASE 2 - Negative construction. Match, do not filter.

The idea in one sentence: there is no "not stressed" label in this dataset, so
we build a comparison group out of unlabelled time - and we have to build it so
that it differs from the positive class ONLY in stress, not in anything else a
model could latch onto.

2.1 Eligibility  - which unlabelled windows are even candidates
2.2 Matching     - draw from those candidates so the activity distribution
                   MIRRORS the positives, within each subject
2.4 Deliverable  - frozen label table, one per seed

Why matching and not filtering: nurses move more when busy, and busy is
stressful. If we picked calm-looking low-motion windows as negatives, the model
would separate them from positives perfectly by detecting movement, score
beautifully, and be a pedometer.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, LOCAL_TZ, load_survey, banner

WINDOW_S = 120
MIN_SESSION_MIN = 30
GUARD_MIN = 30
RATIO = 1.31                 # min achievable under BIN matching (audit A11 + phase2 ceiling
                             # analysis). 2.14 was computed on totals and left a
                             # 1.64x spread across folds; 1.31 halves it to 1.28x.
                             # Exact uniformity needs r~0.54 and is not usable.
N_BINS = 10
SEEDS = [0, 1, 2]


def main():
    banner("PHASE 2 - NEGATIVE CONSTRUCTION")

    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True).astype("datetime64[ns, UTC]")
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True).astype("datetime64[ns, UTC]")
    print(f"\nfrozen Phase 1 table: {len(W):,} windows, "
          f"{W.subject.nunique()} subjects, {int(W.label.sum()):,} positive")

    # ---- 2.1 eligibility ---------------------------------------------------
    banner("2.1 Eligibility - four conditions, all required")

    SV = load_survey()
    SV = SV[~SV.is_exact_dup]
    SV["date_local"] = SV.start_utc.dt.tz_convert(LOCAL_TZ).dt.date
    W["date_local"] = W.start_utc.dt.tz_convert(LOCAL_TZ).dt.date

    # (1) session long enough
    c1 = W.sess_min >= MIN_SESSION_MIN

    # (2) on a subject-day the nurse actually reported something.
    #     This is the rule that makes silence informative: on a day she was
    #     filling in the survey, NOT reporting is evidence.
    ev_days = set(zip(SV.subject, SV.date_local))
    c2 = pd.Series([(s, d) in ev_days for s, d in zip(W.subject, W.date_local)],
                   index=W.index)

    # (3) >=30 min clear of ANY reported event, including the 113 unrated ones.
    #     Unrated events are still events; we just don't know their severity.
    g = pd.Timedelta(minutes=GUARD_MIN)
    guard = np.zeros(len(W), dtype=bool)
    for sub, sg in SV.groupby("subject"):
        m = (W.subject == sub).values
        idx = np.where(m)[0]
        if not len(idx):
            continue
        ws, we = W.start_utc.values[idx], W.end_utc.values[idx]
        for e in sg.itertuples():
            lo = np.datetime64((e.start_utc - g).tz_localize(None))
            hi = np.datetime64((e.end_utc + g).tz_localize(None))
            guard[idx] |= (we > lo) & (ws < hi)
    c3 = ~pd.Series(guard, index=W.index)

    # (4) not a positive window
    c4 = W.label == 0

    W["eligible_neg"] = c1 & c2 & c3 & c4
    for n, c in [("session >= 30 min", c1), ("on a day with a report", c2),
                 (">= 30 min from any event", c3), ("not itself positive", c4)]:
        print(f"  {n:28} passes {int(c.sum()):>7,} / {len(W):,}")
    n_elig = int(W.eligible_neg.sum())
    print(f"\n  ALL FOUR                     {n_elig:>7,}  "
          f"= {n_elig*WINDOW_S/3600:,.1f} hours")
    print(f"  positives                    {int(W.label.sum()):>7,}")
    print(f"  pool ratio available         {n_elig/max(int(W.label.sum()),1):>7.2f} : 1")

    # ---- 2.2 activity matching --------------------------------------------
    banner("2.2 Activity matching - mirror the positives, within subject")
    POS = W[W.label == 1]
    CAND = W[W.eligible_neg]

    # Bin edges come from the POSITIVE distribution, so the bins are defined by
    # the class we are trying to mirror.
    edges = POS.acc_mean.quantile(np.linspace(0, 1, N_BINS + 1)).values.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    W["abin"] = pd.cut(W.acc_mean, edges, labels=False)
    POS = W[W.label == 1]
    CAND = W[W.eligible_neg]

    print(f"\n{len(edges)-1} activity bins from the positive-class quantiles")
    print(f"positives {len(POS):,}  candidates {len(CAND):,}\n")

    summary = []
    for seed in SEEDS:
        picks = []
        shortfall = {}
        for sub, gp in POS.groupby("subject"):
            want = (gp.abin.value_counts() * RATIO).round().astype(int)
            pool = CAND[CAND.subject == sub]
            got = 0
            for b, n in want.items():
                avail = pool[pool.abin == b]
                take = min(int(n), len(avail))
                if take:
                    picks.append(avail.sample(take, random_state=seed))
                got += take
            shortfall[sub] = (int(want.sum()), got)
        NEG = pd.concat(picks) if picks else CAND.iloc[:0]
        lab = pd.concat([
            POS.assign(label=1),
            NEG.assign(label=0),
        ])[["subject", "session", "w", "start_utc", "label", "abin"]]
        lab["seed"] = seed
        lab = lab.rename(columns={"start_utc": "window_start"})
        p = os.path.join(DERIVED, f"labels_seed{seed}.parquet")
        lab.to_parquet(p, index=False)

        per = lab.groupby(["subject", "label"]).size().unstack(fill_value=0)
        per.columns = ["neg", "pos"]
        per["ratio"] = (per.neg / per.pos.replace(0, np.nan)).round(2)
        summary.append(per.assign(seed=seed))
        if seed == SEEDS[0]:
            print("--- seed 0, per subject ---")
            print(per.to_string())
            print("\nrequested vs obtained (shortfall means the pool ran dry):")
            for s, (wnt, got) in sorted(shortfall.items()):
                flag = "" if got >= wnt else f"  <- SHORT by {wnt-got}"
                print(f"  {s:4} wanted {wnt:5}  got {got:5}{flag}")
        print(f"\nseed {seed}: {len(lab):,} rows  "
              f"({int((lab.label==1).sum()):,} pos / {int((lab.label==0).sum()):,} neg)")

    # ---- seed variance: the noise floor -----------------------------------
    banner("Seed-to-seed variance - this is your noise floor")
    ALL = pd.concat(summary).reset_index()
    v = ALL.groupby("subject").agg(neg_min=("neg", "min"), neg_max=("neg", "max"),
                                   ratio_mean=("ratio", "mean"))
    v["neg_spread"] = v.neg_max - v.neg_min
    print()
    print(v.round(2).to_string())
    print("\nAny model improvement smaller than the spread this induces is not")
    print("an improvement. Compute it once the baseline exists and quote it")
    print("beside every comparison.")

    banner("FROZEN - Phase 2.4 deliverable")
    for seed in SEEDS:
        p = os.path.join(DERIVED, f"labels_seed{seed}.parquet")
        d = pd.read_parquet(p)
        print(f"  {os.path.basename(p)}  {len(d):,} rows  "
              f"cols {list(d.columns)}")


if __name__ == "__main__":
    main()
