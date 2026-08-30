"""PHASE 1 - Units, labels, cohort. Produces the frozen window table.

Applies PLAN_v3 section 1.3's exclusion rules IN THE DECLARED ORDER, printing
the attrition at every step so the final count is traceable. Order matters:
v3 section 1.3 records that 13 sessions change status depending on whether
session length is measured before or after non-wear removal. We measure BEFORE,
and that is a recorded judgment call.

Deliverable: derived/windows.parquet
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, LOCAL_TZ, load_survey, banner

WINDOW_S = 120
MIN_SESSION_MIN = 5
EDA_FLOOR_US = 0.05
EDA_FLOOR_PCT = 50
DROP_NO_NEGATIVES = ["6D"]                    # rule 8
DROP_FLAT_EDA = ["7E", "DF", "EG", "CE"]      # rule 9
POSITIVE_OVERLAP = 0.50                       # label rule, section 1.2


def main():
    banner("PHASE 1 - UNITS, LABELS, COHORT")

    W = pd.read_parquet(os.path.join(OUT, "a6_windows.parquet"))
    E = pd.read_parquet(os.path.join(OUT, "a9_eda_stats.parquet"))
    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    W = W.merge(E, on=["session", "w"], how="left")
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True).astype("datetime64[ns, UTC]")
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True).astype("datetime64[ns, UTC]")
    W["sess_min"] = W.session.map(S.set_index("session").dur_s) / 60

    # non-wear and dead-EDA flags, needed by rules 2 and 3
    W["nonwear"] = (W.eda_med < EDA_FLOOR_US) & (W.acc_sd < 0.005)
    sess_q = W.groupby("session").agg(med=("eda_med", "median"),
                                      floor=("eda_floor", "mean"))
    dead = set(sess_q[(sess_q.med < EDA_FLOOR_US) |
                      (sess_q.floor > EDA_FLOOR_PCT / 100)].index)

    print(f"\nstarting windows: {len(W):,}   subjects: {W.subject.nunique()}")

    # ---- the exclusion ladder ---------------------------------------------
    banner("Exclusions, in v3 section 1.3 order")
    alive = pd.Series(True, index=W.index)
    rows = []

    def step(n, name, mask):
        nonlocal alive
        before = int(alive.sum())
        removed = int((alive & mask).sum())
        alive = alive & ~mask
        rows.append({"#": n, "rule": name, "entering": before,
                     "removed": removed, "surviving": int(alive.sum())})

    step(1, f"sessions < {MIN_SESSION_MIN} min (measured BEFORE non-wear)",
         W.sess_min < MIN_SESSION_MIN)
    step(2, "non-wear windows", W.nonwear.fillna(False))
    step(3, "dead-EDA sessions", W.session.isin(dead))
    step(8, f"subject 6D (0 eligible negatives)", W.subject.isin(DROP_NO_NEGATIVES))
    step(9, f"flat-EDA subjects {', '.join(DROP_FLAT_EDA)}",
         W.subject.isin(DROP_FLAT_EDA))

    A = pd.DataFrame(rows)
    print()
    print(A.to_string(index=False))
    print(f"\nrules 4-7 and 10 act on the SURVEY, not on windows - see below.")

    W = W[alive].copy()

    # ---- survey side: rules 4, 5, 6, 7, 10 --------------------------------
    banner("Survey-side exclusions")
    SV = load_survey()
    srows = [("all survey rows", len(SV))]
    SV = SV[~SV.is_exact_dup]                                   # rule 5
    srows.append(("after dropping 3 exact duplicates", len(SV)))
    SV_rated = SV[SV.labelled]                                  # rule 4
    srows.append(("after dropping 113 unrated", len(SV_rated)))
    ev2 = SV_rated[SV_rated["Stress level"] == 2.0]             # rule 7
    srows.append(("after excluding levels 0 and 1", len(ev2)))
    ev2 = ev2[~ev2.subject.isin(DROP_NO_NEGATIVES + DROP_FLAT_EDA)]
    srows.append(("after the cohort cut (rules 8, 9)", len(ev2)))
    for n, v in srows:
        print(f"  {n:42} {v:>4}")

    # rule 6: overlapping pairs -> keep the outer span
    ev2 = ev2.sort_values(["subject", "start_utc"]).reset_index(drop=True)
    merged, cur = [], None
    for r in ev2.itertuples():
        if cur and r.subject == cur["subject"] and r.start_utc <= cur["end_utc"]:
            cur["end_utc"] = max(cur["end_utc"], r.end_utc)     # outer span
            continue
        if cur:
            merged.append(cur)
        cur = {"subject": r.subject, "start_utc": r.start_utc, "end_utc": r.end_utc}
    if cur:
        merged.append(cur)
    EV = pd.DataFrame(merged)
    print(f"  {'after merging overlapping pairs (rule 6)':42} {len(EV):>4}")
    print(f"\nFINAL level-2 events: {len(EV)}   (v3 predicts 156)")

    # ---- rule 10: events with no sensor coverage --------------------------
    covered = []
    for r in EV.itertuples():
        m = ((W.subject == r.subject) & (W.end_utc > r.start_utc)
             & (W.start_utc < r.end_utc))
        covered.append(bool(m.any()))
    EV["has_coverage"] = covered
    n_uncov = int((~EV.has_coverage).sum())
    print(f"rule 10: events with no surviving sensor coverage: {n_uncov} "
          f"(set aside, tolerance = 0)")
    EV = EV[EV.has_coverage].reset_index(drop=True)
    print(f"events carried into Phase 2: {len(EV)}")

    # ---- label rule (section 1.2) -----------------------------------------
    banner("Label rule: positive if >=50% of the window is inside a level-2 event")
    W["label"] = 0
    for r in EV.itertuples():
        m = (W.subject == r.subject).values
        idx = np.where(m)[0]
        ov = ((np.minimum(W.end_utc.values[idx], np.datetime64(r.end_utc.tz_localize(None)))
               - np.maximum(W.start_utc.values[idx],
                            np.datetime64(r.start_utc.tz_localize(None))))
              / np.timedelta64(1, "s"))
        W.iloc[idx, W.columns.get_loc("label")] = np.where(
            ov >= WINDOW_S * POSITIVE_OVERLAP, 1, W.label.values[idx])

    print(f"\npositive windows: {int(W.label.sum()):,} "
          f"({100*W.label.mean():.2f}% of {len(W):,})")
    print("\nper subject:")
    per = W.groupby("subject").agg(windows=("label", "size"),
                                   positives=("label", "sum"))
    per["events"] = EV.groupby("subject").size().reindex(per.index).fillna(0).astype(int)
    print(per.to_string())

    # ---- freeze ------------------------------------------------------------
    keep = ["subject", "session", "w", "start_utc", "end_utc", "label",
            "acc_mean", "acc_sd", "eda_med", "eda_floor",
            "hrv_available", "hrv_run_s", "in_any_event", "sess_min"]
    OUTP = os.path.join(DERIVED, "windows.parquet")
    W[keep].to_parquet(OUTP, index=False)
    EV.to_parquet(os.path.join(DERIVED, "events.parquet"), index=False)
    A.to_csv(os.path.join(DERIVED, "phase1_attrition.csv"), index=False)

    banner("FROZEN")
    print(f"\n  {OUTP}")
    print(f"    {len(W):,} windows | {W.subject.nunique()} subjects | "
          f"{int(W.label.sum()):,} positive")
    print(f"  {os.path.join(DERIVED, 'events.parquet')}")
    print(f"    {len(EV)} level-2 events")
    print("\nDo not rewrite these. Later phases read them.")


if __name__ == "__main__":
    main()
