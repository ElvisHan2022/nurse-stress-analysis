"""A5c - Reconcile the audit's HRV numbers against plan v2.

Plan v2 says HRV is "viable for most subjects" and that v1 was wrong. The audit
(A5) said 5.94% window coverage and recommended demoting it. Both can be true if
they are measuring different things, and they are. Three differences:

  1. DENOMINATOR   v2 counts SESSIONS with >=1 usable run (33/45).
                   A5 counts 120s WINDOWS containing one (5.94%).
  2. ECTOPIC RULE  A5 SPLITS a run at every Malik violation. Standard practice
                   is to drop the offending beat and continue the run. Splitting
                   fragments runs and is far more conservative.
  3. MIN RUN       A5 used 30s. v2 uses min_run=100 as primary.

This isolates each so the disagreement resolves into a number rather than a
disagreement.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, session_dirs, banner
from audit_a5 import load_ibi_raw, TOL, IBI_LO, IBI_HI, MALIK

WINDOW_S = 120


def segment(t, ibi, malik=MALIK, split_on_ectopic=True):
    """Segment into runs of consecutive clean beats.

    split_on_ectopic=True  : a Malik violation ends the run (A5's behaviour)
    split_on_ectopic=False : the offending beat is dropped and the run continues,
                             which is the conventional HRV preprocessing step
    Returns list of (start_s, end_s, n_beats).
    """
    n = len(t)
    if n < 2:
        return []
    dt = np.diff(t)
    dropped = np.zeros(n, dtype=bool)
    dropped[1:] = np.abs(dt - ibi[1:]) > TOL
    nonphys = (ibi < IBI_LO) | (ibi > IBI_HI)
    ect = np.zeros(n, dtype=bool)
    ect[1:] = np.abs(np.diff(ibi)) > malik * ibi[:-1]

    if split_on_ectopic:
        brk = dropped | nonphys | ect
        keep = np.ones(n, dtype=bool)
    else:
        # A gap in the recording still breaks a run; a bad beat is merely removed.
        brk = dropped
        keep = ~(nonphys | ect)

    runs, start = [], None
    for i in range(n):
        if brk[i] or not keep[i]:
            if start is not None and i - 1 > start:
                runs.append((t[start], t[i - 1], int(keep[start:i].sum())))
            start = i if (keep[i] and not brk[i]) else None
            if brk[i] and keep[i]:
                start = i
            continue
        if start is None:
            start = i
    if start is not None and n - 1 > start:
        runs.append((t[start], t[n - 1], int(keep[start:n].sum())))
    return [r for r in runs if r[1] > r[0]]


def main():
    banner("A5c - RECONCILING THE AUDIT WITH PLAN v2")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}

    print("\nparsing IBI once per session, segmenting both ways...")
    cache = {}
    for r in S.itertuples():
        sd = paths.get(r.session)
        if sd is None or not r.dur_s:
            continue
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        cache[r.session] = (
            r.subject, r.dur_s,
            np.array(segment(t, ibi, split_on_ectopic=True)) if len(t) >= 2
            else np.empty((0, 3)),
            np.array(segment(t, ibi, split_on_ectopic=False)) if len(t) >= 2
            else np.empty((0, 3)),
        )
    print(f"sessions parsed: {len(cache)}")

    def stats(split, min_run_s, min_beats=20):
        """Returns session-level and window-level availability."""
        sess_any = tot_sess = 0
        tot_win = cov_win = 0
        per_sub_runs = {}
        for sess, (sub, dur, rs_split, rs_keep) in cache.items():
            arr = rs_split if split else rs_keep
            tot_sess += 1
            usable = arr[(arr[:, 1] - arr[:, 0] >= min_run_s) &
                         (arr[:, 2] >= min_beats)] if len(arr) else np.empty((0, 3))
            per_sub_runs.setdefault(sub, []).append(len(usable))
            if len(usable):
                sess_any += 1
            if dur >= WINDOW_S:
                nwin = int(dur // WINDOW_S)
                tot_win += nwin
                if len(usable):
                    ok = np.zeros(nwin, dtype=bool)
                    for a, b, _ in usable:
                        w0, w1 = max(int(a // WINDOW_S), 0), min(int(b // WINDOW_S), nwin - 1)
                        for w in range(w0, w1 + 1):
                            lo, hi = w * WINDOW_S, (w + 1) * WINDOW_S
                            if min(b, hi) - max(a, lo) >= min_run_s:
                                ok[w] = True
                    cov_win += int(ok.sum())
        med = {k: float(np.median(v)) for k, v in per_sub_runs.items()}
        return sess_any, tot_sess, cov_win, tot_win, med

    # ---- 1. does the ectopic rule explain the gap? -------------------------
    banner("Difference 2 - splitting runs at every ectopic beat vs dropping the beat")
    rows = []
    for min_run in [30, 100, 300]:
        for split, lab in [(True, "SPLIT (A5)"), (False, "drop beat, continue")]:
            sa, ts, cw, tw, _ = stats(split, min_run)
            rows.append({"min_run_s": min_run, "ectopic handling": lab,
                         "sessions >=1 run": f"{sa}/{ts} ({100*sa/ts:.0f}%)",
                         "120s windows covered": f"{100*cw/tw:.2f}%"})
    R = pd.DataFrame(rows)
    print()
    print(R.to_string(index=False))

    print("\nSplitting at every ectopic is the conservative choice and it is what")
    print("A5 did. Conventional HRV preprocessing removes the beat and continues.")
    print("The gap between the two rows at each min_run is the cost of that choice.")

    # ---- 2. the denominator, side by side ----------------------------------
    banner("Difference 1 - session-level vs window-level, same criteria")
    for min_run in [30, 100]:
        sa, ts, cw, tw, med = stats(False, min_run)
        print(f"\nmin_run={min_run}s, drop-beat-and-continue:")
        print(f"  sessions with >=1 usable run   {sa}/{ts} = {100*sa/ts:.1f}%   "
              f"<- the number v2 quotes")
        print(f"  120s windows with usable HRV   {cw:,}/{tw:,} = {100*cw/tw:.2f}%   "
              f"<- the number that governs a per-window feature")
        print(f"  ratio                          {(100*sa/ts)/(100*cw/tw):.1f}x more flattering")

    # ---- 3. per-subject, which is what "most subjects" means ---------------
    banner("Difference 3 - is HRV viable for MOST subjects, or a minority?")
    for min_run in [100]:
        _, _, _, _, med = stats(False, min_run)
        rows = []
        for sess, (sub, dur, _, rs) in cache.items():
            pass
        per = []
        for sub in sorted(med):
            sess_list = [(d, r) for s, (sb, d, _, r) in cache.items() if sb == sub]
            tw = cw = 0
            for dur, arr in sess_list:
                if dur < WINDOW_S:
                    continue
                nwin = int(dur // WINDOW_S)
                tw += nwin
                usable = arr[(arr[:, 1] - arr[:, 0] >= min_run) & (arr[:, 2] >= 20)] \
                    if len(arr) else np.empty((0, 3))
                if not len(usable):
                    continue
                ok = np.zeros(nwin, dtype=bool)
                for a, b, _ in usable:
                    w0, w1 = max(int(a // WINDOW_S), 0), min(int(b // WINDOW_S), nwin - 1)
                    for w in range(w0, w1 + 1):
                        lo, hi = w * WINDOW_S, (w + 1) * WINDOW_S
                        if min(b, hi) - max(a, lo) >= min_run:
                            ok[w] = True
                cw += int(ok.sum())
            per.append({"subject": sub, "median usable runs/session": med[sub],
                        "windows": tw, "covered": cw,
                        "window coverage": f"{100*cw/tw:.2f}%" if tw else "n/a"})
        P = pd.DataFrame(per).sort_values("covered", ascending=False)
        print(f"\nmin_run={min_run}s, drop-beat-and-continue, per subject:")
        print(P.to_string(index=False))
        n_ok = int((P.covered >= 100).sum())
        print(f"\nsubjects with >=100 HRV-covered windows: {n_ok} of {len(P)}")

    R.to_csv(os.path.join(OUT, "a5c_reconciliation.csv"), index=False)
    print(f"\nwrote {os.path.join(OUT, 'a5c_reconciliation.csv')}")


if __name__ == "__main__":
    main()
