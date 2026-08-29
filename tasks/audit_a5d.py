"""A5d - Re-run HRV coverage using Eric's exact segmentation, in BEATS.

Two things were being compared that are not comparable:

  audit A5   MIN_RUN_S = 30   -> 30 SECONDS
  Eric       min_run   = 30   -> 30 BEATS  (his comment: "min_run=30 beats is
                                LENIENT -- conventional short-term HRV uses
                                ~5 min (~300 beats)")

At the observed median interval of ~0.75 s, 30 beats is ~22 s and 120 beats is
~90 s. So "120" in his notebook is not 120 seconds.

His validity rule also differs from the audit's:
  - intervals derived as np.diff(timestamps), not read from the IBI column
  - iv_range (0.4, 1.5) s, i.e. 40-150 bpm, vs the audit's (0.33, 1.50)
  - run-level plausibility gates on mean HR and RMSSD, which the audit lacks

This reimplements his rule exactly and reports what the audit reports:
coverage of the 120 s modelling window over all 609 sessions.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, session_dirs, banner
from audit_a5 import load_ibi_raw

WINDOW_S = 120
IV_RANGE = (0.4, 1.5)
MALIK = 0.20
HR_RANGE = (45, 130)
RMSSD_RANGE = (5, 150)


def eric_runs(t, min_run_beats):
    """Eric's ibi_usability segmentation. Returns list of (start_s, end_s, n_beats)
    for runs passing his plausibility gates."""
    if len(t) < 3:
        return []
    iv = np.diff(t)
    ok = (iv >= IV_RANGE[0]) & (iv <= IV_RANGE[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(np.diff(iv)) / iv[:-1]
    ok[1:] &= rel <= MALIK
    idx = np.where(ok)[0]
    if idx.size == 0:
        return []
    out = []
    for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if len(r) < min_run_beats:
            continue
        v = iv[r] * 1000.0
        hr = 60000.0 / v.mean()
        rmssd = float(np.sqrt(np.mean(np.diff(v) ** 2))) if len(v) > 1 else np.nan
        if not (HR_RANGE[0] <= hr <= HR_RANGE[1]):
            continue
        if not (RMSSD_RANGE[0] <= rmssd <= RMSSD_RANGE[1]):
            continue
        out.append((t[r[0]], t[r[-1] + 1], len(r)))
    return out


def main():
    banner("A5d - ERIC'S SEGMENTATION, AUDIT'S DENOMINATOR")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}

    print("\nparsing IBI for all 609 sessions...")
    cache = {}
    for r in S.itertuples():
        sd = paths.get(r.session)
        if sd is None or not r.dur_s:
            continue
        t, _ = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        cache[r.session] = (r.subject, r.dur_s, t)
    print(f"sessions: {len(cache)}")

    # observed interval, to convert beats <-> seconds honestly
    all_iv = []
    for _, _, t in cache.values():
        if len(t) > 2:
            iv = np.diff(t)
            all_iv.append(iv[(iv >= IV_RANGE[0]) & (iv <= IV_RANGE[1])])
    IV = np.concatenate(all_iv) if all_iv else np.array([0.8])
    med_iv = float(np.median(IV))
    print(f"median valid interval: {med_iv:.3f} s  ->  1 beat ~ {med_iv:.2f} s")

    banner("Unit conversion: what each min_run actually means")
    conv = pd.DataFrame([{"min_run (beats)": b,
                          "approx seconds": f"{b*med_iv:.0f}s",
                          "audit equivalent": f"MIN_RUN_S={b*med_iv:.0f}"}
                         for b in [30, 100, 120, 300]])
    print()
    print(conv.to_string(index=False))
    print(f"\nEric's commit says '120s run'. At {med_iv:.2f}s per beat, min_run=120")
    print(f"beats is ~{120*med_iv:.0f}s, not 120s. Worth confirming which he meant.")

    # ---- the comparison ----------------------------------------------------
    banner("Coverage under Eric's rule, at both denominators")
    rows = []
    for mr in [30, 100, 120, 300]:
        best_sess_subjects = set()
        sess_any = 0
        tot_win = cov_win = 0
        per_sub = {}
        for sess, (sub, dur, t) in cache.items():
            runs = eric_runs(t, mr)
            if runs:
                sess_any += 1
                best_sess_subjects.add(sub)
            if dur >= WINDOW_S:
                nwin = int(dur // WINDOW_S)
                tot_win += nwin
                d = per_sub.setdefault(sub, [0, 0])
                d[1] += nwin
                if runs:
                    ok = np.zeros(nwin, dtype=bool)
                    for a, b, _ in runs:
                        w0 = max(int(a // WINDOW_S), 0)
                        w1 = min(int(b // WINDOW_S), nwin - 1)
                        for w in range(w0, w1 + 1):
                            lo, hi = w * WINDOW_S, (w + 1) * WINDOW_S
                            if min(b, hi) - max(a, lo) >= 30:   # >=30s inside the window
                                ok[w] = True
                    cov_win += int(ok.sum())
                    d[0] += int(ok.sum())
        rows.append({
            "min_run (beats)": mr,
            "~seconds": f"{mr*med_iv:.0f}s",
            "subjects w/ run in BEST session": f"{len(best_sess_subjects)}/15",
            "sessions w/ >=1 run": f"{sess_any}/{len(cache)} ({100*sess_any/len(cache):.0f}%)",
            "120s WINDOWS covered": f"{100*cov_win/tot_win:.2f}%",
            "windows n": cov_win,
        })
        if mr in (30, 120):
            P = pd.DataFrame([{"subject": s, "covered": v[0], "windows": v[1],
                               "pct": round(100*v[0]/v[1], 2) if v[1] else 0.0}
                              for s, v in per_sub.items()]).sort_values(
                "covered", ascending=False)
            print(f"\n--- per subject at min_run={mr} beats (~{mr*med_iv:.0f}s) ---")
            print(P.to_string(index=False))
            print(f"subjects with >=100 covered windows: {int((P.covered>=100).sum())}/15")

    R = pd.DataFrame(rows)
    print()
    print(R.drop(columns=["windows n"]).to_string(index=False))

    banner("READ THIS ROW BY ROW")
    print("""
The 'BEST session' column is the statistic plan v2 quotes. The 'WINDOWS' column
is the one that governs a per-window feature. They diverge by one to two orders
of magnitude, and the divergence grows with min_run because a longer run
requirement is satisfiable in a subject's single best session long after it has
stopped being satisfiable in a typical window.

Both numbers are correct. Only one of them describes the modelling substrate.
""")
    R.to_csv(os.path.join(OUT, "a5d_eric_rule_coverage.csv"), index=False)
    print(f"wrote {os.path.join(OUT, 'a5d_eric_rule_coverage.csv')}")


if __name__ == "__main__":
    main()
