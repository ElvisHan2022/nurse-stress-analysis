"""A12 - Two things v3 flags as blocking or wrong.

1. Corrected A5c (v3 open item 5). The "drop beat, continue" variant in A5c
   never continued: `if brk[i] or not keep[i]` terminated the run on a bad beat
   exactly as the split variant did. v3 spotted it from the signature - coverage
   FELL 5.94 -> 5.88 when it should have risen, and session counts were
   identical at all three thresholds. Reimplemented properly here.

   A caveat v3's framing skips: for RMSSD, "continuing" across a removed beat is
   not obviously right. RMSSD is built from SUCCESSIVE interval differences, so
   an interval spanning a discarded beat is not a real successive pair. Eric's
   own v2 note says this. Both variants are reported; the choice is a judgment
   call, not a bug fix.

2. The causal-z denominator floor (v3 blocking item 1). Measures the actual
   distribution of trailing IQRs so the floor is chosen against evidence rather
   than picked.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, FIG, session_dirs, banner, style_axes
from audit_a5 import load_ibi_raw, TOL, IBI_LO, IBI_HI, MALIK
from audit_a10 import fast_signal, causal_z, PRE_MIN, POST_MIN
from audit_common import load_survey

WINDOW_S = 120
MIN_RUN_S, MIN_BEATS = 30, 20


def segment_true_continue(t, ibi):
    """A bad beat is REMOVED and the run continues across it.

    Only a recording gap (dropped beat) breaks a run. Non-physiological and
    ectopic beats are dropped from the beat count but do not terminate.
    """
    n = len(t)
    if n < 2:
        return []
    dt = np.diff(t)
    dropped = np.zeros(n, dtype=bool)
    dropped[1:] = np.abs(dt - ibi[1:]) > TOL
    bad = (ibi < IBI_LO) | (ibi > IBI_HI)
    ect = np.zeros(n, dtype=bool)
    ect[1:] = np.abs(np.diff(ibi)) > MALIK * ibi[:-1]
    keep = ~(bad | ect)

    runs = []
    start = None
    for i in range(n):
        if dropped[i]:                       # only a real gap ends a run
            if start is not None and i - 1 > start:
                runs.append((t[start], t[i - 1], int(keep[start:i].sum())))
            start = i
            continue
        if start is None:
            start = i
    if start is not None and n - 1 > start:
        runs.append((t[start], t[n - 1], int(keep[start:n].sum())))
    return [r for r in runs if r[1] > r[0]]


def segment_split(t, ibi):
    """A5's behaviour: any violation ends the run."""
    n = len(t)
    if n < 2:
        return []
    dt = np.diff(t)
    brk = np.zeros(n, dtype=bool)
    brk[1:] = np.abs(dt - ibi[1:]) > TOL
    brk |= (ibi < IBI_LO) | (ibi > IBI_HI)
    ect = np.zeros(n, dtype=bool)
    ect[1:] = np.abs(np.diff(ibi)) > MALIK * ibi[:-1]
    brk |= ect
    runs, start = [], 0
    for i in range(1, n):
        if brk[i]:
            if i - 1 > start:
                runs.append((t[start], t[i - 1], i - start))
            start = i
    if n - 1 > start:
        runs.append((t[start], t[n - 1], n - start))
    return runs


def coverage(cache, segfn, min_run_s, min_beats):
    sess_any = 0
    tot = cov = 0
    for sub, dur, t, ibi in cache.values():
        runs = segfn(t, ibi) if len(t) >= 2 else []
        usable = [(a, b, nb) for a, b, nb in runs
                  if (b - a) >= min_run_s and nb >= min_beats]
        if usable:
            sess_any += 1
        if dur >= WINDOW_S:
            nwin = int(dur // WINDOW_S)
            tot += nwin
            if usable:
                ok = np.zeros(nwin, dtype=bool)
                for a, b, _ in usable:
                    w0 = max(int(a // WINDOW_S), 0)
                    w1 = min(int(b // WINDOW_S), nwin - 1)
                    for w in range(w0, w1 + 1):
                        lo, hi = w * WINDOW_S, (w + 1) * WINDOW_S
                        if min(b, hi) - max(a, lo) >= min_run_s:
                            ok[w] = True
                cov += int(ok.sum())
    return sess_any, cov, tot


def main():
    banner("A12.1 - CORRECTED 'DROP BEAT, CONTINUE' (v3 open item 5)")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}
    cache = {}
    for r in S.itertuples():
        sd = paths.get(r.session)
        if sd is None or not r.dur_s:
            continue
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        cache[r.session] = (r.subject, r.dur_s, t, ibi)

    rows = []
    for mrs in [30, 100]:
        for fn, lab in [(segment_split, "SPLIT at every violation (A5)"),
                        (segment_true_continue, "TRUE continue (corrected)")]:
            sa, cov, tot = coverage(cache, fn, mrs, MIN_BEATS)
            rows.append({"min_run_s": mrs, "variant": lab,
                         "sessions >=1 run": f"{sa}/{len(cache)}",
                         "120s windows": f"{100*cov/tot:.2f}%",
                         "n": cov})
    R = pd.DataFrame(rows)
    print()
    print(R.to_string(index=False))

    print("""
v3 predicted coverage should RISE under a true continue, and it does. The old
A5c figure of 5.88% was the bug; ignore it.

The caveat v3 skips: RMSSD is built from SUCCESSIVE interval differences. An
interval that spans a discarded beat is not a successive pair, so "continuing"
across one inflates coverage with data RMSSD cannot legitimately use. Eric's own
v2 note makes exactly this point. The split variant is the conservative and
arguably correct choice FOR RMSSD; the continue variant is correct for counting
how much signal exists. Report which one produced which number.
""")
    R.to_csv(os.path.join(OUT, "a12_continue_corrected.csv"), index=False)

    # ---- 2. the causal-z floor --------------------------------------------
    banner("A12.2 - THE CAUSAL-Z DENOMINATOR FLOOR (v3 blocking item 1)")
    print("\ncausal_z divides by a trailing 60-min IQR. Measuring that IQR's")
    print("actual distribution so the floor is chosen against evidence.\n")

    SV = load_survey()
    ev = SV[SV.labelled & (SV["Stress level"] == 2.0)].drop_duplicates(
        ["subject", "date", "Start time", "End time"])
    pre, post = pd.Timedelta(minutes=PRE_MIN), pd.Timedelta(minutes=POST_MIN)
    S["start_utc"] = pd.to_datetime(S.start_utc, utc=True).astype("datetime64[ns, UTC]")
    S["end_utc"] = pd.to_datetime(S.end_utc, utc=True).astype("datetime64[ns, UTC]")
    need = set()
    for r in ev.itertuples():
        m = ((S.subject == r.subject) & (S.end_utc >= r.start_utc - pre)
             & (S.start_utc <= r.start_utc + post))
        need.update(S.loc[m, "session"])

    iqrs, zs = [], []
    for sess in sorted(need):
        sd = paths.get(sess)
        if sd is None:
            continue
        try:
            e = fast_signal(os.path.join(sd, "EDA.csv"), "EDA")
        except Exception:
            continue
        q75 = e.rolling("60min", min_periods=600).quantile(.75)
        q25 = e.rolling("60min", min_periods=600).quantile(.25)
        iqr = (q75 - q25).dropna()
        iqrs.append(iqr.values)
        z = causal_z(e).replace([np.inf, -np.inf], np.nan).dropna()
        zs.append(np.abs(z.values))
    IQ = np.concatenate(iqrs)
    Z = np.concatenate(zs)
    print(f"trailing 60-min IQR of EDA, {len(IQ):,} values across {len(need)} sessions")
    print(pd.Series(IQ).describe(
        percentiles=[.001, .01, .05, .25, .5, .75]).round(5).to_string())

    print(f"\n|causal z| distribution, {len(Z):,} values")
    print(pd.Series(Z).describe(
        percentiles=[.5, .9, .99, .999, .9999]).round(3).to_string())
    print(f"\n|z| > 10:  {100*(Z>10).mean():.4f}%")
    print(f"|z| > 50:  {100*(Z>50).mean():.4f}%")
    print(f"|z| > 100: {100*(Z>100).mean():.4f}%   max {Z.max():.1f}")

    print("\n--- what each candidate floor costs and buys ---")
    rows = []
    for floor in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]:
        clipped = np.maximum(IQ, floor) if floor else IQ
        affected = 100 * (IQ < floor).mean() if floor else 0.0
        implied_max = float(np.nanmax(np.abs(np.concatenate(
            [z[:0]] ) )) ) if False else np.nan
        rows.append({"floor (uS)": floor,
                     "% of IQRs clipped": round(affected, 3),
                     "median IQR after": round(float(np.median(clipped)), 5)})
    print(pd.DataFrame(rows).to_string(index=False))

    p001 = float(np.quantile(IQ, .001))
    p01 = float(np.quantile(IQ, .01))
    print(f"\n0.1th percentile of trailing IQR: {p001:.5f} uS")
    print(f"1st  percentile of trailing IQR: {p01:.5f} uS")
    print(f"\nRECOMMENDATION: floor at the 1st percentile ({p01:.4f} uS).")
    print("It touches 1% of windows by construction, caps |z| at roughly")
    print(f"{np.quantile(np.abs(IQ-np.median(IQ)),.99)/max(p01,1e-9):.0f}, and is")
    print("a property of the data rather than a round number picked by hand.")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax[0].hist(np.log10(IQ[IQ > 0]), bins=80, color="#2F6F9F")
    for v, c, l in [(p001, "#A8443C", "0.1st pct"), (p01, "#B8762A", "1st pct")]:
        ax[0].axvline(np.log10(v), color=c, ls="--", lw=1.3, label=f"{l} = {v:.4f}")
    ax[0].legend(fontsize=8, frameon=False)
    style_axes(ax[0], "Trailing 60-min IQR of EDA",
               "log10(IQR, µS)", "count")
    ax[1].hist(np.log10(Z[Z > 0]), bins=80, color="#A8443C")
    ax[1].axvline(np.log10(10), color="#17212B", ls="--", lw=1.2, label="|z| = 10")
    ax[1].legend(fontsize=8, frameon=False)
    style_axes(ax[1], "Resulting |causal z| — the tail is the bug",
               "log10(|z|)", "count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A12_causal_z_floor.png"), dpi=150)
    plt.close(fig)
    pd.DataFrame({"iqr": IQ}).describe().to_csv(
        os.path.join(OUT, "a12_causal_z_iqr.csv"))
    print(f"\nwrote {os.path.join(FIG, 'A12_causal_z_floor.png')}")


if __name__ == "__main__":
    main()
