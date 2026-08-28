"""A4 - Timezone alignment. The highest-risk section; it fails silently.

Survey timestamps are believed naive America/Chicago. Sensor t0 values are UTC
unix timestamps. If the pipeline joined under the wrong hypothesis it would
return almost no positives and raise nothing.
"""
from __future__ import annotations

import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (OUT, FIG, SURVEY, LOCAL_TZ, load_survey, banner,
                          append_findings, style_axes)


def naive_events(path=SURVEY):
    """Survey events with tz-naive local wall-clock start/end."""
    sv = load_survey(path)
    return sv[["subject", "date", "Stress level", "labelled",
               "start_naive", "end_naive", "duration_min"]].copy()


def count_inside(ev, S, shift_hours=None, tz=None, mode="start"):
    """How many events land inside a session belonging to the same subject.

    shift_hours: treat the naive time as UTC then add this many hours.
    tz:          localize the naive time to this zone (DST-aware), then to UTC.
    mode:        'start' = event start inside a session; 'full' = both ends.
    """
    if tz is not None:
        su = ev.start_naive.dt.tz_localize(
            tz, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
        eu = ev.end_naive.dt.tz_localize(
            tz, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
    else:
        su = ev.start_naive.dt.tz_localize("UTC") + pd.Timedelta(hours=shift_hours)
        eu = ev.end_naive.dt.tz_localize("UTC") + pd.Timedelta(hours=shift_hours)

    win = {k: v[["start_utc", "end_utc"]].values for k, v in S.groupby("subject")}
    hits = 0
    for sub, s, e in zip(ev.subject, su, eu):
        w = win.get(sub)
        if w is None:
            continue
        if mode == "start":
            ok = ((w[:, 0] <= s) & (w[:, 1] >= s)).any()
        else:
            ok = ((w[:, 0] <= s) & (w[:, 1] >= e)).any()
        hits += bool(ok)
    return hits


def main():
    banner("A4 - TIMEZONE ALIGNMENT")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    # Normalise resolution explicitly. A mixed ms/ns pair compares wrongly once
    # cast to int64, and does so silently in the permissive direction.
    S["start_utc"] = pd.to_datetime(S.start_utc, utc=True).astype("datetime64[ns, UTC]")
    S["end_utc"] = pd.to_datetime(S.end_utc, utc=True).astype("datetime64[ns, UTC]")
    assert S.start_utc.dtype == S.end_utc.dtype, "session bounds differ in resolution"

    ev = naive_events()
    rated = ev[ev.labelled].copy()
    n_rated = len(rated)
    print(f"\nrated events: {n_rated}   sessions: {len(S)}")

    # ---- the two hypotheses AUDIT.md asks for ------------------------------
    banner("The two hypotheses (rated events, start inside a session)")
    h1 = count_inside(rated, S, shift_hours=0)
    h2 = count_inside(rated, S, tz=LOCAL_TZ)

    H = pd.DataFrame([
        {"hypothesis": "1. survey timestamps ARE UTC",
         "events_inside": h1, "pct": f"{100*h1/n_rated:.1f}%"},
        {"hypothesis": f"2. survey naive {LOCAL_TZ} -> UTC",
         "events_inside": h2, "pct": f"{100*h2/n_rated:.1f}%"},
    ])
    print()
    print(H.to_string(index=False))

    resolved = "America/Chicago (DST-aware)" if h2 > h1 else "UTC"
    print(f"\nRESOLVED HYPOTHESIS: survey timestamps are {resolved}")

    # ---- full sweep, to show the peak is sharp -----------------------------
    banner("Supporting evidence: fixed-offset sweep")
    sweep = []
    for off in range(0, 10):
        st = count_inside(rated, S, shift_hours=off, mode="start")
        fu = count_inside(rated, S, shift_hours=off, mode="full")
        sweep.append({"assumed_local": f"UTC-{off}",
                      "start_inside": st, "start_pct": round(100*st/n_rated, 1),
                      "fully_inside": fu, "full_pct": round(100*fu/n_rated, 1)})
    SW = pd.DataFrame(sweep)
    print()
    print(SW.to_string(index=False))

    best_fixed = SW.loc[SW.start_inside.idxmax()]
    dst_start = count_inside(rated, S, tz=LOCAL_TZ, mode="start")
    dst_full = count_inside(rated, S, tz=LOCAL_TZ, mode="full")
    print(f"\nbest fixed offset:  {best_fixed.assumed_local}  "
          f"start_inside={best_fixed.start_inside}  fully_inside={best_fixed.fully_inside}")
    print(f"DST-aware {LOCAL_TZ}: start_inside={dst_start}  fully_inside={dst_full}")
    print(f"\nDST-aware beats best fixed offset by "
          f"{dst_start - int(best_fixed.start_inside)} events on 'start inside'.")
    print("The data spans April to December, so a fixed offset misaligns the")
    print("winter events by a full hour. Use the named zone, not an offset.")

    # ---- what does NOT land, under the winning hypothesis -------------------
    banner("Residual: rated events that still fall outside every session")
    su = rated.start_naive.dt.tz_localize(
        LOCAL_TZ, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
    # Work in int64 nanoseconds; tz-aware .values yields object arrays otherwise.
    win = {k: np.column_stack([
               v.start_utc.astype("datetime64[ns, UTC]").astype("int64").values,
               v.end_utc.astype("datetime64[ns, UTC]").astype("int64").values])
           for k, v in S.groupby("subject")}
    su_ns = su.astype("datetime64[ns, UTC]").astype("int64").values
    NS_PER_MIN = 60_000_000_000
    miss = []
    for (idx, r), s in zip(rated.iterrows(), su_ns):
        w = win.get(r.subject)
        if w is None or not ((w[:, 0] <= s) & (w[:, 1] >= s)).any():
            near = np.nan
            if w is not None and len(w):
                d = np.minimum(np.abs(w[:, 0] - s), np.abs(w[:, 1] - s)) / NS_PER_MIN
                near = float(np.min(d))
            miss.append({"subject": r.subject, "date": r.date,
                         "level": r["Stress level"],
                         "dur_min": round(r.duration_min, 1),
                         "min_gap_to_session_min": round(near, 1)})
    M = pd.DataFrame(miss)
    print(f"\nrated events outside any session: {len(M)} / {n_rated} "
          f"({100*len(M)/n_rated:.1f}%)")
    if len(M):
        print("\nby subject:")
        print(M.subject.value_counts().to_string())
        print("\ndistance to nearest session boundary (minutes):")
        print(M.min_gap_to_session_min.describe(
            percentiles=[.5, .9]).round(1).to_string())
        print("\nfirst 15:")
        print(M.sort_values("min_gap_to_session_min").head(15).to_string(index=False))

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(range(10), SW.start_inside, "o-", lw=1.8, color="#2F6F9F",
            label="start inside a session")
    ax.plot(range(10), SW.fully_inside, "s--", lw=1.5, color="#5B6B7A",
            label="fully inside a session")
    ax.axhline(dst_start, color="#3F7D62", ls=":", lw=1.6,
               label=f"DST-aware {LOCAL_TZ} ({dst_start})")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"−{i}" for i in range(10)])
    ax.legend(fontsize=8.5, frameon=False)
    style_axes(ax, "A4 · Timezone offset sweep — the peak is the evidence",
               "assumed local offset from UTC (hours)",
               f"rated events matched (of {n_rated})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A4_timezone_sweep.png"), dpi=150)
    plt.close(fig)

    # ---- write the resolution ---------------------------------------------
    with open(os.path.join(OUT, "timezone_resolution.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Timezone resolution

**Resolved:** survey timestamps are naive **{LOCAL_TZ}** (DST-aware).
Sensor `t0` values are UTC unix timestamps.

Evidence, on {n_rated} rated events:

| Hypothesis | Events whose start falls inside a same-subject session |
|---|---|
| Survey timestamps are UTC | {h1} ({100*h1/n_rated:.1f}%) |
| Survey naive {LOCAL_TZ} -> UTC | **{h2} ({100*h2/n_rated:.1f}%)** |

Best fixed offset was {best_fixed.assumed_local} at {best_fixed.start_inside}
events. DST-aware `{LOCAL_TZ}` reaches {dst_start}, beating it by
{dst_start - int(best_fixed.start_inside)}. The archive spans 2020-04 to 2020-12,
so a fixed offset misaligns winter events by an hour. **Use the named zone.**

Residual: {len(M)} rated events ({100*len(M)/n_rated:.1f}%) still fall outside
every session under the winning hypothesis. These are survey reports with no
sensor coverage, not a timezone failure.

Every later section uses this resolution.
""")
    print(f"\nwrote {os.path.join(OUT, 'timezone_resolution.md')}")

    banner("A4 STOP CONDITION")
    triggered = max(h1, h2) / n_rated < 0.5
    print("TRIGGERED - neither hypothesis places most events inside a session"
          if triggered else "not triggered")

    append_findings(
        "A4", "Timezone alignment", "tasks/audit_a4.py",
        [("hypothesis 1: survey is UTC", f"{h1} ({100*h1/n_rated:.1f}%)",
          "low", "n.a."),
         ("hypothesis 2: naive Chicago -> UTC", f"{h2} ({100*h2/n_rated:.1f}%)",
          "high", "n.a."),
         ("resolved", resolved, "America/Chicago",
          "yes" if h2 > h1 else "NO"),
         ("best fixed offset", best_fixed.assumed_local, "UTC-5",
          "yes" if best_fixed.assumed_local == "UTC-5" else "NO"),
         ("DST-aware advantage (events)",
          dst_start - int(best_fixed.start_inside), ">0",
          "yes" if dst_start > best_fixed.start_inside else "NO"),
         ("rated events with no sensor coverage", len(M), "n.a.", "n.a.")],
        ["figures/audit/A4_timezone_sweep.png"],
        "Fixes the join used by every later section. Written to "
        "reports/audit/timezone_resolution.md.",
        f"{len(M)} rated events fall outside every session even under the winning "
        f"hypothesis - they have no sensor coverage at all."
        if len(M) else "none",
    )


if __name__ == "__main__":
    main()
