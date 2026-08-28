"""A2 - Session table, duration, and the person-day census.

Duration is computed as n_samples / rate from an actual row count, never from
metadata. The person-day census is an addition to AUDIT.md: subject-days are the
natural unit for the same-day negative-eligibility rule in PLAN.md Phase 2.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (OUT, FIG, LOCAL_TZ, read_e4_header, count_data_rows,
                          session_dirs, banner, show, append_findings, style_axes)


def build_session_table():
    rows = []
    t_start = time.time()
    dirs = session_dirs()
    for i, sd in enumerate(dirs):
        if i and i % 150 == 0:
            print(f"    ... {i}/{len(dirs)} sessions ({time.time()-t_start:.0f}s)")
        eda = os.path.join(sd, "EDA.csv")
        t0, fs = read_e4_header(eda)
        n = count_data_rows(eda)
        rows.append({
            "subject": os.path.basename(os.path.dirname(sd)),
            "session": os.path.basename(sd),
            "path": sd,
            "t0": t0,
            "rate": fs,
            "n_samples": n,
            "dur_s": n / fs if fs else np.nan,
        })
    S = pd.DataFrame(rows)
    # Force both to nanosecond resolution. pd.to_datetime(unit='s') yields ms
    # resolution while adding a timedelta promotes to ns; casting a mixed pair
    # to int64 later then compares milliseconds against nanoseconds and every
    # lower-bound test silently passes.
    S["start_utc"] = pd.to_datetime(S.t0, unit="s", utc=True
                                    ).astype("datetime64[ns, UTC]")
    S["end_utc"] = (S.start_utc + pd.to_timedelta(S.dur_s, unit="s")
                    ).astype("datetime64[ns, UTC]")
    S["start_local"] = S.start_utc.dt.tz_convert(LOCAL_TZ)
    S["end_local"] = S.end_utc.dt.tz_convert(LOCAL_TZ)
    S["dur_min"] = S.dur_s / 60
    return S.sort_values(["subject", "start_utc"]).reset_index(drop=True)


def main():
    banner("A2 - SESSION TABLE AND DURATION")
    print("\nbuilding session table (counting rows in 609 EDA.csv files)...")
    S = build_session_table()
    S.drop(columns=["path"]).to_parquet(os.path.join(OUT, "a2_sessions.parquet"))

    show("sessions.head(10)", S[["subject", "session", "t0", "rate",
                                 "n_samples", "dur_s", "start_utc", "end_utc"]], 10)

    print("\n--- dtypes ---")
    print(S[["subject", "session", "t0", "rate", "n_samples", "dur_s",
             "start_utc", "end_utc"]].dtypes.to_string())

    total_h = S.dur_s.sum() / 3600
    print(f"\ntotal sensor hours: {total_h:,.1f}")
    print(f"date range (UTC):   {S.start_utc.min()}  ->  {S.end_utc.max()}")
    print(f"date range (local): {S.start_local.min()}  ->  {S.end_local.max()}")

    print("\n--- dur_min.describe() with requested percentiles ---")
    desc = S.dur_min.describe(percentiles=[.10, .25, .50, .75, .90])
    print(desc.round(3).to_string())

    med_h = S.dur_s.median() / 3600
    n_under5 = int((S.dur_min < 5).sum())
    print(f"\nmedian session: {med_h:.3f} h")
    print(f"sessions under 5 min: {n_under5}")
    print(f"sessions under 2 min: {int((S.dur_min < 2).sum())}")
    print(f"sessions under 10 min: {int((S.dur_min < 10).sum())}")

    # ---- gaps and overlaps -------------------------------------------------
    S["gap_s"] = S.groupby("subject", group_keys=False).apply(
        lambda g: (g.start_utc.shift(-1) - g.end_utc).dt.total_seconds(),
        include_groups=False).values
    g = S.gap_s.dropna()
    print(f"\ngap to next session: median {g.median():.0f}s | "
          f"<60s {100*(g<60).mean():.0f}% | <5min {100*(g<300).mean():.0f}% | "
          f"overlapping (negative) {int((g<0).sum())}")

    # ---- PERSON-DAY CENSUS -------------------------------------------------
    banner("A2b - PERSON-DAY CENSUS")
    print("\nA person-day is one (subject, local calendar date) with any sensor time.")
    print("This is the unit the same-day negative-eligibility rule operates on.\n")

    S["date_local"] = S.start_local.dt.date
    # A session crossing local midnight contributes to both days; count by start.
    S["crosses_midnight"] = S.start_local.dt.date != S.end_local.dt.date

    PD = (S.groupby(["subject", "date_local"])
            .agg(sessions=("dur_s", "size"),
                 hours=("dur_s", lambda x: x.sum() / 3600),
                 first_start=("start_local", "min"),
                 last_end=("end_local", "max"))
            .reset_index())
    PD["span_h"] = (PD.last_end - PD.first_start).dt.total_seconds() / 3600

    print(f"total person-days with any sensor data: {len(PD)}")
    print(f"sessions crossing local midnight:       {int(S.crosses_midnight.sum())}")

    print("\n--- person-days per subject ---")
    per_subj = (PD.groupby("subject")
                  .agg(person_days=("date_local", "size"),
                       total_h=("hours", "sum"),
                       median_h_per_day=("hours", "median"),
                       max_h_per_day=("hours", "max"))
                  .sort_values("person_days", ascending=False).round(2))
    per_subj.loc["TOTAL"] = [per_subj.person_days.sum(), per_subj.total_h.sum(),
                             np.nan, np.nan]
    print(per_subj.to_string())

    print("\n--- hours per person-day, distribution ---")
    print(PD.hours.describe(percentiles=[.10, .25, .50, .75, .90]).round(3).to_string())

    print("\n--- person-days by sensor-hours bracket ---")
    br = pd.cut(PD.hours, [0, .5, 1, 2, 4, 8, 24],
                labels=["<0.5h", "0.5-1h", "1-2h", "2-4h", "4-8h", "8h+"])
    print(br.value_counts().sort_index().to_string())

    PD.to_parquet(os.path.join(OUT, "a2_person_days.parquet"))

    # ---- figures -----------------------------------------------------------
    subjects = sorted(S.subject.unique())
    ymap = {s: i for i, s in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(13, 5.2))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=min(S.dur_min.max(), 480))
    cmap = plt.get_cmap("viridis")
    for r in S.itertuples():
        ax.barh(ymap[r.subject],
                width=(r.end_local - r.start_local).total_seconds() / 86400,
                left=matplotlib.dates.date2num(r.start_local.tz_localize(None)),
                height=.72, color=cmap(norm(min(r.dur_min, 480))))
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.xaxis_date()
    fig.autofmt_xdate()
    style_axes(ax, "A2 · Wear raster — one bar per session, coloured by duration",
               "calendar date (America/Chicago)", "subject")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="session duration (min, capped 480)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A2_wear_raster.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.sort(S.dur_min.values)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=1.6, color="#2F6F9F")
    ax.set_xscale("log")
    for cut, col in [(2, "#A8443C"), (5, "#B8762A"), (10, "#3F7D62")]:
        frac = (S.dur_min < cut).mean()
        ax.axvline(cut, color=col, ls="--", lw=1)
        ax.annotate(f"{cut} min\n{frac:.1%} below", (cut, .06),
                    fontsize=8, color=col, ha="left",
                    xytext=(4, 0), textcoords="offset points")
    style_axes(ax, "A2 · Session duration ECDF",
               "session duration (minutes, log scale)",
               "fraction of sessions at or below")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A2_duration_ecdf.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(per_subj.drop(index="TOTAL").index,
           per_subj.drop(index="TOTAL").person_days, color="#2F6F9F")
    style_axes(ax, "A2b · Person-days per subject", "subject", "person-days")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A2b_person_days.png"), dpi=150)
    plt.close(fig)

    # ---- stop condition ----------------------------------------------------
    exp = {"sessions": 609, "hours": 1255, "median_h": 1.17, "under5": 92}
    obs = {"sessions": len(S), "hours": total_h, "median_h": med_h,
           "under5": n_under5}
    banner("A2 STOP CONDITION (>few % deviation from reference)")
    stop = []
    for k in exp:
        dev = abs(obs[k] - exp[k]) / exp[k] * 100
        flag = "OK" if dev <= 3 else "DEVIATES"
        if dev > 3:
            stop.append(f"{k}: observed {obs[k]:.4g} vs expected {exp[k]:.4g} ({dev:.1f}%)")
        print(f"  {k:10} observed {obs[k]:>10.4g}  expected {exp[k]:>8.4g}  "
              f"dev {dev:5.1f}%  {flag}")
    print("\nTRIGGERED" if stop else "\nnot triggered")
    for s in stop:
        print("  -", s)

    append_findings(
        "A2", "Session table, duration, person-days",
        "tasks/audit_a2.py",
        [("sessions", len(S), 609, "yes" if len(S) == 609 else "no"),
         ("sensor hours", f"{total_h:,.1f}", "1,255",
          "yes" if abs(total_h - 1255) / 1255 < .03 else "no"),
         ("median session (h)", f"{med_h:.3f}", "1.17",
          "yes" if abs(med_h - 1.17) / 1.17 < .03 else "no"),
         ("sessions < 5 min", n_under5, 92, "yes" if n_under5 == 92 else "no"),
         ("person-days", len(PD), "n.a. (new)", "n.a."),
         ("sessions crossing local midnight", int(S.crosses_midnight.sum()),
          "n.a. (new)", "n.a.")],
        ["figures/audit/A2_wear_raster.png",
         "figures/audit/A2_duration_ecdf.png",
         "figures/audit/A2b_person_days.png"],
        "Fixes the session inventory and the effective denominator. The person-day "
        "count bounds how much the same-day eligibility rule (JC18) can supply.",
        "; ".join(stop) if stop else "none",
    )


if __name__ == "__main__":
    main()
