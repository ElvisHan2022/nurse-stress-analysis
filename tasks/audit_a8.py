"""A8 - Effective sample size and the negative pool.

Computes the achievable negative-to-positive ratio PER SUBJECT, which must be
known before RATIO is fixed. An aggregate ratio that looks comfortable can
conceal subjects that cannot supply any comparison group at all.
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
from audit_common import (OUT, FIG, LOCAL_TZ, load_survey, banner,
                          append_findings, style_axes)

GUARD_MIN = 30
WINDOW_S = 120
MIN_SESSION_MIN = 30
REQUESTED_RATIO = 3


def main():
    banner("A8 - EFFECTIVE SAMPLE SIZE AND THE NEGATIVE POOL")

    W = pd.read_parquet(os.path.join(OUT, "a6_windows.parquet"))
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True).astype("datetime64[ns, UTC]")
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True).astype("datetime64[ns, UTC]")
    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    SV = load_survey().drop_duplicates(["subject", "date", "Start time", "End time"])
    ev2 = SV[SV.labelled & (SV["Stress level"] == 2.0)]

    # ---- the three counts --------------------------------------------------
    banner("The three sample sizes")
    n_win = len(W)
    n_pos_win = int(W.is_level2.sum())
    n_ev = len(ev2)
    n_sub = W.subject.nunique()
    print(f"\n  windows            {n_win:>8,}   <- a compute statistic")
    print(f"  level-2 windows    {n_pos_win:>8,}")
    print(f"  level-2 EVENTS     {n_ev:>8,}   <- honest denominator for "
          f"'can we detect an episode'")
    print(f"  subjects           {n_sub:>8,}   <- honest denominator for "
          f"'does this work on a new nurse'")
    print(f"\n  windows per event  {n_pos_win/max(n_ev,1):>8.1f}")
    print(f"  rows-to-subject    {n_win/max(n_sub,1):>8,.0f}   "
          f"(the guide flags >100 as a problem)")

    # ---- eligibility -------------------------------------------------------
    banner("Negative eligibility (same-day rule + guard band)")
    print(f"\n  1. window in a session >= {MIN_SESSION_MIN} min")
    print(f"  2. on a subject-day carrying at least one reported event")
    print(f"  3. >= {GUARD_MIN} min from the boundary of ANY event, rated or not")
    print("  (non-wear is handled in A9's attrition sequence)\n")

    dur = S.set_index("session").dur_s
    W["sess_min"] = W.session.map(dur) / 60
    W["date_local"] = W.start_utc.dt.tz_convert(LOCAL_TZ).dt.date

    # subject-days carrying any reported event
    SV["date_local"] = SV.start_utc.dt.tz_convert(LOCAL_TZ).dt.date
    ev_days = set(zip(SV.subject, SV.date_local))
    W["day_has_event"] = [(s, d) in ev_days for s, d in zip(W.subject, W.date_local)]

    # guard band around every reported event
    guard = np.zeros(len(W), dtype=bool)
    g = pd.Timedelta(minutes=GUARD_MIN)
    for sub, sg in SV.groupby("subject"):
        m = (W.subject == sub).values
        idx = np.where(m)[0]
        ws = W.start_utc.values[idx]
        we = W.end_utc.values[idx]
        for e in sg.itertuples():
            lo = np.datetime64((e.start_utc - g).tz_convert("UTC").tz_localize(None))
            hi = np.datetime64((e.end_utc + g).tz_convert("UTC").tz_localize(None))
            guard[idx] |= (we > lo) & (ws < hi)
    W["in_guard"] = guard

    W["eligible_neg"] = (~W.is_level2 & ~W.in_guard & W.day_has_event
                         & (W.sess_min >= MIN_SESSION_MIN))

    print(f"windows total                       {len(W):>8,}")
    print(f"  level-2 (positive)                {int(W.is_level2.sum()):>8,}")
    print(f"  inside guard band                 {int(W.in_guard.sum()):>8,}")
    print(f"  on a day with no reported event   {int((~W.day_has_event).sum()):>8,}")
    print(f"  in a session < {MIN_SESSION_MIN} min             "
          f"{int((W.sess_min < MIN_SESSION_MIN).sum()):>8,}")
    print(f"  ELIGIBLE NEGATIVES                {int(W.eligible_neg.sum()):>8,}")
    print(f"\nhours: {len(W)*WINDOW_S/3600:,.0f} total -> "
          f"{int(W.eligible_neg.sum())*WINDOW_S/3600:,.0f} eligible negative")

    # ---- per-subject achievable ratio --------------------------------------
    banner("Achievable negative-to-positive ratio, per subject")
    T = (W.groupby("subject")
           .agg(pos_win=("is_level2", "sum"),
                neg_win=("eligible_neg", "sum"),
                total_win=("is_level2", "size")))
    T["events"] = ev2.groupby("subject").size().reindex(T.index).fillna(0).astype(int)
    T["pos_min"] = T.pos_win * WINDOW_S / 60
    T["neg_min"] = T.neg_win * WINDOW_S / 60
    T["ratio"] = (T.neg_win / T.pos_win.replace(0, np.nan)).round(2)
    T = T.sort_values("ratio")
    print()
    print(T[["events", "pos_win", "neg_win", "pos_min", "neg_min", "ratio"]]
          .to_string())

    agg = T.neg_win.sum() / max(T.pos_win.sum(), 1)
    print(f"\nAGGREGATE ratio: {agg:.2f} : 1")
    valid = T[T.pos_win > 0]
    below = valid[valid.ratio < REQUESTED_RATIO]
    below1 = valid[valid.ratio < 1]
    print(f"subjects below the requested {REQUESTED_RATIO}:1 -> {len(below)} "
          f"of {len(valid)}   ({', '.join(below.index)})")
    print(f"subjects below 1:1           -> {len(below1)}"
          + (f"   ({', '.join(below1.index)})" if len(below1) else ""))
    print(f"\nMINIMUM achievable across subjects with positives: "
          f"{valid.ratio.min():.2f} : 1")
    zero_pos = T[T.pos_win == 0]
    if len(zero_pos):
        print(f"subjects with NO level-2 windows: {', '.join(zero_pos.index)}")
    zero_neg = valid[valid.neg_win == 0]
    if len(zero_neg):
        print(f"subjects with NO eligible negatives: {', '.join(zero_neg.index)}")

    print("\nThe aggregate figure is not the binding constraint. Class balance")
    print("would vary by an order of magnitude across folds, and since balance")
    print("moves the decision threshold and threshold transfer is already")
    print("compromised by the A7 propensity spread, the two failures compound.")

    T.to_csv(os.path.join(OUT, "a8_achievable_ratio.csv"))

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    v = valid.sort_values("ratio")
    cols = ["#A8443C" if r < REQUESTED_RATIO else "#2F6F9F" for r in v.ratio]
    ax.barh(v.index, v.ratio, color=cols)
    ax.axvline(REQUESTED_RATIO, color="#17212B", ls="--", lw=1.3,
               label=f"requested {REQUESTED_RATIO}:1")
    ax.axvline(1, color="#B8762A", ls=":", lw=1.2, label="1:1")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style_axes(ax, "A8 · Achievable negative-to-positive ratio per subject",
               "eligible negative windows per positive window", "subject")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A8_achievable_ratio.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.scatter(T.events, T.pos_min, s=54, color="#2F6F9F", zorder=3)
    for s, r in T.iterrows():
        ax.annotate(s, (r.events, r.pos_min), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    for d in [5, 15, 30, 60]:
        xs = np.array([0, max(T.events.max(), 1)])
        ax.plot(xs, xs * d, lw=.7, ls=":", color="#C3CBD2", zorder=1)
        ax.annotate(f"{d} min/event", (xs[1], xs[1] * d), fontsize=7,
                    color="#5B6B7A", ha="right", va="bottom")
    ax.set_ylim(0, T.pos_min.max() * 1.15)
    style_axes(ax, "A8 · Events against positive minutes, per subject",
               "level-2 events", "positive minutes")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A8_events_vs_minutes.png"), dpi=150)
    plt.close(fig)

    W.to_parquet(os.path.join(OUT, "a8_windows_eligible.parquet"))

    append_findings(
        "A8", "Effective sample size and negative pool", "tasks/audit_a8.py",
        [("windows", f"{n_win:,}", "~4,869 (post-exclusion)", "n.a."),
         ("level-2 events", n_ev, 178, "yes" if n_ev == 178 else "NO"),
         ("subjects", n_sub, 14, "15 pre-exclusion"),
         ("aggregate ratio", f"{agg:.2f}:1", "3.6:1",
          "yes" if abs(agg - 3.6) < .8 else "NO"),
         ("subjects below 3:1", f"{len(below)} of {len(valid)}", "5 of 14",
          "yes" if len(below) == 5 else "NO"),
         ("minimum achievable ratio", f"{valid.ratio.min():.2f}:1", "0.9:1",
          "n.a.")],
        ["figures/audit/A8_achievable_ratio.png",
         "figures/audit/A8_events_vs_minutes.png"],
        "Determines the negative ratio setting (JC21) and must be run before it "
        "is fixed. Feeds decision point 8.",
        f"aggregate {agg:.2f}:1 conceals {len(below)} subjects below "
        f"{REQUESTED_RATIO}:1 and {len(below1)} below 1:1",
    )


if __name__ == "__main__":
    main()
