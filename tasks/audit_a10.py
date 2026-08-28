"""A10 - Does anything actually happen at event onset?

The cheapest way to learn whether the project is well posed. Align every level-2
event at onset, average the causally normalised signal across events, and look.

A visible rise means the labels mark a physiologically distinguishable state.
A flat trace means they do not at this resolution, and no modelling repairs that.
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
from audit_common import (OUT, FIG, LOCAL_TZ, read_e4_header, load_survey,
                          session_dirs, banner, append_findings, style_axes)

PRE_MIN, POST_MIN = 30, 60
CH = ["eda", "hr"]


def fast_signal(path, name):
    """Faster than np.loadtxt for these files. UTC-indexed 1 Hz mean."""
    t0, fs = read_e4_header(path)
    if name == "ACC":
        v = pd.read_csv(path, skiprows=2, header=None).values / 64.0
        mag = np.sqrt((v ** 2).sum(axis=1))
        s = pd.Series(mag, index=pd.to_datetime(
            t0 + np.arange(len(mag)) / fs, unit="s", utc=True))
    else:
        v = pd.read_csv(path, skiprows=2, header=None).iloc[:, 0].values
        s = pd.Series(v, index=pd.to_datetime(
            t0 + np.arange(len(v)) / fs, unit="s", utc=True))
    return s.resample("1s").mean()


def causal_z(s, window="60min", min_periods=600):
    """Trailing robust z. Uses only the past, so no leakage and no
    contamination of the baseline by the event being measured."""
    med = s.rolling(window, min_periods=min_periods).median()
    q75 = s.rolling(window, min_periods=min_periods).quantile(.75)
    q25 = s.rolling(window, min_periods=min_periods).quantile(.25)
    return (s - med) / (q75 - q25).replace(0, np.nan)


def main():
    banner("A10 - EVENT-TRIGGERED AVERAGE")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    S["start_utc"] = pd.to_datetime(S.start_utc, utc=True).astype("datetime64[ns, UTC]")
    S["end_utc"] = pd.to_datetime(S.end_utc, utc=True).astype("datetime64[ns, UTC]")
    paths = {os.path.basename(d): d for d in session_dirs()}
    S["path"] = S.session.map(paths)

    SV = load_survey()
    ev = SV[SV.labelled & (SV["Stress level"] == 2.0)].copy()
    ev = ev.drop_duplicates(["subject", "date", "Start time", "End time"])
    print(f"\nlevel-2 events (deduplicated): {len(ev)}")

    pre = pd.Timedelta(minutes=PRE_MIN)
    post = pd.Timedelta(minutes=POST_MIN)

    # Which sessions do we need? Any overlapping an event's [-30, +60] window.
    need = set()
    for r in ev.itertuples():
        lo, hi = r.start_utc - pre, r.start_utc + post
        m = ((S.subject == r.subject) & (S.end_utc >= lo) & (S.start_utc <= hi))
        need.update(S.loc[m, "session"])
    print(f"sessions needed: {len(need)} of {len(S)}")

    # ---- load and normalise -----------------------------------------------
    print("\nloading EDA + HR, resampling to 1 Hz, causal-z within session...")
    t0 = time.time()
    frames = {}
    for i, sess in enumerate(sorted(need)):
        if i and i % 40 == 0:
            print(f"    ... {i}/{len(need)} ({time.time()-t0:.0f}s)")
        sd = paths.get(sess)
        if sd is None:
            continue
        try:
            d = pd.DataFrame({
                "eda": fast_signal(os.path.join(sd, "EDA.csv"), "EDA"),
                "hr": fast_signal(os.path.join(sd, "HR.csv"), "HR"),
            })
        except Exception as e:
            print(f"    [warn] {sess}: {e}")
            continue
        for c in CH:
            d[c + "_z"] = causal_z(d[c])
        frames[sess] = d
    print(f"loaded {len(frames)} sessions in {time.time()-t0:.0f}s")

    # ---- extract aligned traces -------------------------------------------
    grid = np.arange(-PRE_MIN * 60, POST_MIN * 60 + 1)
    traces = {c: [] for c in CH}
    meta = []
    for r in ev.itertuples():
        lo, hi = r.start_utc - pre, r.start_utc + post
        m = ((S.subject == r.subject) & (S.end_utc >= lo) & (S.start_utc <= hi))
        parts = [frames[s] for s in S.loc[m, "session"] if s in frames]
        if not parts:
            continue
        d = pd.concat(parts).sort_index()
        d = d[~d.index.duplicated()]
        w = d.loc[lo:hi]
        if w.empty:
            continue
        off = ((w.index - r.start_utc).total_seconds()).astype(int)
        got = {}
        for c in CH:
            ser = pd.Series(w[c + "_z"].values, index=off).reindex(grid)
            got[c] = ser.values
        # Require some pre-onset baseline, else the alignment is meaningless.
        if np.isfinite(got["eda"][:PRE_MIN * 60]).sum() < 300:
            continue
        for c in CH:
            traces[c].append(got[c])
        meta.append({"subject": r.subject, "event": r.Index,
                     "dur_min": r.duration_min})
    M = pd.DataFrame(meta)
    n_ev = len(M)
    print(f"\nevents with usable aligned coverage: {n_ev} / {len(ev)}")
    if n_ev == 0:
        print("no usable events; cannot run A10")
        return
    print("\nevents contributing, by subject:")
    print(M.subject.value_counts().to_string())

    A = {c: np.vstack(traces[c]) for c in CH}

    # ---- pooled statistics -------------------------------------------------
    banner("Pooled event-triggered average (causal z units)")
    stats = {}
    for c in CH:
        arr = A[c]
        n = np.isfinite(arr).sum(axis=0)
        mean = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
        stats[c] = (mean, se, n)
        base = np.nanmean(mean[(grid >= -30 * 60) & (grid < 0)])
        p0_10 = np.nanmean(mean[(grid >= 0) & (grid < 10 * 60)])
        p0_30 = np.nanmean(mean[(grid >= 0) & (grid < 30 * 60)])
        peak_i = int(np.nanargmax(mean))
        # Paired shift per event: post-onset 0-10 min minus pre-onset baseline.
        per_ev_base = np.nanmean(arr[:, (grid >= -30 * 60) & (grid < 0)], axis=1)
        per_ev_post = np.nanmean(arr[:, (grid >= 0) & (grid < 10 * 60)], axis=1)
        delta = per_ev_post - per_ev_base
        ok = np.isfinite(delta)
        d_mean = np.nanmean(delta[ok])
        d_se = np.nanstd(delta[ok]) / np.sqrt(ok.sum())
        frac_pos = float((delta[ok] > 0).mean())
        print(f"\n  {c.upper()}")
        print(f"    pre-onset baseline (-30..0 min)  {base:+.4f}")
        print(f"    post-onset mean    ( 0..10 min)  {p0_10:+.4f}")
        print(f"    post-onset mean    ( 0..30 min)  {p0_30:+.4f}")
        print(f"    peak of mean trace at            {grid[peak_i]/60:+.1f} min "
              f"({mean[peak_i]:+.4f})")
        print(f"    per-event shift (0-10 vs base)   {d_mean:+.4f} "
              f"+/- {d_se:.4f} SE   (n={int(ok.sum())})")
        print(f"    events with a positive shift     {frac_pos:.1%}")
        print(f"    |shift| / SE                     {abs(d_mean)/d_se:.2f}")
        stats[c] = (mean, se, n, d_mean, d_se, frac_pos)

    # ---- figures -----------------------------------------------------------
    x = grid / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    colors = {"eda": "#2F6F9F", "hr": "#A8443C"}
    for ax, c in zip(axes, CH):
        mean, se = stats[c][0], stats[c][1]
        ax.plot(x, mean, lw=1.6, color=colors[c])
        ax.fill_between(x, mean - se, mean + se, alpha=.22, color=colors[c], lw=0)
        ax.axvline(0, color="#17212B", lw=1.1)
        ax.axhline(0, color="#5B6B7A", lw=.7, ls=":")
        style_axes(ax, f"{c.upper()} — mean across {n_ev} level-2 events (±1 SE)",
                   None, "causal z")
    axes[-1].set_xlabel("minutes from event onset", fontsize=9)
    fig.suptitle("A10 · Event-triggered average, pooled across subjects",
                 fontsize=12, x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A10_event_triggered_average.png"), dpi=150)
    plt.close(fig)

    top6 = M.subject.value_counts().head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2), sharex=True, sharey=True)
    for ax, sub in zip(axes.ravel(), top6):
        m = (M.subject == sub).values
        for c in CH:
            arr = A[c][m]
            mu = np.nanmean(arr, axis=0)
            nn = np.isfinite(arr).sum(axis=0)
            se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(nn, 1))
            ax.plot(x, mu, lw=1.3, color=colors[c], label=c.upper())
            ax.fill_between(x, mu - se, mu + se, alpha=.18, color=colors[c], lw=0)
        ax.axvline(0, color="#17212B", lw=1)
        ax.axhline(0, color="#5B6B7A", lw=.7, ls=":")
        style_axes(ax, f"{sub}  (n={int(m.sum())} events)", None, None)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("min from onset", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("causal z", fontsize=8.5)
    fig.suptitle("A10 · Event-triggered average, faceted by subject",
                 fontsize=12, x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A10_event_triggered_by_subject.png"), dpi=150)
    plt.close(fig)

    # ---- stop condition ----------------------------------------------------
    banner("A10 STOP CONDITION - is the pooled trace flat?")
    flat = all(abs(stats[c][3]) / stats[c][4] < 2 for c in CH)
    for c in CH:
        _, _, _, d, se, fp = stats[c]
        verdict = "FLAT" if abs(d) / se < 2 else "RESPONDS"
        print(f"  {c.upper():4} shift {d:+.4f} +/- {se:.4f}  "
              f"|t|={abs(d)/se:5.2f}  {verdict}")
    print("\nTRIGGERED - pooled trace is flat on both channels"
          if flat else "\nnot triggered - at least one channel responds at onset")

    pd.DataFrame({"sec_from_onset": grid,
                  **{f"{c}_mean": stats[c][0] for c in CH},
                  **{f"{c}_se": stats[c][1] for c in CH},
                  **{f"{c}_n": stats[c][2] for c in CH}}
                 ).to_csv(os.path.join(OUT, "a10_event_triggered.csv"), index=False)

    append_findings(
        "A10", "Event-triggered average", "tasks/audit_a10.py",
        [("level-2 events with usable coverage", f"{n_ev} / {len(ev)}",
          "n.a.", "n.a."),
         *[(f"{c.upper()} per-event shift (0-10 min vs baseline)",
            f"{stats[c][3]:+.4f} +/- {stats[c][4]:.4f} SE",
            "visible rise", "yes" if abs(stats[c][3]) / stats[c][4] >= 2 else "NO")
           for c in CH],
         *[(f"{c.upper()} events with positive shift", f"{stats[c][5]:.1%}",
            ">50%", "yes" if stats[c][5] > .5 else "NO") for c in CH]],
        ["figures/audit/A10_event_triggered_average.png",
         "figures/audit/A10_event_triggered_by_subject.png"],
        "Decides whether the labels mark a physiologically distinguishable state. "
        "The shape also indicates where the informative part of the interval sits, "
        "which is the empirical input to boundary trimming (JC09).",
        "flat pooled trace - see stop condition" if flat else "none",
    )


if __name__ == "__main__":
    main()
