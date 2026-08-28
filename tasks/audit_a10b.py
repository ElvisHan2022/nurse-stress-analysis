"""A10b - Robustness of the A10 result.

The pooled EDA mean peaks at +25 causal-z units, which is not a physiologically
plausible z-score. The causal-z denominator is a trailing IQR, so any stretch
where EDA is nearly constant drives IQR toward zero and the ratio toward
infinity. A handful of such events can manufacture a "response" in the mean.

This section re-runs the same comparison with outlier-resistant statistics.
If the effect survives the median and the sign test, it is real.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, FIG, load_survey, session_dirs, banner, append_findings, style_axes
from audit_a10 import fast_signal, causal_z, PRE_MIN, POST_MIN, CH


def main():
    banner("A10b - ROBUSTNESS OF THE EVENT-TRIGGERED AVERAGE")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    S["start_utc"] = pd.to_datetime(S.start_utc, utc=True).astype("datetime64[ns, UTC]")
    S["end_utc"] = pd.to_datetime(S.end_utc, utc=True).astype("datetime64[ns, UTC]")
    paths = {os.path.basename(d): d for d in session_dirs()}

    SV = load_survey()
    ev = SV[SV.labelled & (SV["Stress level"] == 2.0)].copy()
    ev = ev.drop_duplicates(["subject", "date", "Start time", "End time"])

    pre, post = pd.Timedelta(minutes=PRE_MIN), pd.Timedelta(minutes=POST_MIN)
    need = set()
    for r in ev.itertuples():
        lo, hi = r.start_utc - pre, r.start_utc + post
        m = (S.subject == r.subject) & (S.end_utc >= lo) & (S.start_utc <= hi)
        need.update(S.loc[m, "session"])

    print(f"\nreloading {len(need)} sessions...")
    t0 = time.time()
    frames = {}
    for sess in sorted(need):
        sd = paths.get(sess)
        if sd is None:
            continue
        try:
            d = pd.DataFrame({"eda": fast_signal(os.path.join(sd, "EDA.csv"), "EDA"),
                              "hr": fast_signal(os.path.join(sd, "HR.csv"), "HR")})
        except Exception:
            continue
        for c in CH:
            d[c + "_z"] = causal_z(d[c])
            # Raw units too, so the effect can be read in microsiemens and bpm.
            d[c + "_raw"] = d[c]
        frames[sess] = d
    print(f"loaded in {time.time()-t0:.0f}s")

    grid = np.arange(-PRE_MIN * 60, POST_MIN * 60 + 1)
    rows = []
    trace_z = {c: [] for c in CH}
    for r in ev.itertuples():
        lo, hi = r.start_utc - pre, r.start_utc + post
        m = (S.subject == r.subject) & (S.end_utc >= lo) & (S.start_utc <= hi)
        parts = [frames[s] for s in S.loc[m, "session"] if s in frames]
        if not parts:
            continue
        d = pd.concat(parts).sort_index()
        d = d[~d.index.duplicated()].loc[lo:hi]
        if d.empty:
            continue
        off = ((d.index - r.start_utc).total_seconds()).astype(int)
        pre_m = off < 0
        post_m = (off >= 0) & (off < 10 * 60)
        if np.isfinite(d["eda_z"].values[pre_m]).sum() < 300 or post_m.sum() < 60:
            continue
        rec = {"subject": r.subject, "dur_min": r.duration_min}
        for c in CH:
            zb = np.nanmean(d[c + "_z"].values[pre_m])
            zp = np.nanmean(d[c + "_z"].values[post_m])
            rb = np.nanmean(d[c + "_raw"].values[pre_m])
            rp = np.nanmean(d[c + "_raw"].values[post_m])
            rec[f"{c}_dz"] = zp - zb
            rec[f"{c}_draw"] = rp - rb
            rec[f"{c}_base_raw"] = rb
            ser = pd.Series(d[c + "_z"].values, index=off).reindex(grid)
            trace_z[c].append(ser.values)
        rows.append(rec)

    E = pd.DataFrame(rows)
    n = len(E)
    print(f"\nevents analysed: {n}")

    banner("Per-event shift: mean vs median vs sign test")
    out = []
    for c in CH:
        dz = E[f"{c}_dz"].dropna()
        draw = E[f"{c}_draw"].dropna()
        npos = int((dz > 0).sum())
        sign_p = st.binomtest(npos, len(dz), .5).pvalue
        w_p = st.wilcoxon(dz)[1] if len(dz) > 10 else np.nan
        print(f"\n  {c.upper()}   (n={len(dz)} events)")
        print(f"    mean   dz  {dz.mean():+10.4f}   <- outlier-sensitive")
        print(f"    MEDIAN dz  {dz.median():+10.4f}   <- outlier-resistant")
        print(f"    IQR    dz  [{dz.quantile(.25):+.4f}, {dz.quantile(.75):+.4f}]")
        print(f"    max    dz  {dz.max():+10.4f}    min {dz.min():+.4f}")
        print(f"    events with positive shift  {npos}/{len(dz)} = {npos/len(dz):.1%}")
        print(f"    sign test p                 {sign_p:.4g}")
        print(f"    Wilcoxon signed-rank p      {w_p:.4g}")
        unit = "uS" if c == "eda" else "bpm"
        print(f"    RAW median shift            {draw.median():+.4f} {unit}"
              f"   (mean {draw.mean():+.4f})")
        print(f"    RAW baseline median         {E[f'{c}_base_raw'].median():.3f} {unit}")
        out.append({"channel": c.upper(), "n": len(dz),
                    "mean_dz": round(dz.mean(), 4),
                    "median_dz": round(dz.median(), 4),
                    "pct_positive": round(100 * npos / len(dz), 1),
                    "sign_p": f"{sign_p:.3g}", "wilcoxon_p": f"{w_p:.3g}",
                    "median_raw_shift": round(draw.median(), 4)})

    R = pd.DataFrame(out)
    print("\n" + R.to_string(index=False))

    # how concentrated is the mean?
    banner("Is the EDA mean driven by a few events?")
    dz = E["eda_dz"].dropna().sort_values()
    tot = dz.sum()
    top5 = dz.tail(5).sum()
    print(f"\n  sum of all {len(dz)} per-event EDA shifts: {tot:.2f}")
    print(f"  contributed by the 5 largest events:     {top5:.2f} "
          f"({100*top5/tot:.1f}% of the total)")
    print(f"  largest 5 shifts: {[round(v,1) for v in dz.tail(5)]}")
    print("\n  A mean dominated by a handful of events is a statement about those")
    print("  events, not about the population. The median and sign test are the")
    print("  defensible summaries here.")

    # ---- robust figure -----------------------------------------------------
    x = grid / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    colors = {"eda": "#2F6F9F", "hr": "#A8443C"}
    for ax, c in zip(axes, CH):
        arr = np.vstack(trace_z[c])
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanquantile(arr, .25, axis=0)
        q75 = np.nanquantile(arr, .75, axis=0)
        ax.plot(x, med, lw=1.6, color=colors[c], label="median across events")
        ax.fill_between(x, q25, q75, alpha=.20, color=colors[c], lw=0,
                        label="inter-quartile range")
        ax.axvline(0, color="#17212B", lw=1.1)
        ax.axhline(0, color="#5B6B7A", lw=.7, ls=":")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        style_axes(ax, f"{c.upper()} — MEDIAN across {n} level-2 events",
                   None, "causal z")
    axes[-1].set_xlabel("minutes from event onset", fontsize=9)
    fig.suptitle("A10b · Event-triggered response, outlier-resistant",
                 fontsize=12, x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A10b_event_triggered_median.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for c, off in zip(CH, [-.18, .18]):
        v = E[f"{c}_dz"].dropna().clip(-6, 6)
        ax.scatter(v, np.random.default_rng(0).normal(off, .055, len(v)),
                   s=13, alpha=.55, color=colors[c], label=f"{c.upper()} (clipped ±6)")
        ax.plot([v.median()], [off], "|", ms=26, mew=2.6, color="#17212B")
    ax.axvline(0, color="#17212B", lw=1)
    ax.set_yticks([-.18, .18])
    ax.set_yticklabels(["EDA", "HR"])
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    style_axes(ax, "A10b · Per-event shift (0–10 min post-onset vs pre-onset baseline)",
               "change in causal z  ·  vertical bar = median", None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A10b_per_event_shift.png"), dpi=150)
    plt.close(fig)

    E.to_csv(os.path.join(OUT, "a10b_per_event_shift.csv"), index=False)

    banner("A10b VERDICT")
    for r in R.itertuples():
        real = (r.pct_positive > 55) and float(r.sign_p) < .05
        print(f"  {r.channel:4} median dz {r.median_dz:+.4f}  "
              f"{r.pct_positive:.0f}% positive  sign p={r.sign_p}  "
              f"-> {'REAL, modest' if real else 'NOT SUPPORTED'}")

    append_findings(
        "A10b", "Robustness of the event-triggered average",
        "tasks/audit_a10b.py",
        [(f"{r.channel} mean dz (outlier-sensitive)", r.mean_dz, "n.a.", "n.a.")
         for r in R.itertuples()] +
        [(f"{r.channel} MEDIAN dz (robust)", r.median_dz, "n.a.", "n.a.")
         for r in R.itertuples()] +
        [(f"{r.channel} events positive", f"{r.pct_positive}%", ">50%",
          "yes" if r.pct_positive > 55 else "NO") for r in R.itertuples()] +
        [(f"{r.channel} sign-test p", r.sign_p, "<0.05",
          "yes" if float(r.sign_p) < .05 else "NO") for r in R.itertuples()],
        ["figures/audit/A10b_event_triggered_median.png",
         "figures/audit/A10b_per_event_shift.png"],
        "Determines whether the A10 result survives outlier-resistant statistics. "
        "The pooled mean is not a defensible summary when the causal-z denominator "
        "can approach zero; the median and sign test are.",
        f"the A10 pooled EDA mean of +9.75 is dominated by a few events; "
        f"the median is {R[R.channel=='EDA'].median_dz.iloc[0]:+.4f}",
    )


if __name__ == "__main__":
    main()
