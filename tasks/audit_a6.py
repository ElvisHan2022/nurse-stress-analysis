"""A6 - Missingness, and whether it is informative.

Builds the 120s window table (reused by A8), then runs the decisive test: a
classifier whose ONLY inputs are missingness indicators and run length,
predicting whether a window falls inside a level-2 event, under LOSO.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (OUT, FIG, read_e4_header, load_survey, session_dirs,
                          banner, append_findings, style_axes)
from audit_a5 import load_ibi_raw, segment_runs, MIN_RUN_S, MIN_RUN_BEATS, WINDOW_S


def acc_window_means(path, nwin):
    """Mean |acc| and its SD per 120s window, without holding the whole file."""
    t0, fs = read_e4_header(path)
    v = pd.read_csv(path, skiprows=2, header=None).values / 64.0
    mag = np.sqrt((v ** 2).sum(axis=1))
    win = (np.arange(len(mag)) / fs // WINDOW_S).astype(int)
    df = pd.DataFrame({"w": win, "m": mag})
    g = df.groupby("w").m.agg(["mean", "std"])
    out_m = np.full(nwin, np.nan)
    out_s = np.full(nwin, np.nan)
    idx = g.index.values
    keep = (idx >= 0) & (idx < nwin)
    out_m[idx[keep]] = g["mean"].values[keep]
    out_s[idx[keep]] = g["std"].values[keep]
    return out_m, out_s


def build_windows():
    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    S["start_utc"] = pd.to_datetime(S.start_utc, utc=True).astype("datetime64[ns, UTC]")
    paths = {os.path.basename(d): d for d in session_dirs()}

    SV = load_survey()
    SV = SV.drop_duplicates(["subject", "date", "Start time", "End time"])
    ev2 = SV[SV.labelled & (SV["Stress level"] == 2.0)]
    ev_any = SV                       # every reported event, rated or not

    rows = []
    t_start = time.time()
    for i, r in enumerate(S.itertuples()):
        if i and i % 100 == 0:
            print(f"    ... {i}/{len(S)} sessions ({time.time()-t_start:.0f}s)")
        sd = paths.get(r.session)
        if sd is None or not r.dur_s or r.dur_s < WINDOW_S:
            continue
        nwin = int(r.dur_s // WINDOW_S)

        # HRV availability per window
        hrv_ok = np.zeros(nwin, dtype=bool)
        hrv_s = np.zeros(nwin)
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        if len(t) >= 2:
            for a, b in segment_runs(t, ibi):
                if (t[b] - t[a]) >= MIN_RUN_S and (b - a + 1) >= MIN_RUN_BEATS:
                    w0, w1 = int(t[a] // WINDOW_S), int(t[b] // WINDOW_S)
                    for w in range(max(w0, 0), min(w1 + 1, nwin)):
                        lo, hi = w * WINDOW_S, (w + 1) * WINDOW_S
                        ov = min(t[b], hi) - max(t[a], lo)
                        if ov >= MIN_RUN_S:
                            hrv_ok[w] = True
                            hrv_s[w] = max(hrv_s[w], ov)

        try:
            accm, accs = acc_window_means(os.path.join(sd, "ACC.csv"), nwin)
        except Exception:
            accm = accs = np.full(nwin, np.nan)

        w_start = r.start_utc + pd.to_timedelta(np.arange(nwin) * WINDOW_S, unit="s")
        w_end = w_start + pd.Timedelta(seconds=WINDOW_S)

        # label: >=50% of the window inside a level-2 event
        sub2 = ev2[ev2.subject == r.subject]
        pos = np.zeros(nwin, dtype=bool)
        for e in sub2.itertuples():
            ov = ((np.minimum(w_end, e.end_utc) - np.maximum(w_start, e.start_utc))
                  .total_seconds())
            pos |= (ov >= WINDOW_S * 0.5)
        # any reported event, used for the guard band in A8
        subA = ev_any[ev_any.subject == r.subject]
        near = np.zeros(nwin, dtype=bool)
        inany = np.zeros(nwin, dtype=bool)
        for e in subA.itertuples():
            ov = ((np.minimum(w_end, e.end_utc) - np.maximum(w_start, e.start_utc))
                  .total_seconds())
            inany |= (ov > 0)
        rows.append(pd.DataFrame({
            "subject": r.subject, "session": r.session,
            "w": np.arange(nwin), "start_utc": w_start, "end_utc": w_end,
            "hrv_available": hrv_ok, "hrv_run_s": hrv_s,
            "acc_mean": accm, "acc_sd": accs,
            "is_level2": pos, "in_any_event": inany,
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    banner("A6 - MISSINGNESS, AND WHETHER IT IS INFORMATIVE")

    wp = os.path.join(OUT, "a6_windows.parquet")
    if os.path.exists(wp):
        print("\nreusing cached window table")
        W = pd.read_parquet(wp)
    else:
        print(f"\nbuilding {WINDOW_S}s window table (reads every ACC.csv)...")
        W = build_windows()
        W.to_parquet(wp)
    print(f"\nwindows: {len(W):,}   subjects: {W.subject.nunique()}")
    print(f"level-2 windows: {int(W.is_level2.sum()):,} "
          f"({100*W.is_level2.mean():.2f}%)")

    print("\n--- head ---")
    print(W.head(8).to_string())

    # ---- missingness rates -------------------------------------------------
    banner("Missingness rate per feature")
    miss = pd.Series({
        "hrv (no usable run)": 1 - W.hrv_available.mean(),
        "acc_mean": W.acc_mean.isna().mean(),
        "acc_sd": W.acc_sd.isna().mean(),
    })
    print()
    print((miss * 100).round(2).to_string())

    print("\n--- HRV missingness per subject (sorted) ---")
    ps = (1 - W.groupby("subject").hrv_available.mean()).sort_values(ascending=False)
    print((ps * 100).round(2).to_string())
    print(f"\nspread: {100*ps.min():.1f}% to {100*ps.max():.1f}% "
          f"(ratio of availability {(1-ps.min())/max(1-ps.max(),1e-9):.1f}x)")

    # ---- HRV availability by ACC decile -----------------------------------
    banner("HRV availability by accelerometer decile and class")
    D = W.dropna(subset=["acc_mean"]).copy()
    D["acc_dec"] = pd.qcut(D.acc_mean, 10, labels=False, duplicates="drop")
    tab = (D.groupby(["acc_dec", "is_level2"]).hrv_available.mean().unstack() * 100)
    tab.columns = ["not level-2", "level-2"]
    print()
    print(tab.round(2).to_string())
    print("\nMechanism: beat rejection is motion-driven, motion tracks being on the")
    print("ward, and events happen on the ward. If availability differs by class")
    print("WITHIN a decile, missingness carries label information.")

    # ---- the decisive test -------------------------------------------------
    banner("Decisive test - can missingness alone predict the label?")
    print("\nfeatures: hrv_available, hrv_run_s, acc_mean missing indicator")
    print("target:   window is inside a level-2 event")
    print("protocol: leave-one-subject-out, pooled AUC\n")

    X = pd.DataFrame({
        "hrv_available": W.hrv_available.astype(float),
        "hrv_run_s": W.hrv_run_s.fillna(0.0),
        "acc_missing": W.acc_mean.isna().astype(float),
    }).values
    y = W.is_level2.astype(int).values
    grp = W.subject.values

    oof = np.full(len(y), np.nan)
    for sub in np.unique(grp):
        te = grp == sub
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    auc = roc_auc_score(y[ok], oof[ok])
    print(f"LOSO pooled AUC from missingness alone: {auc:.4f}")

    per = []
    for sub in np.unique(grp):
        m = (grp == sub) & ok
        if m.sum() and len(np.unique(y[m])) == 2:
            per.append({"subject": sub, "n": int(m.sum()),
                        "pos": int(y[m].sum()),
                        "auc": round(roc_auc_score(y[m], oof[m]), 4)})
    P = pd.DataFrame(per).sort_values("auc", ascending=False)
    print("\n--- per-subject AUC ---")
    print(P.to_string(index=False))
    print(f"\nmedian per-subject AUC: {P.auc.median():.4f}")

    verdict = ("INFORMATIVE - missingness carries label information"
               if auc > 0.55 else "not meaningfully informative")
    print(f"\nVERDICT: {verdict}")
    print("\nNote: LightGBM and XGBoost route missing values natively, so a tree")
    print("can split on missingness whether or not an indicator column is supplied.")
    print("A policy of never imputing does not close this by itself.")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    idx = np.arange(len(tab))
    ax.bar(idx - .19, tab["not level-2"], .38, label="not level-2", color="#5B6B7A")
    ax.bar(idx + .19, tab["level-2"], .38, label="level-2", color="#A8443C")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"D{i+1}" for i in idx])
    ax.legend(fontsize=9, frameon=False)
    style_axes(ax, f"A6 · HRV availability by accelerometer decile and class "
                   f"(LOSO AUC from missingness alone = {auc:.3f})",
               "accelerometer decile (low → high motion)",
               "% of windows with usable HRV")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A6_hrv_availability_by_activity.png"), dpi=150)
    plt.close(fig)

    append_findings(
        "A6", "Missingness and informativeness", "tasks/audit_a6.py",
        [("windows", f"{len(W):,}", "n.a.", "n.a."),
         ("level-2 windows", f"{int(W.is_level2.sum()):,} "
          f"({100*W.is_level2.mean():.2f}%)", "n.a.", "n.a."),
         ("HRV missing overall", f"{100*(1-W.hrv_available.mean()):.2f}%",
          "n.a.", "n.a."),
         ("LOSO AUC from missingness alone", f"{auc:.4f}", ">0.5 means informative",
          "INFORMATIVE" if auc > .55 else "no"),
         ("median per-subject AUC", f"{P.auc.median():.4f}", "n.a.", "n.a.")],
        ["figures/audit/A6_hrv_availability_by_activity.png"],
        "If informative, HRV availability is a label proxy and tree models will "
        "exploit it. Feeds decision point 7 alongside A5's coverage number.",
        verdict,
    )


if __name__ == "__main__":
    main()
