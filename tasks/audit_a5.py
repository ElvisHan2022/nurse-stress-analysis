"""A5 - IBI structure and HRV feasibility.

Verify the time convention before trusting any HRV feature, then segment into
usable runs and report window-level coverage - the number that actually decides
whether HRV enters the baseline feature set.
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
from audit_common import OUT, FIG, session_dirs, banner, append_findings, style_axes

TOL = 0.05          # s, agreement between elapsed-time delta and the interval
IBI_LO, IBI_HI = 0.33, 1.50     # 40-180 bpm
MALIK = 0.20
MIN_RUN_S, MIN_RUN_BEATS = 30, 20
WINDOW_S = 120


def load_ibi_raw(path):
    """Returns (elapsed_s, interval_s) arrays. Row 1 is '<t0>, IBI' text."""
    try:
        v = np.atleast_2d(np.loadtxt(path, skiprows=1, delimiter=","))
    except Exception:
        return np.array([]), np.array([])
    if v.size == 0 or v.ndim != 2 or v.shape[1] < 2:
        return np.array([]), np.array([])
    return v[:, 0], v[:, 1]


def segment_runs(t, ibi):
    """Split into runs, breaking on dropped beats, non-physiological intervals,
    and Malik-criterion ectopics. Returns list of (start_idx, end_idx) inclusive."""
    n = len(t)
    if n < 2:
        return []
    brk = np.zeros(n, dtype=bool)
    dt = np.diff(t)
    brk[1:] |= np.abs(dt - ibi[1:]) > TOL              # dropped beat
    bad = (ibi < IBI_LO) | (ibi > IBI_HI)              # non-physiological
    brk |= bad
    ect = np.zeros(n, dtype=bool)
    ect[1:] = np.abs(np.diff(ibi)) > MALIK * ibi[:-1]  # Malik
    brk |= ect
    runs, start = [], 0
    for i in range(1, n):
        if brk[i]:
            if i - 1 >= start:
                runs.append((start, i - 1))
            start = i
    if n - 1 >= start:
        runs.append((start, n - 1))
    return [(a, b) for a, b in runs if b > a]


def main():
    banner("A5 - IBI STRUCTURE AND HRV FEASIBILITY")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}

    # ---- 1. verify the time convention ------------------------------------
    banner("Step 1 - verify the elapsed/interval convention")
    print("\nConvention: the elapsed column marks the SECOND beat of each pair,")
    print(f"so t[i] - t[i-1] should equal ibi[i] within {TOL*1000:.0f} ms.")
    print("Report the number; an off-by-one here silently corrupts every feature.\n")

    agree_num = agree_den = 0
    checked = 0
    per_sess = []
    for sess, sd in sorted(paths.items()):
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        if len(t) < 20:
            continue
        checked += 1
        ok = np.abs(np.diff(t) - ibi[1:]) <= TOL
        agree_num += int(ok.sum())
        agree_den += int(ok.size)
        per_sess.append({"session": sess, "beats": len(t),
                         "frac_agree": float(ok.mean())})
    PS = pd.DataFrame(per_sess)
    frac = agree_num / max(agree_den, 1)
    print(f"sessions checked (>=20 beats): {checked}")
    print(f"consecutive pairs checked:     {agree_den:,}")
    print(f"fraction satisfying convention: {frac:.4f}")
    print(f"\nper-session fraction: median {PS.frac_agree.median():.4f}  "
          f"p10 {PS.frac_agree.quantile(.1):.4f}  min {PS.frac_agree.min():.4f}")
    print("\nA high fraction confirms the convention AND confirms that a violation")
    print("indicates a dropped beat rather than an indexing error.")

    # ---- 2. segment into usable runs --------------------------------------
    banner("Step 2 - segment into usable runs")
    print(f"\nbreak rules: |dt - ibi| > {TOL}s (dropped beat); "
          f"ibi outside [{IBI_LO}, {IBI_HI}]s; Malik |d ibi| > {MALIK:.0%}")
    print(f"usable run: >= {MIN_RUN_S}s AND >= {MIN_RUN_BEATS} beats (both)\n")

    rows = []
    poincare = []
    for r in S.itertuples():
        sd = paths.get(r.session)
        if sd is None:
            continue
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        rec = {"subject": r.subject, "session": r.session, "dur_s": r.dur_s,
               "beats": len(t), "n_runs": 0, "n_usable": 0,
               "longest_run_s": 0.0, "usable_s": 0.0}
        if len(t) >= 2:
            runs = segment_runs(t, ibi)
            rec["n_runs"] = len(runs)
            usable = []
            for a, b in runs:
                span = t[b] - t[a]
                nb = b - a + 1
                if span >= MIN_RUN_S and nb >= MIN_RUN_BEATS:
                    usable.append((a, b, span))
            rec["n_usable"] = len(usable)
            rec["usable_s"] = float(sum(s for _, _, s in usable))
            rec["longest_run_s"] = float(max((s for _, _, s in usable), default=0.0))
            if usable and len(poincare) < 6:
                a, b, _ = max(usable, key=lambda u: u[2])
                poincare.append((r.session, ibi[a:b + 1]))
        rec["frac_time_usable"] = (rec["usable_s"] / r.dur_s) if r.dur_s else 0.0
        rows.append(rec)
    H = pd.DataFrame(rows)
    H.to_parquet(os.path.join(OUT, "a5_hrv_runs.parquet"))

    print("--- runs per session ---")
    print(H.n_runs.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
    print("\n--- usable runs per session ---")
    print(H.n_usable.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
    print("\n--- longest usable run (s) ---")
    print(H.longest_run_s.describe(percentiles=[.5, .9]).round(1).to_string())
    print("\n--- fraction of session time inside a usable run ---")
    print(H.frac_time_usable.describe(percentiles=[.25, .5, .75, .9]).round(4).to_string())

    n_any = int((H.n_usable > 0).sum())
    print(f"\nsessions with >=1 usable run: {n_any} / {len(H)} "
          f"({100*n_any/len(H):.1f}%)")

    # ---- 3. THE number that matters: window-level coverage -----------------
    banner("Step 3 - window-level coverage (the number that matters)")
    print(f"\nTile each session into {WINDOW_S}s windows; a window is covered if it")
    print(f"contains a usable run of >= {MIN_RUN_S}s. Session-level counts flatter")
    print("the situation: three short runs still leave most windows uncovered.\n")

    wrows = []
    for r in S.itertuples():
        sd = paths.get(r.session)
        if sd is None or not r.dur_s or r.dur_s < WINDOW_S:
            continue
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        nwin = int(r.dur_s // WINDOW_S)
        cov = np.zeros(nwin, dtype=bool)
        if len(t) >= 2:
            for a, b in segment_runs(t, ibi):
                if (t[b] - t[a]) >= MIN_RUN_S and (b - a + 1) >= MIN_RUN_BEATS:
                    # a run covers every window it overlaps by >= MIN_RUN_S
                    w0, w1 = int(t[a] // WINDOW_S), int(t[b] // WINDOW_S)
                    for w in range(max(w0, 0), min(w1 + 1, nwin)):
                        lo, hi = w * WINDOW_S, (w + 1) * WINDOW_S
                        if min(t[b], hi) - max(t[a], lo) >= MIN_RUN_S:
                            cov[w] = True
        wrows.append({"subject": r.subject, "session": r.session,
                      "n_windows": nwin, "n_covered": int(cov.sum())})
    W = pd.DataFrame(wrows)
    W["frac"] = W.n_covered / W.n_windows.replace(0, np.nan)
    tot_w, tot_c = int(W.n_windows.sum()), int(W.n_covered.sum())
    overall = tot_c / tot_w
    print(f"total {WINDOW_S}s windows:      {tot_w:,}")
    print(f"windows with usable HRV:   {tot_c:,}")
    print(f"OVERALL WINDOW COVERAGE:   {overall:.4f}  ({100*overall:.2f}%)")
    print("\n--- per-session coverage fraction ---")
    print(W.frac.describe(percentiles=[.25, .5, .75, .9]).round(4).to_string())
    print("\n--- per-subject window coverage ---")
    PSU = (W.groupby("subject")
             .agg(windows=("n_windows", "sum"), covered=("n_covered", "sum")))
    PSU["coverage"] = (PSU.covered / PSU.windows).round(4)
    print(PSU.sort_values("coverage", ascending=False).to_string())
    W.to_parquet(os.path.join(OUT, "a5_hrv_window_coverage.parquet"))

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.hist(W.frac.dropna() * 100, bins=40, color="#2F6F9F", edgecolor="white",
            linewidth=.4)
    ax.axvline(100 * overall, color="#A8443C", ls="--", lw=1.5,
               label=f"overall {100*overall:.1f}%")
    ax.legend(fontsize=9, frameon=False)
    style_axes(ax, f"A5 · Per-session HRV coverage of {WINDOW_S}s windows",
               "percent of windows containing a usable run", "sessions")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A5_hrv_window_coverage.png"), dpi=150)
    plt.close(fig)

    if poincare:
        ncol = min(len(poincare), 3)
        nrow = int(np.ceil(len(poincare) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow),
                                 squeeze=False)
        for ax, (sess, seq) in zip(axes.ravel(), poincare):
            ax.scatter(seq[:-1], seq[1:], s=11, alpha=.6, color="#2F6F9F")
            lim = [min(seq) * .95, max(seq) * 1.05]
            ax.plot(lim, lim, color="#A8443C", lw=.9, ls="--")
            style_axes(ax, f"{sess}  ({len(seq)} beats)", "IBI(n) s", "IBI(n+1) s")
        for ax in axes.ravel()[len(poincare):]:
            ax.axis("off")
        fig.suptitle("A5 · Poincaré plots of the longest usable run "
                     "(comet along the diagonal = clean)", fontsize=11, x=.02, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "A5_poincare.png"), dpi=150)
        plt.close(fig)

    banner("A5 - DECISION INPUT")
    print(f"\nwindow-level HRV coverage is {100*overall:.2f}%.")
    print("Report the number; the include/demote decision belongs to A6 and the plan.")

    append_findings(
        "A5", "IBI structure and HRV feasibility", "tasks/audit_a5.py",
        [("time-convention agreement", f"{frac:.4f}", "high (>0.9)",
          "yes" if frac > .9 else "NO"),
         ("sessions with >=1 usable run", f"{n_any}/{len(H)}", "75/105 sampled",
          "n.a. (full archive here)"),
         ("median usable runs per session", f"{H.n_usable.median():.0f}", "3",
          "yes" if abs(H.n_usable.median() - 3) <= 1 else "NO"),
         (f"WINDOW-level coverage ({WINDOW_S}s)", f"{100*overall:.2f}%",
          "unknown - this is the new number", "n.a."),
         ("median session-level coverage", f"{100*W.frac.median():.2f}%",
          "n.a.", "n.a.")],
        ["figures/audit/A5_hrv_window_coverage.png",
         "figures/audit/A5_poincare.png"],
        "Feeds decision point 7 (HRV in the baseline feature set or demoted to an "
        "ablation) and JC15/JC17.",
        f"session-level coverage ({100*(H.n_usable>0).mean():.0f}% of sessions have "
        f"a run) is far more flattering than window-level ({100*overall:.1f}%).",
    )


if __name__ == "__main__":
    main()
