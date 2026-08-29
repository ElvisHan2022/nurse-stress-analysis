"""A5b - HRV availability as a function of window length and admission rule.

The 5.94% figure from A5 is not a property of the data alone. It is the joint
consequence of three choices that were never separated:

  1. WINDOW LENGTH   how long one prediction covers (A5 used 120s)
  2. ADMISSION RULE  how much clean beat data counts as usable (A5 used
                     >=30s AND >=20 beats)
  3. WHAT "AVAILABLE" MEANS
                     A5 said: a usable run exists somewhere in the window.
                     It did NOT say: the window is well covered by clean beats.

Choice 3 is the one nobody stated. Under it, a 30s run makes a 300s window
"available" while leaving 90% of that window with no beat data at all - so
longer windows look better purely by being easier to satisfy.

This sweeps all three and prints the table to choose from.
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
from audit_a5 import load_ibi_raw, segment_runs

# Window lengths under discussion. 90 and 120 are the sliding-window candidates;
# 300 is the classical short-term HRV recording length.
WINDOWS = [60, 90, 120, 180, 300]

# (min_run_seconds, min_beats, label)
ADMISSION = [
    (20, 15, "lenient  20s/15b"),
    (30, 20, "A5       30s/20b"),
    (60, 40, "strict   60s/40b"),
]

# Fraction of the window that must be covered by clean beats, under the
# stricter definition of "available".
COVER_FRACS = [0.0, 0.25, 0.50]


def main():
    banner("A5b - HRV AVAILABILITY SENSITIVITY")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}

    # Segment every session once. Segmentation does not depend on the window
    # length or the admission thresholds, so this is the only expensive pass.
    print("\nsegmenting IBI runs for every session (once)...")
    runs_by_session = {}
    for r in S.itertuples():
        sd = paths.get(r.session)
        if sd is None or not r.dur_s:
            continue
        t, ibi = load_ibi_raw(os.path.join(sd, "IBI.csv"))
        if len(t) < 2:
            runs_by_session[r.session] = (np.empty((0, 2)), r.dur_s)
            continue
        segs = segment_runs(t, ibi)
        arr = np.array([[t[a], t[b], b - a + 1] for a, b in segs]) \
            if segs else np.empty((0, 3))
        runs_by_session[r.session] = (arr, r.dur_s)
    n_with = sum(1 for v in runs_by_session.values() if len(v[0]))
    print(f"sessions with at least one raw run: {n_with} / {len(runs_by_session)}")

    def availability(win_s, min_run_s, min_beats, cover_frac):
        """Returns (n_windows, n_available, mean_covered_seconds)."""
        tot = avail = 0
        cov_sum = 0.0
        for sess, (arr, dur_s) in runs_by_session.items():
            if not dur_s or dur_s < win_s:
                continue
            nwin = int(dur_s // win_s)
            tot += nwin
            if not len(arr):
                continue
            usable = arr[(arr[:, 1] - arr[:, 0] >= min_run_s) &
                         (arr[:, 2] >= min_beats)]
            if not len(usable):
                continue
            # seconds of usable run inside each window
            secs = np.zeros(nwin)
            for a, b, _ in usable:
                w0 = max(int(a // win_s), 0)
                w1 = min(int(b // win_s), nwin - 1)
                for w in range(w0, w1 + 1):
                    lo, hi = w * win_s, (w + 1) * win_s
                    secs[w] += max(0.0, min(b, hi) - max(a, lo))
            cov_sum += secs.sum()
            if cover_frac <= 0:
                # A5's definition: a run of >= min_run_s overlaps this window
                # by at least min_run_s.
                ok = np.zeros(nwin, dtype=bool)
                for a, b, _ in usable:
                    w0 = max(int(a // win_s), 0)
                    w1 = min(int(b // win_s), nwin - 1)
                    for w in range(w0, w1 + 1):
                        lo, hi = w * win_s, (w + 1) * win_s
                        if min(b, hi) - max(a, lo) >= min_run_s:
                            ok[w] = True
                avail += int(ok.sum())
            else:
                avail += int((secs >= cover_frac * win_s).sum())
        return tot, avail, (cov_sum / tot if tot else 0.0)

    # ---- 1. A5's definition, swept -----------------------------------------
    banner("Definition A - 'a usable run EXISTS in the window' (what A5 used)")
    print("\nA 30s run marks a 300s window available even though 90% of that")
    print("window has no beat data. Longer windows therefore look better here")
    print("for a reason that has nothing to do with data quality.\n")

    rowsA = []
    for min_run, min_beats, lab in ADMISSION:
        rec = {"admission": lab}
        for win in WINDOWS:
            tot, av, _ = availability(win, min_run, min_beats, 0.0)
            rec[f"{win}s"] = f"{100*av/tot:.1f}%" if tot else "n/a"
        rowsA.append(rec)
    A = pd.DataFrame(rowsA).set_index("admission")
    print(A.to_string())

    # ---- 2. the honest definition ------------------------------------------
    banner("Definition B - 'the window is COVERED by clean beats'")
    print("\nRequires usable-run seconds to reach a fraction of the window.")
    print("This is the number that says whether an HRV feature computed on")
    print("that window is describing the window or a small slice of it.\n")

    for frac in COVER_FRACS[1:]:
        print(f"--- at least {frac:.0%} of the window covered ---")
        rows = []
        for min_run, min_beats, lab in ADMISSION:
            rec = {"admission": lab}
            for win in WINDOWS:
                tot, av, _ = availability(win, min_run, min_beats, frac)
                rec[f"{win}s"] = f"{100*av/tot:.1f}%" if tot else "n/a"
            rows.append(rec)
        print(pd.DataFrame(rows).set_index("admission").to_string())
        print()

    # ---- 3. usable seconds, the definition-free number ---------------------
    banner("Definition-free: mean seconds of usable beat data per window")
    print("\nIndependent of any availability threshold. Divide by the window")
    print("length to get the fraction of a window that is real HRV data.\n")

    rows = []
    for min_run, min_beats, lab in ADMISSION:
        rec = {"admission": lab}
        for win in WINDOWS:
            tot, av, mean_cov = availability(win, min_run, min_beats, 0.0)
            rec[f"{win}s"] = f"{mean_cov:.1f}s ({100*mean_cov/win:.1f}%)"
        rows.append(rec)
    print(pd.DataFrame(rows).set_index("admission").to_string())

    # ---- 4. total usable minutes, the thing that actually matters ----------
    banner("How much usable HRV data exists at all")
    tot_usable = {}
    for min_run, min_beats, lab in ADMISSION:
        secs = 0.0
        for arr, _ in runs_by_session.values():
            if not len(arr):
                continue
            u = arr[(arr[:, 1] - arr[:, 0] >= min_run_s_of(lab)) &
                    (arr[:, 2] >= min_beats)] if False else \
                arr[(arr[:, 1] - arr[:, 0] >= min_run) & (arr[:, 2] >= min_beats)]
            secs += float((u[:, 1] - u[:, 0]).sum()) if len(u) else 0.0
        tot_usable[lab] = secs
        print(f"  {lab}:  {secs/3600:7.1f} h of usable beat data "
              f"({100*secs/(S.dur_s.sum()):.2f}% of {S.dur_s.sum()/3600:,.0f} h recorded)")

    print("\nThis quantity does not depend on the window length at all.")
    print("Window choice only redistributes it; it cannot create more.")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for min_run, min_beats, lab in ADMISSION:
        ya = [100 * availability(w, min_run, min_beats, 0.0)[1] /
              max(availability(w, min_run, min_beats, 0.0)[0], 1) for w in WINDOWS]
        yb = [100 * availability(w, min_run, min_beats, .5)[1] /
              max(availability(w, min_run, min_beats, .5)[0], 1) for w in WINDOWS]
        axes[0].plot(WINDOWS, ya, "o-", lw=1.6, label=lab)
        axes[1].plot(WINDOWS, yb, "o-", lw=1.6, label=lab)
    for ax, t in zip(axes, ["A · a usable run exists in the window",
                            "B · >=50% of the window covered"]):
        ax.axvline(120, color="#5B6B7A", ls=":", lw=1)
        ax.axvline(300, color="#B8762A", ls=":", lw=1)
        ax.legend(fontsize=8.5, frameon=False)
        style_axes(ax, t, "window length (s)", "% of windows with HRV")
    fig.suptitle("A5b · HRV availability depends on the definition as much as the data",
                 fontsize=12, x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A5b_hrv_sensitivity.png"), dpi=150)
    plt.close(fig)
    print(f"\nwrote {os.path.join(FIG, 'A5b_hrv_sensitivity.png')}")


def min_run_s_of(lab):
    return int(lab.split()[-1].split("s")[0])


if __name__ == "__main__":
    main()
