"""A9 - Attrition, and whether rule order matters.

Exclusion rules do not commute. The order is a second judgment call hiding
behind the first, so it is recorded and perturbed like any other.
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
from audit_common import (OUT, FIG, read_e4_header, load_survey, session_dirs,
                          banner, append_findings, style_axes)
from audit_a6 import WINDOW_S

EDA_FLOOR = 0.05
EDA_FLOOR_PCT = 50
MIN_SESSION_MIN = 5


def eda_window_stats(path, nwin):
    t0, fs = read_e4_header(path)
    v = pd.read_csv(path, skiprows=2, header=None).iloc[:, 0].values
    w = (np.arange(len(v)) / fs // WINDOW_S).astype(int)
    d = pd.DataFrame({"w": w, "e": v})
    g = d.groupby("w").e.agg(med="median", floor=lambda x: float((x < EDA_FLOOR).mean()))
    med = np.full(nwin, np.nan)
    flo = np.full(nwin, np.nan)
    idx = g.index.values
    k = (idx >= 0) & (idx < nwin)
    med[idx[k]] = g["med"].values[k]
    flo[idx[k]] = g["floor"].values[k]
    return med, flo


def main():
    banner("A9 - ATTRITION AND RULE ORDER")

    W = pd.read_parquet(os.path.join(OUT, "a6_windows.parquet"))
    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}

    cache = os.path.join(OUT, "a9_eda_stats.parquet")
    if os.path.exists(cache):
        E = pd.read_parquet(cache)
    else:
        print("\ncomputing per-window EDA statistics...")
        t0 = time.time()
        parts = []
        for i, r in enumerate(S.itertuples()):
            if i and i % 150 == 0:
                print(f"    ... {i}/{len(S)} ({time.time()-t0:.0f}s)")
            sd = paths.get(r.session)
            if sd is None or not r.dur_s or r.dur_s < WINDOW_S:
                continue
            nwin = int(r.dur_s // WINDOW_S)
            try:
                med, flo = eda_window_stats(os.path.join(sd, "EDA.csv"), nwin)
            except Exception:
                med = flo = np.full(nwin, np.nan)
            parts.append(pd.DataFrame({"session": r.session, "w": np.arange(nwin),
                                       "eda_med": med, "eda_floor": flo}))
        E = pd.concat(parts, ignore_index=True)
        E.to_parquet(cache)
    W = W.merge(E, on=["session", "w"], how="left")

    dur = S.set_index("session").dur_s
    W["sess_min"] = W.session.map(dur) / 60

    # non-wear: EDA at the floor AND accelerometer stillness
    W["nonwear"] = (W.eda_med < EDA_FLOOR) & (W.acc_sd < 0.005)

    # dead-EDA sessions
    sess_eda = W.groupby("session").agg(
        med=("eda_med", "median"), floor=("eda_floor", "mean"))
    dead = set(sess_eda[(sess_eda.med < EDA_FLOOR) |
                        (sess_eda.floor > EDA_FLOOR_PCT / 100)].index)
    W["dead_eda"] = W.session.isin(dead)
    print(f"\ndead-EDA sessions: {len(dead)} of {S.session.nunique()}")

    SV = load_survey()
    n_dup = int(SV.is_exact_dup.sum())
    n_unrated = int((~SV.labelled).sum())
    ev2 = SV[SV.labelled & (SV["Stress level"] == 2.0)].drop_duplicates(
        ["subject", "date", "Start time", "End time"])

    def run_attrition(order):
        """order: list of rule keys. Returns the attrition table."""
        alive = pd.Series(True, index=W.index)
        rows = []
        for key in order:
            before = int(alive.sum())
            if key == "short_session":
                drop = W.sess_min < MIN_SESSION_MIN
                desc = f"drop sessions < {MIN_SESSION_MIN} min"
            elif key == "nonwear":
                drop = W.nonwear.fillna(False)
                desc = "drop non-wear windows"
            elif key == "dead_eda":
                drop = W.dead_eda
                desc = "drop dead-EDA sessions"
            elif key == "no_event_day":
                drop = ~W.in_any_event & False   # placeholder, no window drop
                desc = "drop unrated events (survey-level, 0 windows)"
            else:
                drop = pd.Series(False, index=W.index)
                desc = key
            removed = int((alive & drop).sum())
            alive = alive & ~drop
            rows.append({"rule": desc, "entering": before, "removed": removed,
                         "surviving": int(alive.sum()),
                         "hours_surviving": round(int(alive.sum()) * WINDOW_S / 3600, 1)})
        return pd.DataFrame(rows), alive

    banner("Attrition, PLAN.md Phase 1 order (windows)")
    order_a = ["short_session", "nonwear", "dead_eda"]
    A, alive_a = run_attrition(order_a)
    print()
    print(A.to_string(index=False))

    banner("Attrition, rules 1 and 2 SWAPPED")
    order_b = ["nonwear", "short_session", "dead_eda"]
    B, alive_b = run_attrition(order_b)
    print()
    print(B.to_string(index=False))

    same = int(alive_a.sum()) == int(alive_b.sum())
    print(f"\nfinal surviving windows, order A: {int(alive_a.sum()):,}")
    print(f"final surviving windows, order B: {int(alive_b.sum()):,}")
    print(f"difference: {abs(int(alive_a.sum())-int(alive_b.sum())):,}")
    print("\nWindow-level exclusion is set intersection, so these rules commute.")
    print("Order matters when a rule's THRESHOLD is computed from surviving data")
    print("rather than applied to a fixed mask - see the session-level check below.")

    # ---- session-level, where order genuinely matters ----------------------
    banner("Session-level: does dropping non-wear first change the session count?")
    print("\nComputing session length AFTER removing non-wear windows changes which")
    print("sessions fall below the 5 min cutoff. This is the non-commuting case.\n")

    worn = W[~W.nonwear.fillna(False)]
    worn_min = worn.groupby("session").size() * WINDOW_S / 60
    raw_min = S.set_index("session").dur_s / 60

    a_sessions = set(raw_min[raw_min >= MIN_SESSION_MIN].index)
    b_sessions = set(worn_min[worn_min >= MIN_SESSION_MIN].index)
    print(f"  order A (length, then non-wear): {len(a_sessions)} sessions kept")
    print(f"  order B (non-wear, then length): {len(b_sessions)} sessions kept")
    print(f"  difference: {len(a_sessions - b_sessions)} sessions kept by A but "
          f"not B; {len(b_sessions - a_sessions)} the reverse")
    if a_sessions - b_sessions:
        print(f"  dropped only under B: "
              f"{', '.join(sorted(a_sessions - b_sessions)[:10])}"
              + (" ..." if len(a_sessions - b_sessions) > 10 else ""))

    # ---- survey-side attrition --------------------------------------------
    banner("Survey-side attrition (events)")
    srows = [
        {"rule": "all survey rows", "entering": len(SV), "removed": 0,
         "surviving": len(SV)},
        {"rule": "drop 3 exact duplicates", "entering": len(SV),
         "removed": n_dup, "surviving": len(SV) - n_dup},
        {"rule": "drop unrated ('na') events", "entering": len(SV) - n_dup,
         "removed": n_unrated, "surviving": len(SV) - n_dup - n_unrated},
        {"rule": "exclude levels 0 and 1", "entering": len(SV) - n_dup - n_unrated,
         "removed": len(SV) - n_dup - n_unrated - len(ev2), "surviving": len(ev2)},
        {"rule": "drop subject 6D", "entering": len(ev2),
         "removed": int((ev2.subject == "6D").sum()),
         "surviving": len(ev2) - int((ev2.subject == "6D").sum())},
    ]
    SR = pd.DataFrame(srows)
    print()
    print(SR.to_string(index=False))
    final_ev = SR.surviving.iloc[-1]
    print(f"\nFINAL level-2 events after all exclusions: {final_ev}")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = ["all"] + list(A.rule)
    vals = [len(W)] + list(A.surviving)
    ax.step(range(len(vals)), vals, where="mid", lw=1.8, color="#2F6F9F")
    ax.scatter(range(len(vals)), vals, s=42, color="#2F6F9F", zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,}", (i, v), fontsize=8.5, xytext=(0, 8),
                    textcoords="offset points", ha="center")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("drop ", "") for l in labels],
                       rotation=18, ha="right", fontsize=8)
    style_axes(ax, "A9 · Window attrition through the Phase 1 exclusion sequence",
               None, "surviving 120s windows")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A9_attrition.png"), dpi=150)
    plt.close(fig)

    A.to_csv(os.path.join(OUT, "a9_attrition_orderA.csv"), index=False)
    B.to_csv(os.path.join(OUT, "a9_attrition_orderB.csv"), index=False)

    append_findings(
        "A9", "Attrition and rule order", "tasks/audit_a9.py",
        [("windows entering", f"{len(W):,}", "n.a.", "n.a."),
         ("windows surviving (order A)", f"{int(alive_a.sum()):,}", "n.a.", "n.a."),
         ("windows surviving (order B)", f"{int(alive_b.sum()):,}", "n.a.", "n.a."),
         ("window-level order effect", "0 (rules commute)", "unknown", "n.a."),
         ("sessions kept, length-first", len(a_sessions), "n.a.", "n.a."),
         ("sessions kept, non-wear-first", len(b_sessions), "n.a.", "n.a."),
         ("session-level order effect",
          f"{len(a_sessions - b_sessions)} sessions", "unknown", "n.a."),
         ("final level-2 events", final_ev, 178,
          "yes" if final_ev == 178 else "NO")],
        ["figures/audit/A9_attrition.png"],
        "Records the exclusion sequence and whether order is itself a judgment "
        "call. Window-level masks commute; the session-length rule does not.",
        f"{len(a_sessions - b_sessions)} sessions change status depending on "
        f"whether non-wear is removed before or after the length cutoff"
        if a_sessions - b_sessions else "rule order has no effect at either level",
    )


if __name__ == "__main__":
    main()
