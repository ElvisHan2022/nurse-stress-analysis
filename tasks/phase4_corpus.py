"""PHASE 4.1 - Build the 1 Hz pretraining corpus.

Self-supervision has no label requirement, so this uses ALL 609 sessions and
all 15 participants, including the five excluded from the supervised cohort.
That is the point: the corpus is 1,251.7 hours against 138 labelled episodes,
which is the asymmetry the whole phase exists to exploit.

Channels, per the plan: eda, hr, acc_mag, acc_sd, temp_delta. Absolute
temperature is excluded because its intraclass correlation with participant
identity is 0.52 and an unsupervised objective would seize on it.

Each channel is causally normalised within session before storage, so the
encoder never sees raw units and cannot learn a participant's baseline offset
as a shortcut.

Writes derived/pretrain_corpus.npz
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (OUT, DERIVED, read_e4_header, session_dirs, banner,
                          causal_z_safe)

CHANNELS = ["eda", "hr", "acc_mag", "acc_sd", "temp_delta"]
TEMP_ROLL_S = 1800
EDA_FLOOR = 0.05
STILL_SD = 0.005


def read_1hz(sd, name):
    """Native-rate read, resampled to a 1 Hz grid keyed on absolute time."""
    p = os.path.join(sd, name + ".csv")
    if not os.path.exists(p) or os.path.getsize(p) < 40:
        return None
    t0, fs = read_e4_header(p)
    if name == "ACC":
        v = pd.read_csv(p, skiprows=2, header=None).values / 64.0
        mag = np.sqrt((v ** 2).sum(axis=1))
        idx = pd.to_datetime(t0 + np.arange(len(mag)) / fs, unit="s", utc=True)
        s = pd.Series(mag, index=idx).resample("1s")
        # within-second SD is computed at 32 Hz BEFORE averaging; averaging
        # first would destroy the motion energy this channel exists to carry
        return s.mean(), s.std()
    v = pd.read_csv(p, skiprows=2, header=None).iloc[:, 0].values
    idx = pd.to_datetime(t0 + np.arange(len(v)) / fs, unit="s", utc=True)
    return pd.Series(v, index=idx).resample("1s").mean()


def build_session(sd):
    eda = read_1hz(sd, "EDA")
    hr = read_1hz(sd, "HR")
    temp = read_1hz(sd, "TEMP")
    acc = read_1hz(sd, "ACC")
    if eda is None or acc is None:
        return None
    acc_mag, acc_sd = acc

    df = pd.DataFrame({"eda": eda, "acc_mag": acc_mag, "acc_sd": acc_sd})
    if hr is not None:
        df["hr"] = hr
    else:
        df["hr"] = np.nan
    if temp is not None:
        roll = temp.rolling(f"{TEMP_ROLL_S}s", min_periods=60).median()
        df["temp_delta"] = temp - roll
    else:
        df["temp_delta"] = np.nan

    # non-wear: skin conductance at the floor AND the wrist still
    nonwear = (df.eda < EDA_FLOOR) & (df.acc_sd < STILL_SD)
    df = df[~nonwear.fillna(False)]
    if len(df) < 300:
        return None

    out = pd.DataFrame(index=df.index)
    for c in CHANNELS:
        out[c] = causal_z_safe(df[c], window="60min", min_periods=600)
    return out[CHANNELS].astype(np.float32)


def main():
    banner("PHASE 4.1 - PRETRAINING CORPUS")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}
    print(f"\n{len(S)} sessions, {S.subject.nunique()} participants, "
          f"{S.dur_s.sum()/3600:,.0f} h")
    print("all participants included; pretraining has no label requirement\n")

    blocks, meta = [], []
    t0 = time.time()
    for i, r in enumerate(S.itertuples()):
        if i and i % 100 == 0:
            print(f"    {i}/{len(S)}  ({time.time()-t0:.0f}s)")
        sd = paths.get(r.session)
        if sd is None:
            continue
        try:
            d = build_session(sd)
        except Exception as e:
            print(f"    [warn] {r.session}: {type(e).__name__}")
            continue
        if d is None or len(d) < 300:
            continue
        blocks.append(d.values)
        meta.append({"subject": r.subject, "session": r.session,
                     "n": len(d), "start": d.index[0].value})

    X = np.concatenate(blocks, axis=0)
    M = pd.DataFrame(meta)
    M["offset"] = np.concatenate([[0], np.cumsum(M.n.values)[:-1]])
    print(f"\nbuilt in {time.time()-t0:.0f}s")
    print(f"corpus  {X.shape[0]:,} seconds x {X.shape[1]} channels "
          f"= {X.nbytes/1e6:.0f} MB")
    print(f"        {X.shape[0]/3600:,.0f} usable hours from {len(M)} sessions")

    banner("Channel coverage after non-wear removal and normalisation")
    cov = pd.DataFrame({
        "NaN %": (100 * np.isnan(X).mean(axis=0)).round(2),
        "median": np.nanmedian(X, axis=0).round(3),
        "p1": np.nanpercentile(X, 1, axis=0).round(2),
        "p99": np.nanpercentile(X, 99, axis=0).round(2),
    }, index=CHANNELS)
    print()
    print(cov.to_string())
    print("\np1/p99 inside the clip bounds confirms causal_z_safe is holding.")

    np.savez_compressed(os.path.join(DERIVED, "pretrain_corpus.npz"),
                        X=X, channels=np.array(CHANNELS))
    M.to_parquet(os.path.join(DERIVED, "pretrain_index.parquet"), index=False)
    banner("FROZEN")
    print(f"\n  derived/pretrain_corpus.npz   {X.shape}")
    print(f"  derived/pretrain_index.parquet {len(M)} sessions")


if __name__ == "__main__":
    main()
