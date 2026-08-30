"""PHASE 3.1 - Feature extraction, weighted toward EDA.

Section 0.2 established that skin conductance responds at event onset and heart
rate does not, so EDA carries the feature set and everything else is supporting.

The rule that governs this file: RATE-SENSITIVE FEATURES ARE EXTRACTED AT THE
NATIVE RATE, then aggregated to the window. Detecting skin-conductance responses
on a 1 Hz downsample loses ~40% of them (measured: 110 vs 184 SCRs on
5C_1587297777). The 1 Hz grid is for the label join and coarse features only.

  EDA   4 Hz   tonic mean/slope, phasic SD, SCR count, SCR mean amplitude
  HR    1 Hz   mean, SD, slope, max, delta vs trailing 30-min median
  ACC  32 Hz   magnitude mean/SD/p2p, fraction of seconds above threshold
  TEMP  4 Hz   slope and delta-from-rolling-median ONLY - never absolute
                (ICC(subject) 0.52 makes absolute temperature a fingerprint)
  ctx          hour of day, minutes into session

HRV is deliberately absent. It covers ~6% of windows and is an ablation arm,
not a baseline feature.

Writes derived/features.parquet, keyed to the frozen window table.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import neurokit2 as nk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, read_e4_header, session_dirs, banner

WINDOW_S = 120
ACC_MOVE_G = 1.05          # |acc| above this counts as movement
TEMP_ROLL_S = 1800         # 30 min rolling median for the temperature delta
HR_ROLL_S = 1800           # 30 min rolling median for the HR delta


def _read(path, name):
    """Native-rate signal with its own time base. No resampling."""
    t0, fs = read_e4_header(path)
    if name == "ACC":
        v = pd.read_csv(path, skiprows=2, header=None).values / 64.0
        return np.sqrt((v ** 2).sum(axis=1)), fs
    v = pd.read_csv(path, skiprows=2, header=None).iloc[:, 0].values
    return v, fs


def _win_index(n, fs):
    """Which 120 s window each native-rate sample belongs to."""
    return (np.arange(n) / fs // WINDOW_S).astype(int)


def _agg(vals, widx, nwin, fns):
    """Apply named aggregators per window without materialising groups."""
    df = pd.DataFrame({"w": widx, "v": vals})
    g = df.groupby("w").v
    out = {}
    for name, fn in fns.items():
        s = g.agg(fn)
        a = np.full(nwin, np.nan)
        idx = s.index.values
        k = (idx >= 0) & (idx < nwin)
        a[idx[k]] = s.values[k]
        out[name] = a
    return out


def eda_features(path, nwin):
    """Tonic/phasic decomposition and SCR detection at the native 4 Hz."""
    v, fs = _read(path, "EDA")
    fs = int(fs)
    out = {k: np.full(nwin, np.nan) for k in
           ["eda_tonic_mean", "eda_tonic_slope", "eda_phasic_sd",
            "eda_scr_count", "eda_scr_amp"]}
    if len(v) < fs * 10:
        return out
    try:
        ph = nk.eda_phasic(nk.signal_sanitize(v), sampling_rate=fs, method="highpass")
        tonic, phasic = ph["EDA_Tonic"].values, ph["EDA_Phasic"].values
    except Exception:
        return out
    widx = _win_index(len(v), fs)

    a = _agg(tonic, widx, nwin, {
        "eda_tonic_mean": "mean",
        "eda_tonic_slope": lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1),
    })
    out.update(a)
    out.update(_agg(phasic, widx, nwin, {"eda_phasic_sd": "std"}))

    # SCRs are events, so they are counted per window rather than aggregated.
    try:
        _, info = nk.eda_peaks(phasic, sampling_rate=fs)
        pk = np.asarray(info.get("SCR_Peaks", []), dtype=int)
        amp = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
        if len(pk):
            pw = (pk / fs // WINDOW_S).astype(int)
            ok = (pw >= 0) & (pw < nwin)
            cnt = np.bincount(pw[ok], minlength=nwin).astype(float)
            asum = np.bincount(pw[ok], weights=np.nan_to_num(amp[ok]), minlength=nwin)
            out["eda_scr_count"] = cnt
            with np.errstate(invalid="ignore", divide="ignore"):
                out["eda_scr_amp"] = np.where(cnt > 0, asum / np.maximum(cnt, 1), 0.0)
        else:
            out["eda_scr_count"] = np.zeros(nwin)
            out["eda_scr_amp"] = np.zeros(nwin)
    except Exception:
        pass
    return out


def hr_features(path, nwin):
    v, fs = _read(path, "HR")
    fs = int(fs) or 1
    if len(v) < 10:
        return {k: np.full(nwin, np.nan) for k in
                ["hr_mean", "hr_sd", "hr_slope", "hr_max", "hr_delta30"]}
    widx = _win_index(len(v), fs)
    out = _agg(v, widx, nwin, {
        "hr_mean": "mean", "hr_sd": "std", "hr_max": "max",
        "hr_slope": lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1),
    })
    roll = pd.Series(v).rolling(HR_ROLL_S * fs, min_periods=fs * 60).median().values
    out.update(_agg(v - roll, widx, nwin, {"hr_delta30": "mean"}))
    return out


def temp_features(path, nwin):
    """Slope and delta only. Absolute temperature is a subject fingerprint."""
    v, fs = _read(path, "TEMP")
    fs = int(fs)
    if len(v) < fs * 10:
        return {k: np.full(nwin, np.nan) for k in ["temp_slope", "temp_delta30"]}
    widx = _win_index(len(v), fs)
    out = _agg(v, widx, nwin, {
        "temp_slope": lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1)})
    roll = pd.Series(v).rolling(TEMP_ROLL_S * fs, min_periods=fs * 60).median().values
    out.update(_agg(v - roll, widx, nwin, {"temp_delta30": "mean"}))
    return out


def acc_features(path, nwin):
    mag, fs = _read(path, "ACC")
    fs = int(fs)
    if len(mag) < fs * 10:
        return {k: np.full(nwin, np.nan) for k in
                ["acc_mag_mean", "acc_mag_sd", "acc_p2p", "acc_frac_move"]}
    widx = _win_index(len(mag), fs)
    return _agg(mag, widx, nwin, {
        "acc_mag_mean": "mean", "acc_mag_sd": "std",
        "acc_p2p": lambda x: x.max() - x.min(),
        "acc_frac_move": lambda x: float((x > ACC_MOVE_G).mean()),
    })


def main():
    banner("PHASE 3.1 - NATIVE-RATE FEATURE EXTRACTION")

    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    paths = {os.path.basename(d): d for d in session_dirs()}
    sessions = sorted(W.session.unique())
    dur = S.set_index("session").dur_s
    print(f"\ncohort: {W.subject.nunique()} subjects, {len(W):,} windows, "
          f"{len(sessions)} sessions")
    print("extracting at native rates (EDA 4, HR 1, TEMP 4, ACC 32 Hz)...\n")

    parts = []
    t0 = time.time()
    for i, sess in enumerate(sessions):
        if i and i % 50 == 0:
            print(f"    {i}/{len(sessions)}  ({time.time()-t0:.0f}s)")
        sd = paths.get(sess)
        if sd is None:
            continue
        nwin = int(dur.get(sess, 0) // WINDOW_S)
        if nwin <= 0:
            continue
        rec = {"session": sess, "w": np.arange(nwin)}
        for fn, fname in [(eda_features, "EDA.csv"), (hr_features, "HR.csv"),
                          (temp_features, "TEMP.csv"), (acc_features, "ACC.csv")]:
            try:
                rec.update(fn(os.path.join(sd, fname), nwin))
            except Exception as e:
                print(f"    [warn] {sess} {fname}: {type(e).__name__}")
        parts.append(pd.DataFrame(rec))
    F = pd.concat(parts, ignore_index=True)
    print(f"\nextracted in {time.time()-t0:.0f}s")

    # log1p the level-like EDA features: 70/105 sessions are right-skewed
    for c in ["eda_tonic_mean", "eda_scr_amp"]:
        F[c] = np.log1p(F[c].clip(lower=0))

    D = W[["subject", "session", "w", "start_utc", "label"]].merge(
        F, on=["session", "w"], how="left")
    D["hour"] = pd.to_datetime(D.start_utc, utc=True).dt.tz_convert(
        "America/Chicago").dt.hour
    D["min_into_session"] = D.w * 2.0

    feat_cols = [c for c in D.columns if c not in
                 ("subject", "session", "w", "start_utc", "label")]
    print(f"\n{len(feat_cols)} features on {len(D):,} windows")

    banner("Coverage and distribution")
    cov = pd.DataFrame({
        "missing %": (100 * D[feat_cols].isna().mean()).round(2),
        "median": D[feat_cols].median().round(4),
        "IQR": (D[feat_cols].quantile(.75) - D[feat_cols].quantile(.25)).round(4),
    })
    print()
    print(cov.to_string())

    D.to_parquet(os.path.join(DERIVED, "features.parquet"), index=False)
    banner("FROZEN")
    print(f"\n  {os.path.join(DERIVED, 'features.parquet')}")
    print(f"    {len(D):,} windows x {len(feat_cols)} features")
    print("\n  No HRV: ~6% coverage, ablation arm only.")
    print("  No absolute TEMP: ICC(subject) 0.52 makes it a fingerprint.")


if __name__ == "__main__":
    main()
