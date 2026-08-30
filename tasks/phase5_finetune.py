"""PHASE 5 - Fine-tuning, and the label-efficiency sweep.

The sweep is the deliverable, not the head-to-head score. At 138 episodes a
pretrained encoder beating hand-crafted features outright is unlikely, and a
single number at full label count would not tell us why. The informative
question is whether pretraining reaches a given level with FEWER episodes.

  separation at low counts that closes by 138  -> pretraining buys label
                                                  efficiency, and the crossover
                                                  tells the next study how many
                                                  labels it needs
  no separation anywhere                       -> a reportable negative about
                                                  self-supervision at this scale

Both answers speak directly to the paper's question, which is whether the
constraint is the labels or the representation.

Three arms on identical folds and identical windows:
  1. linear probe on frozen pretrained embeddings
  2. full fine-tune, encoder at one tenth the head's learning rate
  3. the same architecture from random initialisation, which is the control
     that isolates what pretraining contributed as opposed to the architecture

Everything here is TRANSDUCTIVE: the encoder saw all participants during
pretraining. Results are upper bounds.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (DERIVED, OUT, read_e4_header, session_dirs, banner,
                          causal_z_safe)
from phase4_pretrain import Encoder, WIN, LATENT, N_CH
from phase4_corpus import build_session, CHANNELS

EPISODE_GRID = [25, 50, 100, 138]
FT_EPOCHS = 25
SEED = 0
EVENT_FIRE_FRAC = 0.50
FA_PER_HOUR = 1.0


class Head(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, z):
        return self.fc(z.mean(dim=2)).squeeze(-1)


def build_labelled_windows():
    """Raw 120 s x 5 channel tensors for the labelled cohort, keyed to the
    frozen label table so folds match Phase 3 exactly."""
    cache = os.path.join(DERIVED, "phase5_windows.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], pd.DataFrame(d["meta"], columns=d["cols"])

    L = pd.read_parquet(os.path.join(DERIVED, "labels_seed0.parquet"))
    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True)
    paths = {os.path.basename(d): d for d in session_dirs()}

    need = sorted(L.session.unique())
    print(f"  reading {len(need)} sessions at 1 Hz...")
    sess_cache = {}
    t0 = time.time()
    for i, s in enumerate(need):
        if i and i % 50 == 0:
            print(f"    {i}/{len(need)} ({time.time()-t0:.0f}s)")
        sd = paths.get(s)
        if sd is None:
            continue
        try:
            sess_cache[s] = build_session(sd)
        except Exception:
            pass

    rows, X = [], []
    for r in L.itertuples():
        d = sess_cache.get(r.session)
        if d is None:
            continue
        lo = r.w * 120
        blk = d.values[lo:lo + WIN]
        if blk.shape[0] != WIN or np.isnan(blk).any():
            continue
        X.append(blk)
        rows.append((r.subject, r.session, r.w, r.label))
    X = np.stack(X).astype(np.float32)
    M = pd.DataFrame(rows, columns=["subject", "session", "w", "label"])

    # attach event ids so episode-level recall matches Phase 3
    EV = pd.read_parquet(os.path.join(DERIVED, "events.parquet"))
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True)
    W["event_id"] = -1
    for i, e in EV.iterrows():
        ov = ((np.minimum(W.end_utc, e.end_utc)
               - np.maximum(W.start_utc, e.start_utc)).dt.total_seconds())
        W.loc[(W.subject == e.subject) & (ov >= 60), "event_id"] = i
    M = M.merge(W[["subject", "session", "w", "event_id"]],
                on=["subject", "session", "w"], how="left")
    M["event_id"] = M.event_id.fillna(-1).astype(int)

    np.savez_compressed(cache, X=X, meta=M.values, cols=np.array(M.columns))
    return X, M


def episode_recall(y, score, ev, neg_per_hour, fa=FA_PER_HOUR):
    negs = np.sort(score[y == 0])[::-1]
    if not len(negs):
        return np.nan
    k = max(int(round((fa / neg_per_hour) * len(negs))) - 1, 0)
    pred = (score >= negs[min(k, len(negs) - 1)]).astype(int)
    d = pd.DataFrame({"ev": ev, "pred": pred, "y": y})
    d = d[(d.y == 1) & (d.ev >= 0)]
    return float((d.groupby("ev").pred.mean() >= EVENT_FIRE_FRAC).mean()) \
        if len(d) else np.nan


def fit_arm(Xtr, ytr, Xte, arm, seed=SEED):
    """arm in {probe, finetune, scratch}."""
    torch.manual_seed(seed)
    enc = Encoder()
    if arm in ("probe", "finetune"):
        ck = torch.load(os.path.join(DERIVED, "encoder.pt"), weights_only=True)
        enc.load_state_dict(ck["encoder"])

    xtr = torch.tensor(Xtr).permute(0, 2, 1)
    xte = torch.tensor(Xte).permute(0, 2, 1)

    if arm == "probe":
        enc.eval()
        with torch.no_grad():
            ztr = enc(xtr).mean(dim=2).numpy()
            zte = enc(xte).mean(dim=2).numpy()
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=0.1,
                                               class_weight="balanced"))
        clf.fit(ztr, ytr)
        return clf.predict_proba(zte)[:, 1]

    head = Head()
    enc_lr = 1e-4 if arm == "finetune" else 1e-3
    opt = torch.optim.Adam([{"params": enc.parameters(), "lr": enc_lr},
                            {"params": head.parameters(), "lr": 1e-3}],
                           weight_decay=1e-4)
    w = torch.tensor(len(ytr) / (2 * max(ytr.sum(), 1)), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    yt = torch.tensor(ytr, dtype=torch.float32)

    enc.train(); head.train()
    for _ in range(FT_EPOCHS):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(head(enc(xtr[b])), yt[b])
            loss.backward()
            opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(enc(xte))).numpy()


def main():
    banner("PHASE 5 - FINE-TUNING AND LABEL EFFICIENCY")

    if not os.path.exists(os.path.join(DERIVED, "encoder.pt")):
        print("\nencoder.pt not found. Run tasks/phase4_pretrain.py first.")
        return

    X, M = build_labelled_windows()
    M["label"] = M.label.astype(int)
    M["event_id"] = M.event_id.astype(int)
    print(f"\nlabelled windows {X.shape}, {M.subject.nunique()} participants, "
          f"{int(M.label.sum())} positive")

    Wall = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    neg_per_hour = (3600 / 120) * (1 - float(Wall.label.mean()))
    subs = np.array(sorted(M.subject.unique()))
    all_ev = sorted(M.loc[M.label == 1, "event_id"].unique())
    all_ev = [e for e in all_ev if e >= 0]
    print(f"episodes represented: {len(all_ev)}")

    rng = np.random.default_rng(SEED)
    rows = []
    for n_ep in EPISODE_GRID:
        keep_ev = set(all_ev) if n_ep >= len(all_ev) else \
            set(rng.choice(all_ev, size=n_ep, replace=False).tolist())
        for arm in ["probe", "finetune", "scratch"]:
            oof = np.full(len(M), np.nan)
            t0 = time.time()
            for held in subs:
                te = (M.subject == held).values
                tr = ~te
                # restrict TRAINING positives to the sampled episodes
                drop = tr & (M.label == 1).values & \
                    (~M.event_id.isin(keep_ev)).values
                tr = tr & ~drop
                if M.label.values[tr].sum() < 5:
                    continue
                oof[te] = fit_arm(X[tr], M.label.values[tr], X[te], arm)
            ok = np.isfinite(oof)
            auc = roc_auc_score(M.label.values[ok], oof[ok])
            rec = episode_recall(M.label.values[ok], oof[ok],
                                 M.event_id.values[ok], neg_per_hour)
            rows.append({"episodes": n_ep, "arm": arm,
                         "AUC": round(auc, 4),
                         "episode recall": round(rec, 4),
                         "sec": round(time.time() - t0)})
            print(f"  {n_ep:>3} episodes  {arm:9}  AUC {auc:.4f}  "
                  f"recall {rec:.4f}  ({time.time()-t0:.0f}s)")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "phase5_label_efficiency.csv"), index=False)

    banner("LABEL-EFFICIENCY SWEEP")
    for metric in ["AUC", "episode recall"]:
        print(f"\n{metric}")
        print(R.pivot(index="episodes", columns="arm",
                      values=metric).to_string())

    banner("READING")
    piv = R.pivot(index="episodes", columns="arm", values="AUC")
    sep = (piv["probe"] - piv["scratch"]).round(4)
    print(f"""
  pretrained probe minus random init, by episode count:
{sep.to_string()}

  Separation at low counts that closes by 138 means pretraining buys label
  efficiency. Flat or negative separation everywhere is a negative result
  about self-supervision at this data scale, which is the more likely outcome
  and is still worth reporting.

  Phase 3 baseline for comparison: AUC 0.7654, episode recall 0.1063.
  All numbers here are TRANSDUCTIVE upper bounds.
""")


if __name__ == "__main__":
    main()
