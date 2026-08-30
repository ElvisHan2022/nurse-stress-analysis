"""PHASE 4.2-4.3 - Masked reconstruction, then the subject-shortcut probe.

PRETEXT TASK. Mask random 30-120 s spans across all channels and reconstruct
them from the surrounding context, with MSE loss computed on the masked
positions only. Chosen over contrastive learning deliberately: standard
time-series augmentations include amplitude scaling, and EDA amplitude IS the
signal here, so a scale-invariant objective would train the encoder to discard
exactly what Phase 0 showed is the only responsive channel.

THE GATE. Freeze the encoder, fit a linear classifier on its embeddings to
predict participant identity, and compare against 0.768, the accuracy already
reachable from sensible hand-crafted features. NOT against chance at 0.100.
Physiology genuinely differs between people, so some recoverability is
expected and is not by itself a failure. An encoder near 0.95 has learned
identity rather than physiology and the phase stops.

LEAKAGE. Pretraining on all participants and then evaluating leave-one-out is
transductive. We pretrain once and label the result an upper bound, which is
what most published wearable-SSL work does without saying so. Per-fold
pretraining is the clean alternative and costs ten times the compute.

Writes derived/encoder.pt and reports the probe.
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import DERIVED, OUT, banner

WIN = 120
HOP = 60
N_CH = 5
LATENT = 64
EPOCHS = 12
BATCH = 256
LR = 1e-3
MASK_MIN, MASK_MAX = 30, 120     # fixed a priori, not tuned on downstream
SEED = 0
HANDCRAFTED_PROBE = 0.768        # the reference the gate compares against


class Encoder(nn.Module):
    """Small dilated 1D CNN. Kept deliberately low-capacity: 4.5M timesteps is
    a modest corpus and a large encoder would memorise participants."""

    def __init__(self, n_ch=N_CH, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(64, latent, 5, padding=16, dilation=8), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent=LATENT, n_ch=N_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, n_ch, 1),
        )

    def forward(self, z):
        return self.net(z)


def make_windows(X, index):
    """Non-overlapping-in-session windows, dropping any that contain NaN."""
    out, sub = [], []
    for r in index.itertuples():
        block = X[r.offset:r.offset + r.n]
        for s in range(0, len(block) - WIN + 1, HOP):
            w = block[s:s + WIN]
            if not np.isnan(w).any():
                out.append(w)
                sub.append(r.subject)
    return np.stack(out).astype(np.float32), np.array(sub)


def mask_batch(x, rng):
    """Zero a random 30-120 s span per example, across all channels."""
    b, c, t = x.shape
    m = torch.ones_like(x)
    for i in range(b):
        span = int(rng.integers(MASK_MIN, MASK_MAX + 1))
        span = min(span, t)
        st = int(rng.integers(0, t - span + 1))
        m[i, :, st:st + span] = 0.0
    return x * m, (1.0 - m)


def main():
    banner("PHASE 4.2 - MASKED RECONSTRUCTION PRETRAINING")

    d = np.load(os.path.join(DERIVED, "pretrain_corpus.npz"))
    X = d["X"]
    idx = pd.read_parquet(os.path.join(DERIVED, "pretrain_index.parquet"))
    print(f"\ncorpus {X.shape[0]:,} s x {X.shape[1]} channels")

    W, subj = make_windows(X, idx)
    print(f"windows {W.shape[0]:,} of {WIN} s, complete cases only")
    print(f"participants represented: {len(np.unique(subj))}")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    Wt = torch.tensor(W).permute(0, 2, 1)          # (N, C, T)

    enc, dec = Encoder(), Decoder()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=LR, weight_decay=1e-5)
    n = Wt.shape[0]

    print(f"\nencoder params {sum(p.numel() for p in enc.parameters()):,}")
    print(f"mask span {MASK_MIN}-{MASK_MAX} s, fixed a priori\n")

    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = cnt = 0.0
        for i in range(0, n, BATCH):
            xb = Wt[perm[i:i + BATCH]]
            xm, mask = mask_batch(xb, rng)
            opt.zero_grad()
            rec = dec(enc(xm))
            # loss on masked positions only
            loss = ((rec - xb) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
            cnt += xb.shape[0]
        print(f"  epoch {ep+1:2d}/{EPOCHS}  masked MSE {tot/cnt:.4f}  "
              f"({time.time()-t0:.0f}s)")

    torch.save({"encoder": enc.state_dict(),
                "config": {"win": WIN, "latent": LATENT, "n_ch": N_CH}},
               os.path.join(DERIVED, "encoder.pt"))
    print(f"\nsaved derived/encoder.pt after {time.time()-t0:.0f}s")

    # ---- 4.3 the gate ------------------------------------------------------
    banner("PHASE 4.3 - SUBJECT-SHORTCUT PROBE (MANDATORY GATE)")

    enc.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, n, 512):
            z = enc(Wt[i:i + 512])
            embs.append(z.mean(dim=2).numpy())     # pool over time
    E = np.concatenate(embs)
    y = LabelEncoder().fit_transform(subj)
    print(f"\nembeddings {E.shape}, {len(np.unique(y))} participants")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    acc = cross_val_score(clf, E, y, cv=StratifiedKFold(3, shuffle=True,
                                                        random_state=0),
                          scoring="accuracy").mean()
    chance = 1.0 / len(np.unique(y))

    print(f"""
  linear probe accuracy      {acc:.4f}
  chance                     {chance:.4f}
  hand-crafted reference     {HANDCRAFTED_PROBE:.4f}
""")
    if acc >= 0.95:
        verdict = "FAIL - the encoder learned identity. Phase 4 stops here."
    elif acc > HANDCRAFTED_PROBE:
        verdict = (f"CAUTION - above the hand-crafted reference by "
                   f"{acc - HANDCRAFTED_PROBE:+.4f}. Proceed but report it.")
    else:
        verdict = (f"PASS - at or below the hand-crafted reference "
                   f"({acc - HANDCRAFTED_PROBE:+.4f}).")
    print(f"  {verdict}\n")

    pd.DataFrame([{"probe_accuracy": round(float(acc), 4),
                   "chance": round(chance, 4),
                   "handcrafted_reference": HANDCRAFTED_PROBE,
                   "verdict": verdict}]).to_csv(
        os.path.join(OUT, "phase4_subject_probe.csv"), index=False)

    print("Reminder: this encoder saw every participant, so any downstream")
    print("leave-one-out result using it is TRANSDUCTIVE and is an upper")
    print("bound, not an estimate of performance on an unseen person.")


if __name__ == "__main__":
    main()
