"""Experiment Phase 1 : RegimeWorldModel vs flat-latent baseline.

Hypothesis : regime-aware structured latents preserve dynamical regime
information BETTER than flat uniform latent spaces.

Test :
1. Build synth dataset 3 régimes × N_per_regime, T=128.
2. Train two models on identical data :
   A. RegimeWorldModel (full MVP)
   B. FlatBaseline (same encoder + simple GRU, no descriptors, no router,
      no regime-conditional losses)
3. Evaluate via :
   - Linear probe AUROC on regime classification from pooled latent.
   - Direct regime accuracy from RegimeWorldModel.router (not for baseline).
4. Multi-seed, bootstrap CI.

Pad/crop tous samples à d_in=3.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from backend.regime_world import (
    RegimeWorldModel, make_dataset, split_train_eval,
    TrainConfig, train, encode_eval, REGIME_TO_IDX,
)


def pad_to_d_in(sample, d_in=3):
    if sample.traj.shape[1] < d_in:
        pad = np.zeros((sample.traj.shape[0], d_in - sample.traj.shape[1]),
                       dtype=sample.traj.dtype)
        sample.traj = np.concatenate([sample.traj, pad], axis=1)
    elif sample.traj.shape[1] > d_in:
        sample.traj = sample.traj[:, :d_in]
    return sample


class FlatBaseline(nn.Module):
    """Same encoder + simple GRU, no descriptors, no router, no regime losses.

    Trained only on recon + dynamics. Same param budget approximately.
    """

    def __init__(self, d_in=3, d_h=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 64), nn.SiLU(), nn.Linear(64, d_h),
        )
        self.gru = nn.GRUCell(d_h, d_h)
        self.proj_next = nn.Linear(d_h, d_h)
        self.decoder = nn.Sequential(
            nn.Linear(d_h, 64), nn.SiLU(), nn.Linear(64, d_in),
        )
        self.d_h = d_h

    def forward(self, x):
        z = self.encoder(x)
        T = z.shape[0]
        h = torch.zeros(self.d_h, device=z.device)
        z_seq = []
        z_pred = []
        for t in range(T):
            h = self.gru(z[t:t + 1], h.unsqueeze(0)).squeeze(0)
            z_seq.append(h)
            z_pred.append(self.proj_next(h))
        z_seq = torch.stack(z_seq)
        z_pred = torch.stack(z_pred)
        x_recon = self.decoder(z)
        return {"z_seq": z_seq, "z_pred": z_pred, "x_recon": x_recon}


def train_baseline(model, dataset, n_epochs=60, lr=3e-4, device="cpu"):
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        for sample in dataset:
            x = torch.from_numpy(sample.traj).float().to(device)
            x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
            out = model(x)
            l_recon = ((out["x_recon"] - x) ** 2).mean()
            l_dyn = ((out["z_pred"][:-1] - out["z_seq"][1:].detach()) ** 2).mean()
            loss = l_recon + l_dyn
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
    return model


@torch.no_grad()
def encode_baseline(model, dataset, device="cpu"):
    model.eval()
    z_pooled = []
    labels = []
    for sample in dataset:
        x = torch.from_numpy(sample.traj).float().to(device)
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        out = model(x)
        z_pooled.append(out["z_seq"].mean(dim=0).cpu().numpy())
        labels.append(REGIME_TO_IDX[sample.regime])
    return np.stack(z_pooled), np.array(labels)


def linear_probe_auroc(X_train, y_train, X_test, y_test):
    """3-class one-vs-rest AUROC via logistic regression."""
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xte = sc.transform(X_test)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(Xtr, y_train)
    probas = clf.predict_proba(Xte)
    aurocs = []
    for k in range(probas.shape[1]):
        y_bin = (y_test == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        aurocs.append(roc_auc_score(y_bin, probas[:, k]))
    preds = clf.predict(Xte)
    acc = float(accuracy_score(y_test, preds))
    return float(np.mean(aurocs)) if aurocs else 0.0, acc


def run_one_seed(seed: int, n_per_regime=20, T=128, n_epochs=40):
    print(f"\n=== SEED {seed} ===")
    ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in ds:
        pad_to_d_in(s, d_in=3)
    train_ds, eval_ds = split_train_eval(ds, train_ratio=0.7, seed=seed)
    print(f"  train={len(train_ds)} eval={len(eval_ds)}")

    # A. RegimeWorldModel — unsupervised (no regime labels in training)
    torch.manual_seed(seed); np.random.seed(seed)
    rwm = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4)
    cfg_u = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                        w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                        w_entropy=0.1, w_regime_sup=0.0)
    print(f"  training RWM unsupervised ({rwm.count_params()} params)...")
    ts = time.time()
    train(rwm, train_ds, cfg_u, verbose=False)
    t_rwm = time.time() - ts

    rwm_eval_out = encode_eval(rwm, eval_ds)
    rwm_train_out = encode_eval(rwm, train_ds)

    rwm_auc, rwm_acc = linear_probe_auroc(
        rwm_train_out["z_slow_pooled"], rwm_train_out["regime_true_idx"],
        rwm_eval_out["z_slow_pooled"], rwm_eval_out["regime_true_idx"],
    )
    r_pred = rwm_eval_out["r"].argmax(axis=1)
    direct_acc = float(np.mean(r_pred == rwm_eval_out["regime_true_idx"]))

    # A2. RegimeWorldModel — semi-supervised (w_regime_sup > 0)
    torch.manual_seed(seed); np.random.seed(seed)
    rwm_s = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4)
    cfg_s = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                        w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                        w_entropy=0.1, w_regime_sup=1.0)
    print(f"  training RWM supervised ...")
    ts = time.time()
    train(rwm_s, train_ds, cfg_s, verbose=False)
    t_rwm_s = time.time() - ts

    rwm_s_eval = encode_eval(rwm_s, eval_ds)
    rwm_s_train = encode_eval(rwm_s, train_ds)
    rwm_s_auc, rwm_s_acc = linear_probe_auroc(
        rwm_s_train["z_slow_pooled"], rwm_s_train["regime_true_idx"],
        rwm_s_eval["z_slow_pooled"], rwm_s_eval["regime_true_idx"],
    )
    r_pred_s = rwm_s_eval["r"].argmax(axis=1)
    direct_acc_s = float(np.mean(r_pred_s == rwm_s_eval["regime_true_idx"]))

    # B. FlatBaseline
    torch.manual_seed(seed); np.random.seed(seed)
    base = FlatBaseline(d_in=3, d_h=32)
    n_params_base = sum(p.numel() for p in base.parameters())
    print(f"  training FlatBaseline ({n_params_base} params)...")
    ts = time.time()
    train_baseline(base, train_ds, n_epochs=n_epochs)
    t_base = time.time() - ts

    Z_train_b, y_train_b = encode_baseline(base, train_ds)
    Z_eval_b, y_eval_b = encode_baseline(base, eval_ds)
    base_auc, base_acc = linear_probe_auroc(Z_train_b, y_train_b,
                                              Z_eval_b, y_eval_b)

    print(f"  RWM-unsup  probe AUROC={rwm_auc:.3f} acc={rwm_acc:.3f}  "
          f"router acc={direct_acc:.3f}  ({t_rwm:.1f}s)")
    print(f"  RWM-sup    probe AUROC={rwm_s_auc:.3f} acc={rwm_s_acc:.3f}  "
          f"router acc={direct_acc_s:.3f}  ({t_rwm_s:.1f}s)")
    print(f"  BASE       probe AUROC={base_auc:.3f} acc={base_acc:.3f}  "
          f"({t_base:.1f}s)")

    return {
        "seed": seed,
        "rwm_unsup_probe_auroc": rwm_auc, "rwm_unsup_router_acc": direct_acc,
        "rwm_sup_probe_auroc": rwm_s_auc, "rwm_sup_router_acc": direct_acc_s,
        "base_probe_auroc": base_auc, "base_probe_acc": base_acc,
        "rwm_unsup_time_s": t_rwm, "rwm_sup_time_s": t_rwm_s,
        "base_time_s": t_base,
    }


def main():
    print("=" * 75)
    print("REGIME-AWARE WORLD MODEL vs FLAT BASELINE — probe AUROC")
    print("=" * 75)
    rows = []
    t0 = time.time()
    for seed in range(5):
        try:
            rows.append(run_one_seed(seed))
        except Exception as e:
            print(f"  seed {seed} failed : {e}")
    print(f"\n[time] total {time.time() - t0:.1f}s")

    if not rows:
        print("no rows")
        return

    rwm_u_auc = [r["rwm_unsup_probe_auroc"] for r in rows]
    rwm_s_auc = [r["rwm_sup_probe_auroc"] for r in rows]
    base_auc = [r["base_probe_auroc"] for r in rows]
    rwm_u_router = [r["rwm_unsup_router_acc"] for r in rows]
    rwm_s_router = [r["rwm_sup_router_acc"] for r in rows]

    print("\n" + "=" * 75)
    print("AGGREGATE (mean ± std)")
    print("=" * 75)
    print(f"  RWM unsupervised  AUROC : {np.mean(rwm_u_auc):.3f} ± {np.std(rwm_u_auc):.3f}   router_acc : {np.mean(rwm_u_router):.3f} ± {np.std(rwm_u_router):.3f}")
    print(f"  RWM supervised    AUROC : {np.mean(rwm_s_auc):.3f} ± {np.std(rwm_s_auc):.3f}   router_acc : {np.mean(rwm_s_router):.3f} ± {np.std(rwm_s_router):.3f}")
    print(f"  Flat Baseline     AUROC : {np.mean(base_auc):.3f} ± {np.std(base_auc):.3f}")

    from math import erf, sqrt
    def welch(a, b):
        a, b = np.array(a), np.array(b)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = np.sqrt(va / na + vb / nb)
        if se < 1e-12: return float("nan"), 1.0
        t = (ma - mb) / se
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)

    print("\n  Welch t-tests :")
    for label, (a, b) in [
        ("RWM-unsup vs Base",   (rwm_u_auc, base_auc)),
        ("RWM-sup vs Base",     (rwm_s_auc, base_auc)),
        ("RWM-sup vs RWM-unsup",(rwm_s_auc, rwm_u_auc)),
    ]:
        t, p = welch(a, b)
        sig = "yes (*)" if p < 0.05 else "no"
        print(f"    {label:<26} t={t:+.2f}  p={p:.4f}  {sig}")

    print("\n  Router accuracy comparison (1/3 = random) :")
    print(f"    RWM-unsup : {np.mean(rwm_u_router):.3f}  (chance = 0.33)")
    print(f"    RWM-sup   : {np.mean(rwm_s_router):.3f}  (chance = 0.33)")
    t_router, p_router = welch(rwm_s_router, rwm_u_router)
    print(f"    Welch RWM-sup vs RWM-unsup router : t={t_router:+.2f}, p={p_router:.4f}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "rows": rows,
        "agg": {
            "rwm_unsup_auroc_mean": float(np.mean(rwm_u_auc)),
            "rwm_sup_auroc_mean": float(np.mean(rwm_s_auc)),
            "base_auroc_mean": float(np.mean(base_auc)),
            "rwm_unsup_router_mean": float(np.mean(rwm_u_router)),
            "rwm_sup_router_mean": float(np.mean(rwm_s_router)),
        },
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_probe.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/regime_world_probe.json")


if __name__ == "__main__":
    main()
