"""Router fix attempt — Phase 2 follow-up.

Phase 2 diag established :
- Descriptors are massively separable across regimes (sep ratio 10-175×)
- Router MLP fails to use them (33% accuracy = random)

3-step diagnostic + fix :

STEP 1 : Upper bound via linear probe on descriptor outputs.
         If linear classifier on d_dyn achieves >>33%, confirms info
         exists in descriptor space.

STEP 2 : Try router fix configurations :
         - baseline : w_entropy=0.1, w_regime_sup=1.0 (current)
         - fix_A    : w_entropy=0.01 (10× reduce) + w_regime_sup=3.0
         - fix_B    : w_entropy=0.0 (off) + w_regime_sup=5.0
         - fix_C    : w_entropy=0.0 + w_regime_sup=10.0

STEP 3 : Compare router accuracy + MSE per config.

Goal : restore router accuracy without breaking MSE advantage.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from backend.regime_world import RegimeWorldModel, TrainConfig, train, REGIME_TO_IDX
from backend.regime_world.video_data import build_windowed_dataset
from experiments.regime_world_video import (
    FlatVideoBaseline, train_baseline_video, train_rwm_video,
    eval_sample, split_by_regime,
)


CONFIGS = {
    "baseline":  {"w_entropy": 0.1,  "w_regime_sup": 1.0},
    "fix_A":     {"w_entropy": 0.01, "w_regime_sup": 3.0},
    "fix_B":     {"w_entropy": 0.0,  "w_regime_sup": 5.0},
    "fix_C":     {"w_entropy": 0.0,  "w_regime_sup": 10.0},
}


@torch.no_grad()
def collect_descriptors(model, samples):
    """Extract d_dyn + true regime label per sample."""
    model.eval()
    d_list = []
    y_list = []
    for s in samples:
        x = torch.from_numpy(s.features).float()
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        out = model(x)
        d_list.append(out["d_dyn"].cpu().numpy())
        y_list.append(REGIME_TO_IDX[s.regime])
    return np.stack(d_list), np.array(y_list)


def linear_probe(X_train, y_train, X_test, y_test):
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xte = sc.transform(X_test)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, y_train)
    return float(accuracy_score(y_test, clf.predict(Xte)))


def train_with_config(seed, train_ds, n_epochs, d_in, cfg_dict):
    torch.manual_seed(seed); np.random.seed(seed)
    rwm = RegimeWorldModel(d_in=d_in, d_fast=64, d_slow=64, slow_stride=4)
    cfg = TrainConfig(
        n_epochs=n_epochs, lr=3e-4, device="cpu",
        w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
        w_entropy=cfg_dict["w_entropy"],
        w_regime_sup=cfg_dict["w_regime_sup"],
    )
    train(rwm, train_ds, cfg, verbose=False)
    return rwm


def eval_router(model, samples):
    correct = 0
    mses = []
    for s in samples:
        r = eval_sample(model, s, kind="rwm")
        mses.append(r["mse"])
        true_idx = REGIME_TO_IDX[s.regime]
        if int(np.argmax(r["router"])) == true_idx:
            correct += 1
    return correct / max(len(samples), 1), float(np.mean(mses))


def main():
    print("=" * 80)
    print("ROUTER FIX — Phase 2 follow-up")
    print("=" * 80)

    # Build dataset (cached features re-used)
    print("\n[setup] building dataset (cached DINOv2 features)...")
    samples = build_windowed_dataset(
        n_videos_per_regime=4, n_frames_per_video=128,
        window=64, stride=16, seed_base=0, verbose=True,
    )
    train_ds, eval_ds = split_by_regime(samples, train_ratio=0.7, seed=0)
    d_in = train_ds[0].features.shape[1]
    print(f"  train={len(train_ds)} eval={len(eval_ds)} d_in={d_in}")

    t0 = time.time()

    # STEP 1 : linear probe on descriptors from baseline model
    print("\n" + "=" * 80)
    print("STEP 1 : linear probe upper bound on descriptors")
    print("=" * 80)
    baseline_rwm = train_with_config(0, train_ds, n_epochs=40, d_in=d_in,
                                       cfg_dict=CONFIGS["baseline"])
    Xtr, ytr = collect_descriptors(baseline_rwm, train_ds)
    Xte, yte = collect_descriptors(baseline_rwm, eval_ds)
    probe_acc = linear_probe(Xtr, ytr, Xte, yte)
    print(f"  Linear probe accuracy on descriptor space : {probe_acc:.3f}")
    print(f"  (vs chance 0.33. Higher = router has tappable info available)")

    # STEP 2 : try fixes
    print("\n" + "=" * 80)
    print("STEP 2 : router config comparison (single seed = 0 for speed)")
    print("=" * 80)
    rows = []
    for cfg_name, cfg_dict in CONFIGS.items():
        rwm = train_with_config(0, train_ds, n_epochs=40, d_in=d_in, cfg_dict=cfg_dict)
        router_acc, mse = eval_router(rwm, eval_ds)
        rows.append({"config": cfg_name, **cfg_dict,
                      "router_acc": router_acc, "mse": mse})
        print(f"  {cfg_name:<12}  entropy={cfg_dict['w_entropy']:.3f}  "
              f"sup={cfg_dict['w_regime_sup']:.1f}  "
              f"router={router_acc:.3f}  mse={mse:.4f}")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Linear probe ceiling (descriptor info)     : {probe_acc:.3f}")
    print(f"  Router accuracy by config :")
    for r in rows:
        gap = r["router_acc"] - 0.33
        print(f"    {r['config']:<12}  acc={r['router_acc']:.3f}  "
              f"Δ chance={gap:+.3f}  mse={r['mse']:.4f}")

    best = max(rows, key=lambda x: x["router_acc"])
    print(f"\n  Best config : {best['config']}  router_acc={best['router_acc']:.3f}")
    if best["router_acc"] > 0.6:
        print(f"  ✓ ROUTER FIXED")
    elif best["router_acc"] > 0.4:
        print(f"  ≈ PARTIAL FIX (above chance but below probe ceiling)")
    else:
        print(f"  ✗ NO FIX — router cannot use info even with config changes")

    Path("results").mkdir(exist_ok=True)
    out = {
        "linear_probe_acc": probe_acc,
        "configs": rows,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_router_fix.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/regime_world_router_fix.json")


if __name__ == "__main__":
    main()
