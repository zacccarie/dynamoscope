"""Re-eval multi-step rollout après F10 fix.

Trois variantes en parallèle :
  - RWM-orig    : stable=False, w_multistep=0  (le modèle du papier, qui diverge)
  - RWM-stable  : stable=True (spectral_norm + tanh-bounded delta)
  - RWM-stable-ms : stable=True + w_multistep=2 (full fix)
  - Flat-56     : baseline matched-capacity

Pour chaque K ∈ {1,2,4,8,16,24}, MSE per-step. Sortie : JSON + tableau.
Si RWM-stable-ms se rapproche du baseline plat à K=24 et garde son avantage
1-step → F10 fix validé → §10.9 du papier.
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

from backend.regime_world import (
    RegimeWorldModel, TrainConfig, train, make_dataset,
)
from experiments.regime_world_probe import (
    FlatBaseline, train_baseline, pad_to_d_in,
)
from experiments.regime_world_rollout import eval_rollout


def train_rwm(seed: int, train_ds, n_epochs: int, *,
              stable: bool, w_multistep: float):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=stable)
    cfg = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                       w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                       w_entropy=0.0, w_regime_sup=5.0,
                       w_multistep=w_multistep, K_multistep=4,
                       gamma_multistep=0.9, n_starts_multistep=4)
    train(model, train_ds, cfg, verbose=False)
    return model


def run_one_seed(seed: int, n_per_regime=12, n_epochs=40, T=128, K_max=24):
    print(f"\n=== SEED {seed} ===")
    train_ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in train_ds:
        pad_to_d_in(s, 3)
    eval_ds = make_dataset(n_per_regime=8, T=T, base_seed=seed * 100 + 5000)
    for s in eval_ds:
        pad_to_d_in(s, 3)

    results = {"seed": seed}

    # 1) RWM-orig (baseline du papier, diverge)
    ts = time.time()
    rwm_orig = train_rwm(seed, train_ds, n_epochs,
                          stable=False, w_multistep=0.0)
    print(f"  RWM-orig trained ({time.time() - ts:.1f}s)")
    r = eval_rollout(rwm_orig, eval_ds, K_max=K_max, kind="rwm")
    results["rwm_orig_mse_per_step"] = r["mse_per_step"]

    # 2) RWM-stable (juste spectral_norm + tanh, pas de multi-step loss)
    ts = time.time()
    rwm_stab = train_rwm(seed, train_ds, n_epochs,
                          stable=True, w_multistep=0.0)
    print(f"  RWM-stable trained ({time.time() - ts:.1f}s)")
    r = eval_rollout(rwm_stab, eval_ds, K_max=K_max, kind="rwm")
    results["rwm_stable_mse_per_step"] = r["mse_per_step"]

    # 3) RWM-stable-ms (spectral_norm + tanh + multi-step rollout loss)
    ts = time.time()
    rwm_ms = train_rwm(seed, train_ds, n_epochs,
                        stable=True, w_multistep=2.0)
    print(f"  RWM-stable+ms trained ({time.time() - ts:.1f}s)")
    r = eval_rollout(rwm_ms, eval_ds, K_max=K_max, kind="rwm")
    results["rwm_stable_ms_mse_per_step"] = r["mse_per_step"]

    # 4) Flat-56 baseline (matched capacity)
    torch.manual_seed(seed); np.random.seed(seed)
    flat56 = FlatBaseline(d_in=3, d_h=56)
    ts = time.time()
    train_baseline(flat56, train_ds, n_epochs=n_epochs)
    print(f"  Flat-56 trained ({time.time() - ts:.1f}s)")
    r = eval_rollout(flat56, eval_ds, K_max=K_max, kind="flat")
    results["flat56_mse_per_step"] = r["mse_per_step"]

    return results


def main():
    print("=" * 80)
    print("F10 FIX : multi-step rollout — RWM-orig vs RWM-stab vs RWM-stab+ms vs Flat-56")
    print("=" * 80)

    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "3"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    N_EPOCHS = int(os.environ.get("RWM_N_EPOCHS", "40"))

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, n_epochs=N_EPOCHS, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} failed : {e}")
            import traceback; traceback.print_exc()
    print(f"\n[time] total {time.time() - t0:.1f}s")

    if not all_runs:
        return

    orig_mat = np.array([r["rwm_orig_mse_per_step"] for r in all_runs])
    stab_mat = np.array([r["rwm_stable_mse_per_step"] for r in all_runs])
    ms_mat = np.array([r["rwm_stable_ms_mse_per_step"] for r in all_runs])
    flat_mat = np.array([r["flat56_mse_per_step"] for r in all_runs])

    orig_mean = orig_mat.mean(axis=0)
    stab_mean = stab_mat.mean(axis=0)
    ms_mean = ms_mat.mean(axis=0)
    flat_mean = flat_mat.mean(axis=0)

    print("\n" + "=" * 90)
    print("AGGREGATE per-step MSE (mean across seeds)")
    print("=" * 90)
    header = f"  {'K':>3}{'RWM-orig':>14}{'RWM-stab':>14}{'RWM-stab+ms':>14}{'Flat-56':>14}{'ms/F56':>10}"
    print(header)
    print("-" * 90)
    for k in [0, 1, 3, 7, 15, K_MAX - 1]:
        if k >= len(orig_mean):
            continue
        o = orig_mean[k]; s = stab_mean[k]; m = ms_mean[k]; f = flat_mean[k]
        ratio = m / max(f, 1e-9)
        print(f"  {k+1:>3}{o:>14.3e}{s:>14.3e}{m:>14.3e}{f:>14.3e}{ratio:>10.3f}")

    # F10-fix verdict
    print("\n" + "=" * 90)
    print("F10 FIX VERDICT")
    print("=" * 90)
    orig_K1 = orig_mean[0]; orig_Kmax = orig_mean[-1]
    ms_K1 = ms_mean[0]; ms_Kmax = ms_mean[-1]
    flat_K1 = flat_mean[0]; flat_Kmax = flat_mean[-1]
    print(f"  RWM-orig    K=1 : {orig_K1:.3e}   K={K_MAX} : {orig_Kmax:.3e}   "
          f"degradation: {orig_Kmax / max(orig_K1, 1e-9):.2e}×")
    print(f"  RWM-stab+ms K=1 : {ms_K1:.3e}   K={K_MAX} : {ms_Kmax:.3e}   "
          f"degradation: {ms_Kmax / max(ms_K1, 1e-9):.2e}×")
    print(f"  Flat-56     K=1 : {flat_K1:.3e}   K={K_MAX} : {flat_Kmax:.3e}   "
          f"degradation: {flat_Kmax / max(flat_K1, 1e-9):.2e}×")

    if ms_Kmax < flat_Kmax:
        verdict = "✓✓ F10 FIX SUPERIOR : RWM-stab+ms beats Flat at long horizon"
    elif ms_Kmax < 2 * flat_Kmax:
        verdict = "✓ F10 FIX VALIDATED : RWM-stab+ms within 2× of Flat at long horizon"
    elif ms_Kmax < 100 * flat_Kmax:
        verdict = "≈ F10 FIX PARTIAL : RWM-stab+ms bounded but worse than Flat"
    else:
        verdict = "✗ F10 FIX FAILED : RWM-stab+ms still diverges"
    print(f"\n  {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "all_runs": all_runs,
        "orig_mean_per_step": orig_mean.tolist(),
        "stab_mean_per_step": stab_mean.tolist(),
        "ms_mean_per_step": ms_mean.tolist(),
        "flat_mean_per_step": flat_mean.tolist(),
        "K_max": K_MAX,
        "verdict": verdict,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_rollout_fixed.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_rollout_fixed.json")


if __name__ == "__main__":
    main()
