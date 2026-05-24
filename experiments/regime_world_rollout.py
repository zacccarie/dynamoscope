"""Multi-step rollout evaluation — addresses 1-step circularity.

Phase 1+2 evaluations were all 1-step latent MSE = model's own training
objective. Partially circular. This experiment measures K-step rollout
fidelity for K ∈ {1, 2, 4, 8, 16, 32}.

For each K :
- Encode context (first 16 z_slow steps)
- Predict next K z_slow steps autoregressively
- Compare to ground truth z_slow (encoded from full trajectory)
- Report MSE_k per step + cumulative

Compare RWM-sup vs Flat baseline on mixed-regime synth zoo.

Hypothesis (H2) : RWM-sup degrades MORE GRACEFULLY than Flat as K grows.
If H2 holds : regime-aware architecture is a better WORLD model not just
a better 1-step predictor.
If H2 fails : 1-step advantage doesn't transfer to multi-step → the
28× headline is partially artefactual.
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

from backend.regime_world import (
    RegimeWorldModel, TrainConfig, train, REGIME_TO_IDX, make_dataset,
)
from experiments.regime_world_probe import (
    FlatBaseline, train_baseline, pad_to_d_in,
)


def flat_rollout(model: FlatBaseline, x_context: torch.Tensor, K: int) -> torch.Tensor:
    """K-step rollout for Flat baseline."""
    with torch.no_grad():
        # Run full context
        out = model(x_context)
        z_ctx = out["z_seq"]
        # Last hidden + predicted next
        h = z_ctx[-1]
        preds = []
        for _ in range(K):
            h_next = model.proj_next(h)
            preds.append(h_next)
            h = h_next
        return torch.stack(preds)


def rwm_rollout(model: RegimeWorldModel, x_context: torch.Tensor, K: int) -> dict:
    """K-step rollout via RegimeWorldModel.rollout()."""
    return model.rollout(x_context, K)


@torch.no_grad()
def gt_z_slow(model, x_full: torch.Tensor, kind="rwm") -> torch.Tensor:
    """Ground-truth z_slow from full trajectory encoding."""
    if kind == "rwm":
        out = model(x_full)
        return out["z_slow"]
    else:
        out = model(x_full)
        return out["z_seq"]


def eval_rollout(model, dataset, K_max: int = 32, n_ctx_slow: int = 4,
                  kind: str = "rwm") -> dict:
    """For each sample, compare K-step rollout to ground-truth latent.

    Note for RWM: z_slow has stride k=4 from x → need T_ctx = n_ctx_slow * 4
    frames to encode n_ctx_slow slow steps as context.
    For Flat: per-step latent, T_ctx = n_ctx_slow * 4 to match.
    """
    mse_per_step = np.zeros(K_max)
    valid_counts = np.zeros(K_max)
    T_ctx = n_ctx_slow * 4  # context length in frames

    for sample in dataset:
        x = torch.from_numpy(sample.traj).float()
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        T = x.shape[0]
        if T < T_ctx + K_max * 4:
            continue  # too short
        x_context = x[:T_ctx]
        x_full = x[: T_ctx + K_max * 4]

        # GT slow trajectory (for RWM) or per-step latent (for Flat)
        gt_full = gt_z_slow(model, x_full, kind=kind)
        # Skip context portion
        if kind == "rwm":
            # gt_full has shape (n_total_slow_steps, D_slow)
            # The first n_ctx_slow are context. Predict the next K
            n_ctx_steps = T_ctx // 4
            gt_target = gt_full[n_ctx_steps : n_ctx_steps + K_max]
        else:
            # per-step
            gt_target = gt_full[T_ctx : T_ctx + K_max]

        # Rollout
        if kind == "rwm":
            out = rwm_rollout(model, x_context, K_max)
            pred = out["z_slow_pred"]
        else:
            pred = flat_rollout(model, x_context, K_max)

        K_actual = min(pred.shape[0], gt_target.shape[0])
        for k in range(K_actual):
            mse_k = ((pred[k] - gt_target[k]) ** 2).mean().item()
            mse_per_step[k] += mse_k
            valid_counts[k] += 1

    mse_per_step = mse_per_step / np.maximum(valid_counts, 1)
    return {"mse_per_step": mse_per_step.tolist(),
             "valid_counts": valid_counts.tolist()}


def run_one_seed(seed: int, n_per_regime=12, n_epochs=40, T=128, K_max=24):
    print(f"\n=== SEED {seed} ===")
    train_ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in train_ds:
        pad_to_d_in(s, 3)
    # Eval = held-out seeds
    eval_ds = make_dataset(n_per_regime=8, T=T, base_seed=seed * 100 + 5000)
    for s in eval_ds:
        pad_to_d_in(s, 3)

    # Train RWM-sup
    torch.manual_seed(seed); np.random.seed(seed)
    rwm = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4)
    cfg = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                       w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                       w_entropy=0.0, w_regime_sup=5.0)
    ts = time.time()
    train(rwm, train_ds, cfg, verbose=False)
    print(f"  RWM trained ({time.time() - ts:.1f}s)")

    # Train Flat-32
    torch.manual_seed(seed); np.random.seed(seed)
    flat32 = FlatBaseline(d_in=3, d_h=32)
    ts = time.time()
    train_baseline(flat32, train_ds, n_epochs=n_epochs)
    print(f"  Flat-32 trained ({time.time() - ts:.1f}s)")

    # Train Flat-56 (matched ~30K params)
    torch.manual_seed(seed); np.random.seed(seed)
    flat56 = FlatBaseline(d_in=3, d_h=56)
    ts = time.time()
    train_baseline(flat56, train_ds, n_epochs=n_epochs)
    print(f"  Flat-56 trained ({time.time() - ts:.1f}s)")

    # Rollout eval
    rwm_res = eval_rollout(rwm, eval_ds, K_max=K_max, kind="rwm")
    f32_res = eval_rollout(flat32, eval_ds, K_max=K_max, kind="flat")
    f56_res = eval_rollout(flat56, eval_ds, K_max=K_max, kind="flat")

    return {
        "seed": seed,
        "rwm_mse_per_step": rwm_res["mse_per_step"],
        "f32_mse_per_step": f32_res["mse_per_step"],
        "f56_mse_per_step": f56_res["mse_per_step"],
        "valid_counts": rwm_res["valid_counts"],
    }


def main():
    print("=" * 80)
    print("MULTI-STEP ROLLOUT — RWM vs Flat-32 vs Flat-56 (matched)")
    print("=" * 80)
    print("  Context : first 16 frames (= 4 slow steps for RWM)")
    print("  Predict : K ∈ {1, 2, 4, 8, 16, 24} steps autoregressively")
    print("  Metric  : per-step MSE vs GT-encoded z_slow")
    print("=" * 80)

    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "3"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} failed : {e}")
            import traceback; traceback.print_exc()
    print(f"\n[time] total {time.time() - t0:.1f}s")

    if not all_runs:
        return

    # Aggregate per-step
    rwm_mat = np.array([r["rwm_mse_per_step"] for r in all_runs])
    f32_mat = np.array([r["f32_mse_per_step"] for r in all_runs])
    f56_mat = np.array([r["f56_mse_per_step"] for r in all_runs])

    rwm_mean = rwm_mat.mean(axis=0)
    f32_mean = f32_mat.mean(axis=0)
    f56_mean = f56_mat.mean(axis=0)

    print("\n" + "=" * 80)
    print("AGGREGATE per-step MSE (mean across seeds)")
    print("=" * 80)
    print(f"  {'K':>4}{'RWM':>14}{'F32':>14}{'F56':>14}{'R/F32':>10}{'R/F56':>10}")
    print("-" * 70)
    for k_report in [0, 1, 3, 7, 15, K_MAX - 1]:
        if k_report >= len(rwm_mean):
            continue
        rwm = rwm_mean[k_report]
        f32 = f32_mean[k_report]
        f56 = f56_mean[k_report]
        ratio32 = rwm / max(f32, 1e-9)
        ratio56 = rwm / max(f56, 1e-9)
        print(f"  {k_report+1:>4}{rwm:>14.5f}{f32:>14.5f}{f56:>14.5f}{ratio32:>10.2f}{ratio56:>10.2f}")

    # H2 test : does ratio degrade or stay stable as K grows?
    print("\n" + "=" * 80)
    print("H2 : RWM degradation vs Flat across K")
    print("=" * 80)
    ratios_f56 = rwm_mean / np.maximum(f56_mean, 1e-9)
    print(f"  R/F56 ratio at K=1  : {ratios_f56[0]:.3f}")
    print(f"  R/F56 ratio at K=8  : {ratios_f56[7]:.3f}")
    print(f"  R/F56 ratio at K=16 : {ratios_f56[15]:.3f}")
    print(f"  R/F56 ratio at K=last: {ratios_f56[-1]:.3f}")

    initial_advantage = 1.0 - ratios_f56[0]  # > 0 = RWM better
    final_advantage = 1.0 - ratios_f56[-1]
    print(f"\n  Initial advantage (K=1)  : {initial_advantage:+.3f}")
    print(f"  Final advantage (K={K_MAX}) : {final_advantage:+.3f}")

    if final_advantage > 0.3:
        verdict = "✓ H2 SUPPORTED : RWM maintains substantial advantage at long horizons"
    elif final_advantage > 0.0:
        verdict = "≈ H2 PARTIAL : RWM still slightly better but advantage shrinks"
    elif final_advantage > -0.3:
        verdict = "✗ H2 FALSIFIED : RWM advantage disappears at long horizons (1-step circular)"
    else:
        verdict = "✗✗ H2 STRONGLY FALSIFIED : RWM WORSE than Flat at long horizons"
    print(f"\n  {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "all_runs": all_runs,
        "rwm_mean_per_step": rwm_mean.tolist(),
        "f32_mean_per_step": f32_mean.tolist(),
        "f56_mean_per_step": f56_mean.tolist(),
        "ratios_vs_f56": ratios_f56.tolist(),
        "K_max": K_MAX,
        "verdict": verdict,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_rollout.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/regime_world_rollout.json")


if __name__ == "__main__":
    main()
