"""OOD parameter shift — Lorenz ρ sweep.

Hypothesis : regime-aware structured latents degrade gracefully on
out-of-distribution parameter values; flat latents degrade catastrophically
when system crosses regime boundary.

Setup :
- Train both models on Lorenz at ρ=28 (canonical strange attractor, λ≈0.906)
- Eval on ρ ∈ {16, 20, 24, 28, 32, 36, 40}
  * ρ < 24.74 : Hopf bifurcation — origin loses stability, two fixed
    points stable → SMOOTH regime
  * ρ ≈ 24.74 : transition
  * ρ > 24.74 : strange attractor (chaotic), with parameter-shifted shape

Measures :
- 1-step prediction MSE on held-out trajectory (lower = better)
- Latent norm drift (||z_slow|| over time, catastrophic = explodes)
- RWM router regime distribution at each ρ (does it detect transition?)

Expectation :
- Flat baseline : MSE explodes when crossing ρ=24.74 (regime change unseen)
- RWM-sup : router shifts r_smooth at low ρ, dynamics mixture adapts
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

from backend.regime_world import RegimeWorldModel, TrainConfig, train, REGIME_TO_IDX
from backend.regime_world.synth import gen_lorenz, TrajSample
from experiments.regime_world_probe import FlatBaseline, train_baseline


def lorenz_at(rho: float, seed: int = 0, T: int = 200) -> TrajSample:
    """Lorenz at given ρ. Regime label assigned based on ρ vs 24.74 (Hopf)."""
    # gen_lorenz default uses rho=28 ; override via direct integration here
    return gen_lorenz(T=T, rho=rho, seed=seed)


def _override_regime_for_rho(sample: TrajSample, rho: float) -> TrajSample:
    """Lorenz regime depends on ρ : <24.74 smooth (fixed points), > = chaotic."""
    if rho < 24.74:
        sample.regime = "smooth"
        sample.lambda_true = -0.5
    else:
        sample.regime = "chaotic"
        sample.lambda_true = 0.906
    return sample


def standardize_pad(traj: np.ndarray, d_in: int = 3) -> torch.Tensor:
    """Pad/crop + standardize per-trajectory."""
    if traj.shape[1] < d_in:
        pad = np.zeros((traj.shape[0], d_in - traj.shape[1]))
        traj = np.concatenate([traj, pad], axis=1)
    elif traj.shape[1] > d_in:
        traj = traj[:, :d_in]
    x = torch.from_numpy(traj.astype(np.float32))
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def rwm_metrics(model: RegimeWorldModel, x: torch.Tensor) -> dict:
    """1-step pred MSE + latent norm + router distribution."""
    out = model(x)
    z_slow = out["z_slow"]
    z_pred = out["z_slow_pred"]
    if z_slow.shape[0] < 2:
        return {"mse_1step": float("nan"), "latent_norm": float("nan"),
                "r": [0., 0., 0.]}
    mse = float(((z_pred[:-1] - z_slow[1:]) ** 2).mean().item())
    norm = float(z_slow.norm(dim=1).mean().item())
    r = out["r"].cpu().numpy().tolist()
    return {"mse_1step": mse, "latent_norm": norm, "r": r}


@torch.no_grad()
def flat_metrics(model: FlatBaseline, x: torch.Tensor) -> dict:
    out = model(x)
    z = out["z_seq"]
    zp = out["z_pred"]
    if z.shape[0] < 2:
        return {"mse_1step": float("nan"), "latent_norm": float("nan")}
    mse = float(((zp[:-1] - z[1:]) ** 2).mean().item())
    norm = float(z.norm(dim=1).mean().item())
    return {"mse_1step": mse, "latent_norm": norm}


def run_one_seed(seed: int, n_train: int = 24, n_eval_per_rho: int = 8,
                 n_epochs: int = 40):
    print(f"\n=== SEED {seed} ===")
    # Training data : Lorenz ρ=28 only
    train_ds = []
    for i in range(n_train):
        s = lorenz_at(rho=28.0, seed=seed * 1000 + i)
        train_ds.append(_override_regime_for_rho(s, 28.0))
    print(f"  train : {len(train_ds)} Lorenz ρ=28 trajectories")

    # Train RWM-sup
    torch.manual_seed(seed); np.random.seed(seed)
    rwm = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4)
    cfg = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                       w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                       w_entropy=0.1, w_regime_sup=1.0)
    ts = time.time()
    train(rwm, train_ds, cfg, verbose=False)
    t_rwm = time.time() - ts
    print(f"  RWM trained ({t_rwm:.1f}s)")

    # Train flat baseline
    torch.manual_seed(seed); np.random.seed(seed)
    base = FlatBaseline(d_in=3, d_h=32)
    ts = time.time()
    train_baseline(base, train_ds, n_epochs=n_epochs)
    t_base = time.time() - ts
    print(f"  Flat trained ({t_base:.1f}s)")

    # Eval on ρ sweep
    rhos = [16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0]
    rows = []
    for rho in rhos:
        rwm_mse_list, flat_mse_list = [], []
        rwm_norm_list, flat_norm_list = [], []
        r_smooth_list, r_periodic_list, r_chaotic_list = [], [], []
        for i in range(n_eval_per_rho):
            s = lorenz_at(rho=rho, seed=(seed + 1) * 10000 + i)
            x = standardize_pad(s.traj)
            rwm_m = rwm_metrics(rwm, x)
            flat_m = flat_metrics(base, x)
            rwm_mse_list.append(rwm_m["mse_1step"])
            flat_mse_list.append(flat_m["mse_1step"])
            rwm_norm_list.append(rwm_m["latent_norm"])
            flat_norm_list.append(flat_m["latent_norm"])
            r_smooth_list.append(rwm_m["r"][0])
            r_periodic_list.append(rwm_m["r"][1])
            r_chaotic_list.append(rwm_m["r"][2])
        rows.append({
            "rho": rho,
            "expected_regime": "smooth" if rho < 24.74 else "chaotic",
            "rwm_mse_mean": float(np.mean(rwm_mse_list)),
            "rwm_mse_std": float(np.std(rwm_mse_list)),
            "flat_mse_mean": float(np.mean(flat_mse_list)),
            "flat_mse_std": float(np.std(flat_mse_list)),
            "rwm_norm_mean": float(np.mean(rwm_norm_list)),
            "flat_norm_mean": float(np.mean(flat_norm_list)),
            "r_smooth_mean": float(np.mean(r_smooth_list)),
            "r_periodic_mean": float(np.mean(r_periodic_list)),
            "r_chaotic_mean": float(np.mean(r_chaotic_list)),
        })
        print(f"  ρ={rho:>5.1f}  expected={'smooth' if rho<24.74 else 'chaotic':<8}  "
              f"RWM mse={rows[-1]['rwm_mse_mean']:.4f}  Flat mse={rows[-1]['flat_mse_mean']:.4f}  "
              f"r=(s={rows[-1]['r_smooth_mean']:.2f},p={rows[-1]['r_periodic_mean']:.2f},c={rows[-1]['r_chaotic_mean']:.2f})")
    return {"seed": seed, "rows": rows}


def main():
    print("=" * 80)
    print("OOD LORENZ ρ SWEEP — RegimeWorldModel vs Flat Baseline")
    print("=" * 80)
    print("  Train : ρ=28 only ; Eval : ρ ∈ {16, 20, 24, 28, 32, 36, 40}")
    print("  Bifurcation at ρ≈24.74 : ρ<24.74 = fixed-point (smooth)")
    print("                            ρ>24.74 = strange attractor (chaotic)")
    print("=" * 80)

    all_runs = []
    t0 = time.time()
    for seed in range(3):
        try:
            all_runs.append(run_one_seed(seed))
        except Exception as e:
            print(f"  seed {seed} failed : {e}")
    print(f"\n[time] total {time.time() - t0:.1f}s")

    if not all_runs:
        return

    # Aggregate per ρ
    rhos = [r["rho"] for r in all_runs[0]["rows"]]
    print("\n" + "=" * 80)
    print("AGGREGATE (mean across 3 seeds)")
    print("=" * 80)
    print(f"{'ρ':>6}{'expected':<10}{'RWM MSE':>14}{'Flat MSE':>14}{'Δ(RWM-Flat)':>14}{'router (s,p,c)':>22}")
    print("-" * 80)
    agg = {}
    for rho in rhos:
        rwm_mses = [run["rows"][rhos.index(rho)]["rwm_mse_mean"] for run in all_runs]
        flat_mses = [run["rows"][rhos.index(rho)]["flat_mse_mean"] for run in all_runs]
        rs = [run["rows"][rhos.index(rho)]["r_smooth_mean"] for run in all_runs]
        rp = [run["rows"][rhos.index(rho)]["r_periodic_mean"] for run in all_runs]
        rc = [run["rows"][rhos.index(rho)]["r_chaotic_mean"] for run in all_runs]
        rwm_m = float(np.mean(rwm_mses))
        flat_m = float(np.mean(flat_mses))
        delta = rwm_m - flat_m
        expected = "smooth" if rho < 24.74 else "chaotic"
        print(f"{rho:>6.1f}{expected:<10}{rwm_m:>14.5f}{flat_m:>14.5f}{delta:>+14.5f}"
              f"   ({np.mean(rs):.2f},{np.mean(rp):.2f},{np.mean(rc):.2f})")
        agg[rho] = {"rwm_mse": rwm_m, "flat_mse": flat_m, "delta": delta,
                    "r": [float(np.mean(rs)), float(np.mean(rp)), float(np.mean(rc))],
                    "expected": expected}

    # Test if RWM degrades less than Flat at OOD (rho != 28)
    ood_rhos = [r for r in rhos if r != 28.0]
    rwm_ood_mse = float(np.mean([agg[r]["rwm_mse"] for r in ood_rhos]))
    flat_ood_mse = float(np.mean([agg[r]["flat_mse"] for r in ood_rhos]))
    print(f"\n  Mean OOD MSE (ρ ≠ 28) : RWM={rwm_ood_mse:.5f}  Flat={flat_ood_mse:.5f}")
    print(f"  RWM/Flat MSE ratio OOD : {rwm_ood_mse / max(flat_ood_mse, 1e-9):.2f}")

    # Router responsiveness : does r_smooth go up at low rho?
    r_smooth_at_low = float(np.mean([agg[r]["r"][0] for r in [16.0, 20.0]]))
    r_chaotic_at_high = float(np.mean([agg[r]["r"][2] for r in [32.0, 36.0, 40.0]]))
    print(f"\n  Router test :")
    print(f"    r_smooth at ρ≤20 (OOD smooth) : {r_smooth_at_low:.3f}")
    print(f"    r_chaotic at ρ≥32 (still chaotic) : {r_chaotic_at_high:.3f}")
    if r_smooth_at_low > 0.5:
        print(f"    ✓ Router detects OOD regime change at low ρ")
    else:
        print(f"    ✗ Router does NOT detect OOD regime change (stays on training class)")

    Path("results").mkdir(exist_ok=True)
    out = {
        "rhos": rhos, "agg": {str(k): v for k, v in agg.items()},
        "all_runs": all_runs,
        "rwm_ood_mse": rwm_ood_mse, "flat_ood_mse": flat_ood_mse,
        "r_smooth_at_low": r_smooth_at_low,
        "r_chaotic_at_high": r_chaotic_at_high,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_ood.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/regime_world_ood.json")


if __name__ == "__main__":
    main()
