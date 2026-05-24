"""Mixed-regime training — FAIR test of regime-aware vs flat.

Per Phase 1 OOD findings : single-regime training collapses router.
Real test : train on diverse 3-regime mix, eval IN-dist + OOD parameter
shifts within each regime class.

Hypothesis (H1) : RWM-sup with diverse training generalizes OOD better
than Flat baseline because regime-aware dynamics modules adapt to
parameter shifts within a regime.

Setup :
- Training : balanced 3 regimes via make_dataset (default systems +
  default params)
- Eval IN-DIST : held-out seed of same systems same params
- Eval OOD : same systems, parameter shifts ±30%

For each (regime, system, distribution) compute :
- 1-step prediction MSE on z (regime-conditional pred for RWM, GRU pred for Flat)
- Router accuracy if RWM
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
from backend.regime_world.synth import (
    gen_damped_oscillator, gen_exp_decay,
    gen_sine_pair, gen_van_der_pol, gen_kuramoto_sync,
    gen_lorenz, gen_rossler, gen_logistic, gen_henon,
)
from experiments.regime_world_probe import (
    FlatBaseline, train_baseline, pad_to_d_in,
)
from experiments.regime_world_ood import standardize_pad, rwm_metrics, flat_metrics


# Each entry : (label, generator, in_dist_params, ood_params)
SYSTEMS = {
    "damped_osc": {
        "regime": "smooth",
        "in_dist": [{"gamma": 0.15, "w0": 1.0}],
        "ood": [{"gamma": 0.05, "w0": 1.0}, {"gamma": 0.30, "w0": 1.5}],
        "gen": gen_damped_oscillator,
    },
    "exp_decay": {
        "regime": "smooth",
        "in_dist": [{"tau": 5.0}],
        "ood": [{"tau": 2.5}, {"tau": 10.0}],
        "gen": gen_exp_decay,
    },
    "sine_pair": {
        "regime": "periodic",
        "in_dist": [{"freq": 0.05}],
        "ood": [{"freq": 0.025}, {"freq": 0.10}],
        "gen": gen_sine_pair,
    },
    "van_der_pol": {
        "regime": "periodic",
        "in_dist": [{"mu": 1.5}],
        "ood": [{"mu": 0.8}, {"mu": 2.5}],
        "gen": gen_van_der_pol,
    },
    "lorenz": {
        "regime": "chaotic",
        "in_dist": [{"rho": 28.0}],
        "ood": [{"rho": 24.0}, {"rho": 36.0}],
        "gen": lambda T, seed, rho=28.0: gen_lorenz(T=T, seed=seed, rho=rho),
    },
    "rossler": {
        "regime": "chaotic",
        "in_dist": [{}],
        "ood": [{"a": 0.15}, {"a": 0.30}],
        "gen": lambda T, seed, a=0.2: gen_rossler(T=T, seed=seed, a=a),
    },
}


def make_eval_set(systems_dict, n_per_param=4, T=128, base_seed=10000):
    """For each system, gen N samples per in_dist + ood params."""
    out = {"in_dist": {}, "ood": {}}
    seed = base_seed
    for sname, info in systems_dict.items():
        out["in_dist"][sname] = []
        out["ood"][sname] = []
        for params in info["in_dist"]:
            for i in range(n_per_param):
                s = info["gen"](T=T, seed=seed, **params)
                s.regime = info["regime"]
                pad_to_d_in(s, 3)
                out["in_dist"][sname].append(s)
                seed += 1
        for params in info["ood"]:
            for i in range(n_per_param):
                s = info["gen"](T=T, seed=seed, **params)
                s.regime = info["regime"]
                pad_to_d_in(s, 3)
                out["ood"][sname].append(s)
                seed += 1
    return out


def eval_model_on_samples(model, samples, kind="rwm"):
    mses, norms, latent_stds = [], [], []
    routers = []
    for s in samples:
        x = standardize_pad(s.traj)
        if kind == "rwm":
            m = rwm_metrics(model, x)
            mses.append(m["mse_1step"])
            norms.append(m["latent_norm"])
            routers.append(m["r"])
            # Collapse diagnostic : z_slow variance across time
            with torch.no_grad():
                out = model(x)
                z_slow = out["z_slow"]
                latent_stds.append(float(z_slow.std(dim=0).mean().item()))
        else:
            m = flat_metrics(model, x)
            mses.append(m["mse_1step"])
            norms.append(m["latent_norm"])
            with torch.no_grad():
                out = model(x)
                z = out["z_seq"]
                latent_stds.append(float(z.std(dim=0).mean().item()))
    out = {"mse_mean": float(np.mean(mses)), "mse_std": float(np.std(mses)),
           "norm_mean": float(np.mean(norms)),
           "latent_std_mean": float(np.mean(latent_stds))}
    if routers:
        r_arr = np.array(routers)
        out["router_mean"] = r_arr.mean(axis=0).tolist()
    return out


def run_one_seed(seed: int, n_per_regime=12, n_epochs=40, T=128):
    print(f"\n=== SEED {seed} ===")
    # Train : balanced mix from make_dataset
    train_ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in train_ds:
        pad_to_d_in(s, 3)
    print(f"  train : {len(train_ds)} samples across 3 regimes")

    # Eval data
    eval_set = make_eval_set(SYSTEMS, n_per_param=4, T=T, base_seed=seed * 99999)

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

    # Train flat baseline (small, d_h=32, 12K params)
    torch.manual_seed(seed); np.random.seed(seed)
    base = FlatBaseline(d_in=3, d_h=32)
    ts = time.time()
    train_baseline(base, train_ds, n_epochs=n_epochs)
    t_base = time.time() - ts
    print(f"  Flat-32 trained ({t_base:.1f}s)")

    # Train flat baseline (matched capacity, d_h=56, ~30K params = RWM size)
    torch.manual_seed(seed); np.random.seed(seed)
    base_lg = FlatBaseline(d_in=3, d_h=56)
    ts = time.time()
    train_baseline(base_lg, train_ds, n_epochs=n_epochs)
    t_base_lg = time.time() - ts
    print(f"  Flat-56 (matched 30K params) trained ({t_base_lg:.1f}s)")

    # Eval per system per distribution
    results = {}
    for sname, info in SYSTEMS.items():
        results[sname] = {"regime": info["regime"]}
        for dist in ["in_dist", "ood"]:
            samples = eval_set[dist][sname]
            rwm_m = eval_model_on_samples(rwm, samples, kind="rwm")
            flat_m = eval_model_on_samples(base, samples, kind="flat")
            flat_lg_m = eval_model_on_samples(base_lg, samples, kind="flat")
            regime_true = REGIME_TO_IDX[info["regime"]]
            r_arr = np.array(rwm_m.get("router_mean", [0, 0, 0]))
            router_correct = int(np.argmax(r_arr) == regime_true)
            results[sname][dist] = {
                "rwm_mse": rwm_m["mse_mean"], "flat_mse": flat_m["mse_mean"],
                "flat_lg_mse": flat_lg_m["mse_mean"],
                "rwm_router": rwm_m.get("router_mean", [0, 0, 0]),
                "router_correct": router_correct,
                "rwm_lstd": rwm_m["latent_std_mean"],
                "flat_lstd": flat_m["latent_std_mean"],
                "flat_lg_lstd": flat_lg_m["latent_std_mean"],
            }
        print(f"  {sname:<14} ({info['regime']:<8})  IN  : "
              f"RWM={results[sname]['in_dist']['rwm_mse']:.4f}  "
              f"F32={results[sname]['in_dist']['flat_mse']:.4f}  "
              f"F56={results[sname]['in_dist']['flat_lg_mse']:.4f}")
        print(f"  {' ':<14}             OOD : "
              f"RWM={results[sname]['ood']['rwm_mse']:.4f}  "
              f"F32={results[sname]['ood']['flat_mse']:.4f}  "
              f"F56={results[sname]['ood']['flat_lg_mse']:.4f}")

    return {"seed": seed, "results": results}


def main():
    print("=" * 80)
    print("MIXED-REGIME TRAINING — fair test of RWM-sup vs Flat")
    print("=" * 80)
    print("  Train : balanced mix 3 regimes (smooth/periodic/chaotic, 8 systems)")
    print("  Eval IN-DIST : held-out seed of same systems same params")
    print("  Eval OOD     : same systems with ±parameter shifts")
    print("=" * 80)

    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "3"))
    N_PER_REGIME = int(os.environ.get("RWM_N_PER_REGIME", "12"))
    N_EPOCHS = int(os.environ.get("RWM_N_EPOCHS", "40"))
    print(f"  config : N_SEEDS={N_SEEDS}  N_PER_REGIME={N_PER_REGIME}  "
          f"N_EPOCHS={N_EPOCHS}")

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, n_per_regime=N_PER_REGIME,
                                          n_epochs=N_EPOCHS))
        except Exception as e:
            print(f"  seed {seed} failed : {e}")
            import traceback; traceback.print_exc()
    print(f"\n[time] total {time.time() - t0:.1f}s")

    if not all_runs:
        return

    # Aggregate per (regime, dist)
    print("\n" + "=" * 80)
    print("AGGREGATE (mean across 3 seeds)")
    print("=" * 80)

    regime_acc = {"in_dist": [], "ood": []}
    by_regime = {r: {"in_dist": {"rwm": [], "flat": [], "flat_lg": [],
                                  "rwm_lstd": [], "flat_lstd": []},
                     "ood": {"rwm": [], "flat": [], "flat_lg": [],
                              "rwm_lstd": [], "flat_lstd": []}}
                 for r in ["smooth", "periodic", "chaotic"]}

    for run in all_runs:
        for sname, sdata in run["results"].items():
            reg = sdata["regime"]
            for dist in ["in_dist", "ood"]:
                by_regime[reg][dist]["rwm"].append(sdata[dist]["rwm_mse"])
                by_regime[reg][dist]["flat"].append(sdata[dist]["flat_mse"])
                by_regime[reg][dist]["flat_lg"].append(sdata[dist].get("flat_lg_mse", 0))
                by_regime[reg][dist]["rwm_lstd"].append(sdata[dist].get("rwm_lstd", 0))
                by_regime[reg][dist]["flat_lstd"].append(sdata[dist].get("flat_lstd", 0))
                regime_acc[dist].append(sdata[dist]["router_correct"])

    print(f"\n  {'Regime':<10}{'Dist':<8}{'RWM MSE':>10}{'F32 MSE':>10}{'F56 MSE':>10}{'R/F32':>8}{'R/F56':>8}")
    print("-" * 80)
    for reg in ["smooth", "periodic", "chaotic"]:
        for dist in ["in_dist", "ood"]:
            rwm_m = float(np.mean(by_regime[reg][dist]["rwm"]))
            flat_m = float(np.mean(by_regime[reg][dist]["flat"]))
            flat_lg = float(np.mean(by_regime[reg][dist]["flat_lg"]))
            ratio_s = rwm_m / max(flat_m, 1e-9)
            ratio_l = rwm_m / max(flat_lg, 1e-9)
            print(f"  {reg:<10}{dist:<8}{rwm_m:>10.5f}{flat_m:>10.5f}{flat_lg:>10.5f}{ratio_s:>8.2f}{ratio_l:>8.2f}")

    print(f"\n  Router accuracy across all systems :")
    print(f"    IN-DIST : {np.mean(regime_acc['in_dist']):.3f}")
    print(f"    OOD     : {np.mean(regime_acc['ood']):.3f}")

    # H1 tests : RWM better OOD than (Flat-32) AND (Flat-56 matched)?
    rwm_ood_all = sum([by_regime[r]["ood"]["rwm"] for r in by_regime], [])
    flat_ood_all = sum([by_regime[r]["ood"]["flat"] for r in by_regime], [])
    flat_lg_ood_all = sum([by_regime[r]["ood"]["flat_lg"] for r in by_regime], [])
    ratio_s = np.mean(rwm_ood_all) / max(np.mean(flat_ood_all), 1e-9)
    ratio_l = np.mean(rwm_ood_all) / max(np.mean(flat_lg_ood_all), 1e-9)
    print(f"\n  Global OOD ratios :")
    print(f"    RWM / Flat-32 (12K params) : {ratio_s:.2f}")
    print(f"    RWM / Flat-56 (30K params, matched) : {ratio_l:.2f}")
    if ratio_l < 0.9:
        print(f"  ✓ H1 ARCHITECTURE WINS : RWM beats matched-param Flat OOD")
    elif ratio_l < 1.1:
        print(f"  ≈ H1 CAPACITY-EXPLAINED : RWM ≈ matched Flat OOD (gain from capacity, not arch)")
    else:
        print(f"  ✗ H1 REJECTED : RWM worse than matched Flat OOD")

    Path("results").mkdir(exist_ok=True)
    out = {
        "all_runs": all_runs,
        "by_regime": {r: {d: {"rwm_mean": float(np.mean(by_regime[r][d]["rwm"])),
                              "flat_mean": float(np.mean(by_regime[r][d]["flat"])),
                              "flat_lg_mean": float(np.mean(by_regime[r][d]["flat_lg"]))}
                          for d in ["in_dist", "ood"]}
                       for r in ["smooth", "periodic", "chaotic"]},
        "router_acc_in_dist": float(np.mean(regime_acc["in_dist"])),
        "router_acc_ood": float(np.mean(regime_acc["ood"])),
        "h1_ratio_vs_flat32": float(ratio_s),
        "h1_ratio_vs_flat56_matched": float(ratio_l),
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_mixed.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/regime_world_mixed.json")


if __name__ == "__main__":
    main()
