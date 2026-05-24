"""Sprint 4 architectural sweep.

Test #3 (Lyap regularization) + #5 (continuous regime parameter) +
combined avec Sprint 3 winner (per_timestep).

  A) baseline           : stab+ms baseline (control)
  B) lyap_reg           : baseline + differentiable Lyap σ_max penalty
  C) continuous         : continuous regime θ ∈ [0,1] + cos-basis dynamics
  D) per_timestep       : Sprint 3 winner re-included
  E) per_ts + lyap      : Sprint 3 winner + Sprint 4 Lyap regularizer

Eval = harder bench (Sprint 3 protocol).
10 seeds, Welch t-tests vs baseline.
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
    RegimeWorldModel, TrainConfig, train,
    make_dataset, make_harder_dataset,
)
from experiments.regime_world_probe import pad_to_d_in
from experiments.regime_world_rollout_attractor import eval_model_per_regime


def build(variant, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    base_kw = dict(d_in=3, d_fast=32, d_slow=32, slow_stride=4, stable=True)
    base_cfg = dict(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                     w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                     w_regime_sup=5.0, w_multistep=2.0, K_multistep=4,
                     gamma_multistep=0.9, n_starts_multistep=4)
    if variant == "baseline":
        m = RegimeWorldModel(**base_kw)
        cfg = TrainConfig(**base_cfg)
    elif variant == "lyap_reg":
        m = RegimeWorldModel(**base_kw)
        cfg = TrainConfig(**base_cfg, w_lyap_reg=0.5,
                           lyap_targets=(0.9, 1.0, 1.0))
    elif variant == "continuous":
        m = RegimeWorldModel(**base_kw, continuous_regime=True, n_basis=4)
        # Continuous = no regime supervision (no taxonomy)
        cfg_c = dict(base_cfg)
        cfg_c["w_regime_sup"] = 0.0
        cfg = TrainConfig(**cfg_c)
    elif variant == "per_timestep":
        m = RegimeWorldModel(**base_kw, per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg)
    elif variant == "per_ts_lyap":
        m = RegimeWorldModel(**base_kw, per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg, w_lyap_reg=0.5,
                           lyap_targets=(0.9, 1.0, 1.0))
    else:
        raise ValueError(variant)
    return m, cfg


def run_one_seed(seed, n_per_regime=12, n_epochs=40, T=128, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in train_ds: pad_to_d_in(s, 3)
    eval_hard = make_harder_dataset(n_per_regime=8, T=T,
                                      base_seed=seed * 100 + 7000,
                                      noise_std=0.10, mask_prob=0.3,
                                      param_shift=True)
    for s in eval_hard: pad_to_d_in(s, 3)
    results = {"seed": seed}
    for variant in ["baseline", "lyap_reg", "continuous", "per_timestep",
                      "per_ts_lyap"]:
        t0 = time.time()
        try:
            m, cfg = build(variant, seed)
            train(m, train_ds, cfg, verbose=False)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            res = eval_model_per_regime(m, eval_hard, K_max=K_max, kind="rwm")
            results[variant] = {"params": params, "hard": res}
            print(f"  {variant:<14} ({time.time()-t0:.1f}s, {params/1000:.1f}K)",
                  flush=True)
        except Exception as e:
            print(f"  {variant:<14} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[variant] = {"failed": str(e)}
    return results


def welch_ttest(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
    try:
        from scipy.stats import ttest_ind
        t, p = ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except ImportError:
        return float("nan"), float("nan")


def main():
    print("=" * 96, flush=True)
    print("SPRINT 4 : baseline vs +lyap_reg vs continuous vs per_timestep vs per_ts+lyap", flush=True)
    print("Eval = HARDER bench (Sprint 3 protocol)", flush=True)
    print("=" * 96, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "10"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} fatal: {e}", flush=True)
    print(f"\n[time] total {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    if not all_runs: return

    VARIANTS = ["baseline", "lyap_reg", "continuous", "per_timestep", "per_ts_lyap"]
    REGIMES = ["smooth", "periodic", "chaotic"]
    K = K_MAX - 1

    def collect(variant, regime, k):
        vals = []
        for run in all_runs:
            res = run.get(variant, {})
            if "failed" in res: continue
            mse = res.get("hard", {}).get("mse_per_regime", {}).get(regime, None)
            if mse is None or k >= len(mse): continue
            v = mse[k]
            if np.isfinite(v) and abs(v) < 1e15: vals.append(v)
        return np.array(vals)

    print(f"\n{'=' * 96}", flush=True)
    print(f"K={K+1} MSE per regime — HARDER eval", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'smooth':>22}{'periodic':>22}{'chaotic':>22}",
          flush=True)
    for variant in VARIANTS:
        row = [f"  {variant:<14}"]
        for regime in REGIMES:
            v = collect(variant, regime, K)
            if len(v) == 0: row.append(f"{'n/a':>22}")
            else: row.append(f"{v.mean():>12.3e}±{v.std():.0e}({len(v)})".rjust(22))
        print("".join(row), flush=True)

    print(f"\n{'=' * 96}", flush=True)
    print(f"Welch t-tests vs baseline @ K={K+1}", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'regime':>10}{'t':>8}{'p':>10}  effect", flush=True)
    base = {r: collect("baseline", r, K) for r in REGIMES}
    for variant in VARIANTS:
        if variant == "baseline": continue
        for regime in REGIMES:
            v = collect(variant, regime, K)
            t, p = welch_ttest(v, base[regime])
            if len(v) == 0: eff = "no data"
            elif np.isnan(p): eff = "—"
            else:
                better = v.mean() < base[regime].mean()
                arrow = "↓ better" if better else "↑ worse"
                sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                        "*" if p < 0.05 else "ns")
                eff = f"{arrow} {sig}"
            tstr = f"{t:>8.2f}" if not np.isnan(t) else f"{'—':>8}"
            pstr = f"{p:>10.4f}" if not np.isnan(p) else f"{'—':>10}"
            print(f"  {variant:<14}{regime:>10}{tstr}{pstr}  {eff}", flush=True)

    summary = {}
    for variant in VARIANTS:
        s = {"hard": {}}
        for regime in REGIMES:
            stack = []
            for run in all_runs:
                res = run.get(variant, {})
                if "failed" in res: continue
                mse = res.get("hard", {}).get("mse_per_regime", {}).get(regime, None)
                if mse is None: continue
                arr = np.asarray(mse, dtype=float)
                if np.isfinite(arr).all() and (np.abs(arr) < 1e15).all():
                    stack.append(arr)
            if stack:
                a = np.array(stack)
                s["hard"][regime] = {"mean": a.mean(axis=0).tolist(),
                                       "std": a.std(axis=0).tolist(),
                                       "n": int(a.shape[0])}
        params_list = [r[variant]["params"] for r in all_runs
                       if variant in r and "failed" not in r[variant]
                       and "params" in r[variant]]
        s["params"] = int(np.mean(params_list)) if params_list else None
        summary[variant] = s
    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "all_runs": all_runs,
            "summary": summary}
    json.dump(out, open("results/regime_world_sprint4.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint4.json", flush=True)


if __name__ == "__main__":
    main()
