"""Sprint 1 unified architectural sweep.

Comparaisons (sur 10 seeds, mixed-regime synth, attractor + Welch t-tests) :

  A) stab+ms       : baseline (§10.9 fix : K=3, MLP router 32, residual+tanh)
  B) wider-router  : A + router MLP 4→128→3 + skip-connection (#6)
  C) koopman       : A but Koopman dynamics (orthogonal K + lift) (#10)
  D) K6-gumbel     : K=6 experts + Gumbel-softmax + load balance (#9 ; no sup)
  E) flat56+ms     : flat-GRU control (capacity-fair, 30K)

Pour chaque sample : MSE per K + attractor metrics.
Pour chaque pair (A vs B, A vs C, ...) : Welch t-test on K=24 MSE.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
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
from experiments.regime_world_rollout_attractor import eval_model_per_regime


def build(name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if name == "stab_ms":
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=5.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg, "rwm"
    if name == "wider_router":
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True, router_hidden=128,
                              router_skip=True)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=5.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg, "rwm"
    if name == "koopman":
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              koopman=True, koopman_residual=False)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=5.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg, "rwm"
    if name == "k6_gumbel":
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True, n_regimes=6,
                              router_hidden=64, router_gumbel=True)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=0.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4, w_load_balance=0.1)
        return m, cfg, "rwm"
    if name == "flat56_ms":
        m = FlatBaseline(d_in=3, d_h=56)
        return m, None, "flat"
    raise ValueError(name)


def train_one(name, seed, train_ds):
    m, cfg, kind = build(name, seed)
    if kind == "rwm":
        train(m, train_ds, cfg, verbose=False)
    else:
        train_baseline(m, train_ds, n_epochs=40, w_multistep=2.0,
                        K_multistep=4, gamma_multistep=0.9,
                        n_starts_multistep=4)
    return m, kind


def run_one_seed(seed, n_per_regime=12, n_epochs=40, T=128, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in train_ds: pad_to_d_in(s, 3)
    eval_ds = make_dataset(n_per_regime=8, T=T, base_seed=seed * 100 + 5000)
    for s in eval_ds: pad_to_d_in(s, 3)
    results = {"seed": seed}
    for name in ["stab_ms", "wider_router", "koopman", "k6_gumbel", "flat56_ms"]:
        t0 = time.time()
        try:
            m, kind = train_one(name, seed, train_ds)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind=kind)
            results[name] = {"params": params, **res}
            print(f"  {name:<14} ({time.time()-t0:.1f}s, {params/1000:.1f}K)",
                  flush=True)
        except Exception as e:
            print(f"  {name:<14} FAILED: {e}", flush=True)
            results[name] = {"failed": str(e)}
    return results


def welch_ttest(a, b):
    """Welch's t-test : returns (t, p). Two-sided. Handles unequal var."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
    try:
        from scipy.stats import ttest_ind
        t, p = ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except ImportError:
        # Manual Welch
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = np.sqrt(va / na + vb / nb)
        if se < 1e-12: return float("nan"), float("nan")
        t = (ma - mb) / se
        return float(t), float("nan")  # no p without scipy


def main():
    print("=" * 96, flush=True)
    print("SPRINT 1 unified : stab+ms vs wider-router vs koopman vs K6-gumbel vs flat", flush=True)
    print("=" * 96, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "10"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    N_EPOCHS = int(os.environ.get("RWM_N_EPOCHS", "40"))
    print(f"  Seeds {N_SEEDS}  K_max {K_MAX}  epochs {N_EPOCHS}", flush=True)

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, n_epochs=N_EPOCHS, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} skipped : {e}", flush=True)
    elapsed = time.time() - t0
    print(f"\n[time] total {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    if not all_runs: return

    MODELS = ["stab_ms", "wider_router", "koopman", "k6_gumbel", "flat56_ms"]
    REGIMES = ["smooth", "periodic", "chaotic"]

    # Aggregate (filter NaN/inf)
    def collect(model, regime, K_idx):
        vals = []
        for run in all_runs:
            if model not in run: continue
            res = run[model]
            if "failed" in res: continue
            mse = res.get("mse_per_regime", {}).get(regime, None)
            if mse is None or K_idx >= len(mse): continue
            v = mse[K_idx]
            if np.isfinite(v) and abs(v) < 1e15: vals.append(v)
        return np.array(vals)

    # Print per-regime table at K=24
    K = K_MAX - 1
    print("\n" + "=" * 96, flush=True)
    print(f"MSE at K={K+1} per regime, mean ± std across seeds", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Model':<14}{'smooth':>20}{'periodic':>20}{'chaotic':>20}", flush=True)
    for model in MODELS:
        row = [f"  {model:<14}"]
        for regime in REGIMES:
            v = collect(model, regime, K)
            if len(v) == 0:
                row.append(f"{'n/a':>20}")
            else:
                row.append(f"{v.mean():>10.3e}±{v.std():.0e}({len(v)})".rjust(20))
        print("".join(row), flush=True)

    # Welch t-tests : each variant vs baseline (stab_ms), per regime, K=24
    print("\n" + "=" * 96, flush=True)
    print(f"Welch t-test vs stab_ms (baseline) at K={K+1}", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'regime':>12}{'t':>10}{'p':>10}  effect", flush=True)
    base = {r: collect("stab_ms", r, K) for r in REGIMES}
    for model in MODELS:
        if model == "stab_ms": continue
        for regime in REGIMES:
            v = collect(model, regime, K)
            t, p = welch_ttest(v, base[regime])
            if len(v) == 0:
                eff = "no data"
            elif np.isnan(p):
                eff = "—"
            else:
                # Negative t = variant has LOWER MSE = better
                better = v.mean() < base[regime].mean()
                arrow = "↓ better" if better else "↑ worse"
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                       ("*" if p < 0.05 else "ns"))
                eff = f"{arrow} {sig}"
            tstr = f"{t:>10.2f}" if not np.isnan(t) else f"{'—':>10}"
            pstr = f"{p:>10.4f}" if not np.isnan(p) else f"{'—':>10}"
            print(f"  {model:<14}{regime:>12}{tstr}{pstr}  {eff}", flush=True)

    # Aggregate full per-K curves
    summary = {}
    for model in MODELS:
        mse_full = {}
        for regime in REGIMES:
            stack = []
            for run in all_runs:
                if model not in run: continue
                res = run[model]
                if "failed" in res: continue
                mse = res.get("mse_per_regime", {}).get(regime, None)
                if mse is None: continue
                arr = np.asarray(mse, dtype=float)
                if np.isfinite(arr).all() and (np.abs(arr) < 1e15).all():
                    stack.append(arr)
            if stack:
                a = np.array(stack)
                mse_full[regime] = {
                    "mean": a.mean(axis=0).tolist(),
                    "std": a.std(axis=0).tolist(),
                    "n": int(a.shape[0]),
                }
        params_list = [run[model]["params"] for run in all_runs
                       if model in run and "failed" not in run[model]
                       and "params" in run[model]]
        params = int(np.mean(params_list)) if params_list else None
        summary[model] = {"mse_per_regime": mse_full, "params": params}

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "n_epochs": N_EPOCHS,
            "all_runs": all_runs, "summary": summary}
    json.dump(out, open("results/regime_world_sprint1.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint1.json", flush=True)


if __name__ == "__main__":
    main()
