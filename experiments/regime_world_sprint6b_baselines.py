"""Sprint 6B baselines : HMM + Markov-switching + Transformer on real BTC+ETH.

Compares per_ts_lyap (Sprint 6A winner) against standard finance literature
baselines :
  - HMM (Gaussian K=3, hmmlearn) — Hamilton 1989 lineage
  - MarkovSwitch (statsmodels MarkovRegression) — switching variance AR(1)
  - Transformer (1-layer 4-head encoder) — modern ML baseline

If per_ts_lyap still beats these → architecture has independent value beyond
its (possibly collapsed) routing mechanism.
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

from backend.regime_world import RegimeWorldModel, TrainConfig, train
from backend.regime_world.finance_real import make_real_market_dataset
from backend.regime_world.finance import REGIMES as FIN_REGIMES
from experiments.finance_baselines import (
    HMMBaseline, MarkovSwitchBaseline, TransformerBaseline,
)
from experiments.regime_world_probe import train_baseline
from experiments.regime_world_rollout_attractor import eval_model_per_regime

FIN_TO_SYNTH = {"bull": "smooth", "sideways": "periodic", "bear": "chaotic"}


def relabel(samples):
    for s in samples:
        dom_idx = int(np.bincount(s.regime_labels, minlength=3).argmax())
        s.regime = FIN_TO_SYNTH[FIN_REGIMES[dom_idx]]
    return samples


def build_per_ts_lyap(seed, d_in=5):
    torch.manual_seed(seed); np.random.seed(seed)
    m = RegimeWorldModel(d_in=d_in, d_fast=32, d_slow=32, slow_stride=4,
                          stable=True, per_timestep_router=True, router_window=4)
    cfg = TrainConfig(n_epochs=30, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                       w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                       w_regime_sup=5.0, w_multistep=2.0, K_multistep=4,
                       gamma_multistep=0.9, n_starts_multistep=4,
                       w_lyap_reg=0.5, lyap_targets=(0.9, 1.0, 1.0))
    return m, cfg


def build_transformer(seed, d_in=5):
    torch.manual_seed(seed); np.random.seed(seed)
    return TransformerBaseline(d_in=d_in, d_model=64, n_heads=4, n_layers=1)


def split_train_eval(samples, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(len(idx) * train_ratio)
    return [samples[i] for i in idx[:n_tr]], [samples[i] for i in idx[n_tr:]]


def run_one_seed(seed, samples, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_ds, eval_ds = split_train_eval(samples, 0.8, seed=seed)
    results = {"seed": seed}

    # 1) per_ts_lyap (reference)
    try:
        t0 = time.time()
        m, cfg = build_per_ts_lyap(seed)
        train(m, train_ds, cfg, verbose=False)
        p = sum(x.numel() for x in m.parameters() if x.requires_grad)
        res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind="rwm")
        results["per_ts_lyap"] = {"params": p, "eval": res}
        print(f"  per_ts_lyap   ({time.time()-t0:.1f}s, {p/1000:.1f}K)", flush=True)
    except Exception as e:
        print(f"  per_ts_lyap FAILED: {e}", flush=True)
        results["per_ts_lyap"] = {"failed": str(e)}

    # 2) HMM
    try:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        m = HMMBaseline(d_in=5, n_states=3)
        m.fit(train_ds)
        res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind="flat")
        results["hmm"] = {"params": 0, "eval": res}
        print(f"  hmm           ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"  hmm FAILED: {e}", flush=True)
        results["hmm"] = {"failed": str(e)}

    # 3) Markov-switch
    try:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        m = MarkovSwitchBaseline(d_in=5, n_states=3)
        m.fit(train_ds)
        if m.fitted:
            res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind="flat")
            results["markov_switch"] = {"params": 0, "eval": res}
            print(f"  markov_switch ({time.time()-t0:.1f}s)", flush=True)
        else:
            results["markov_switch"] = {"failed": "fit did not converge"}
            print(f"  markov_switch FAILED: fit did not converge", flush=True)
    except Exception as e:
        print(f"  markov_switch FAILED: {e}", flush=True)
        results["markov_switch"] = {"failed": str(e)}

    # 4) Transformer
    try:
        t0 = time.time()
        m = build_transformer(seed)
        train_baseline(m, train_ds, n_epochs=30, w_multistep=2.0,
                        K_multistep=4, gamma_multistep=0.9,
                        n_starts_multistep=4)
        p = sum(x.numel() for x in m.parameters() if x.requires_grad)
        res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind="flat")
        results["transformer"] = {"params": p, "eval": res}
        print(f"  transformer   ({time.time()-t0:.1f}s, {p/1000:.1f}K)", flush=True)
    except Exception as e:
        print(f"  transformer FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        results["transformer"] = {"failed": str(e)}
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
    print("SPRINT 6B BASELINES : per_ts_lyap vs HMM vs Markov-switch vs Transformer", flush=True)
    print("=" * 96, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "10"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    samples = make_real_market_dataset(tickers=["BTC-USD", "ETH-USD"],
                                         period="5y", window_T=256, stride=64)
    samples = relabel(samples)
    print(f"  total samples: {len(samples)}", flush=True)
    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, samples, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} fatal: {e}", flush=True)
    print(f"\n[time] total {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    if not all_runs: return

    VARIANTS = ["per_ts_lyap", "hmm", "markov_switch", "transformer"]
    REGIMES = ["smooth", "periodic", "chaotic"]
    LABELS = {"smooth": "bull", "periodic": "sideways", "chaotic": "bear"}
    K = K_MAX - 1

    def collect(variant, regime, k):
        vals = []
        for run in all_runs:
            res = run.get(variant, {})
            if "failed" in res: continue
            mse = res.get("eval", {}).get("mse_per_regime", {}).get(regime, None)
            if mse is None or k >= len(mse): continue
            v = mse[k]
            if np.isfinite(v) and abs(v) < 1e15: vals.append(v)
        return np.array(vals)

    print(f"\n{'=' * 96}", flush=True)
    print(f"K={K+1} MSE per (mapped) regime — real BTC+ETH", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'bull':>22}{'sideways':>22}{'bear':>22}", flush=True)
    for variant in VARIANTS:
        row = [f"  {variant:<14}"]
        for r in REGIMES:
            v = collect(variant, r, K)
            if len(v) == 0: row.append(f"{'n/a':>22}")
            else: row.append(f"{v.mean():>12.3e}±{v.std():.0e}({len(v)})".rjust(22))
        print("".join(row), flush=True)

    print(f"\n{'=' * 96}", flush=True)
    print("Welch t-tests : each baseline vs per_ts_lyap", flush=True)
    print("=" * 96, flush=True)
    base = {r: collect("per_ts_lyap", r, K) for r in REGIMES}
    for variant in ["hmm", "markov_switch", "transformer"]:
        for regime in REGIMES:
            v = collect(variant, regime, K)
            t, p = welch_ttest(v, base[regime])
            if len(v) == 0: eff = "no data"
            elif np.isnan(p): eff = "—"
            else:
                worse = v.mean() > base[regime].mean()
                arrow = "↑ worse" if worse else "↓ better"
                sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                        "*" if p < 0.05 else "ns")
                eff = f"{arrow} {sig}  (baseline {LABELS[regime]})"
            tstr = f"{t:>8.2f}" if not np.isnan(t) else f"{'—':>8}"
            pstr = f"{p:>10.4f}" if not np.isnan(p) else f"{'—':>10}"
            print(f"  {variant:<14}{LABELS[regime]:>10}{tstr}{pstr}  {eff}",
                  flush=True)

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "all_runs": all_runs}
    json.dump(out, open("results/regime_world_sprint6b_baselines.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint6b_baselines.json", flush=True)


if __name__ == "__main__":
    main()
