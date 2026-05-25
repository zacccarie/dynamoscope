"""Sprint 6A : real BTC + ETH finance data, replicate Sprint 5 protocol.

Same 4 RWM variants as Sprint 5 + LSTM baseline (standard finance ML)
on real daily candles (BTC-USD + ETH-USD, 5 years, ~50 windows × 256 days).

This is the first non-synthetic, non-procedural-video test of the
regime-aware architecture. If per_ts_lyap (Sprint 4+5 winner) wins
here with p<0.05 again, we have a real-data architectural claim.
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
from experiments.regime_world_probe import (
    FlatBaseline, LSTMBaseline, train_baseline,
)
from experiments.regime_world_rollout_attractor import eval_model_per_regime


FIN_TO_SYNTH = {"bull": "smooth", "sideways": "periodic", "bear": "chaotic"}


def relabel(samples):
    """Map finance regime → synth regime for K=3 architecture reuse."""
    for s in samples:
        dom_idx = int(np.bincount(s.regime_labels, minlength=3).argmax())
        s.regime = FIN_TO_SYNTH[FIN_REGIMES[dom_idx]]
    return samples


def build(variant, seed, d_in=5):
    torch.manual_seed(seed); np.random.seed(seed)
    base_kw = dict(d_in=d_in, d_fast=32, d_slow=32, slow_stride=4, stable=True)
    base_cfg = dict(n_epochs=30, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                     w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                     w_regime_sup=5.0, w_multistep=2.0, K_multistep=4,
                     gamma_multistep=0.9, n_starts_multistep=4)
    if variant == "baseline":
        m = RegimeWorldModel(**base_kw); cfg = TrainConfig(**base_cfg); kind = "rwm"
    elif variant == "per_timestep":
        m = RegimeWorldModel(**base_kw, per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg); kind = "rwm"
    elif variant == "per_ts_lyap":
        m = RegimeWorldModel(**base_kw, per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg, w_lyap_reg=0.5,
                           lyap_targets=(0.9, 1.0, 1.0)); kind = "rwm"
    elif variant == "flat_h64_ms":
        m = FlatBaseline(d_in=d_in, d_h=64); return m, None, "flat"
    elif variant == "lstm_h48_ms":
        m = LSTMBaseline(d_in=d_in, d_h=48); return m, None, "flat"
    else:
        raise ValueError(variant)
    return m, cfg, kind


def train_one(variant, m, cfg, train_ds, kind):
    if kind == "rwm":
        train(m, train_ds, cfg, verbose=False)
    else:
        train_baseline(m, train_ds, n_epochs=30, w_multistep=2.0,
                        K_multistep=4, gamma_multistep=0.9,
                        n_starts_multistep=4)


def split_train_eval(samples, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(len(idx) * train_ratio)
    return [samples[i] for i in idx[:n_tr]], [samples[i] for i in idx[n_tr:]]


def run_one_seed(seed, samples, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_ds, eval_ds = split_train_eval(samples, 0.8, seed=seed)
    from collections import Counter
    cnt = Counter(s.regime for s in samples)
    print(f"  samples: {len(samples)}  regime dist: {dict(cnt)}", flush=True)
    print(f"  train: {len(train_ds)}  eval: {len(eval_ds)}", flush=True)
    results = {"seed": seed}
    for variant in ["baseline", "per_timestep", "per_ts_lyap",
                      "flat_h64_ms", "lstm_h48_ms"]:
        t0 = time.time()
        try:
            m, cfg, kind = build(variant, seed)
            train_one(variant, m, cfg, train_ds, kind)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind=kind)
            results[variant] = {"params": params, "eval": res}
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
    print("SPRINT 6A : REAL BTC + ETH data — RWM variants vs LSTM/Flat baselines", flush=True)
    print("=" * 96, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "10"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    WINDOW_T = int(os.environ.get("FIN_WINDOW", "256"))
    STRIDE = int(os.environ.get("FIN_STRIDE", "64"))
    TICKERS = os.environ.get("FIN_TICKERS", "BTC-USD,ETH-USD").split(",")

    print(f"  tickers={TICKERS}  window_T={WINDOW_T}  stride={STRIDE}"
          f"  seeds={N_SEEDS}  K_max={K_MAX}", flush=True)

    # Build once, reused across seeds (data is identical; randomness is in
    # model init + train/eval split).
    samples = make_real_market_dataset(
        tickers=TICKERS, period="5y", interval="1d",
        window_T=WINDOW_T, stride=STRIDE,
    )
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

    VARIANTS = ["baseline", "per_timestep", "per_ts_lyap",
                 "flat_h64_ms", "lstm_h48_ms"]
    SYNTH_REGIMES = ["smooth", "periodic", "chaotic"]
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
    print(f"K={K+1} MSE per (synth) regime — eval on real BTC+ETH", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'smooth (bull)':>22}{'periodic (sideways)':>22}{'chaotic (bear)':>22}",
          flush=True)
    for variant in VARIANTS:
        row = [f"  {variant:<14}"]
        for r in SYNTH_REGIMES:
            v = collect(variant, r, K)
            if len(v) == 0: row.append(f"{'n/a':>22}")
            else: row.append(f"{v.mean():>12.3e}±{v.std():.0e}({len(v)})".rjust(22))
        print("".join(row), flush=True)

    print(f"\n{'=' * 96}", flush=True)
    print(f"Welch t-tests vs baseline @ K={K+1}", flush=True)
    print("=" * 96, flush=True)
    base = {r: collect("baseline", r, K) for r in SYNTH_REGIMES}
    for variant in VARIANTS:
        if variant == "baseline": continue
        for r in SYNTH_REGIMES:
            v = collect(variant, r, K)
            t, p = welch_ttest(v, base[r])
            if len(v) == 0: eff = "no data"
            elif np.isnan(p): eff = "—"
            else:
                better = v.mean() < base[r].mean()
                arrow = "↓ better" if better else "↑ worse"
                sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                        "*" if p < 0.05 else "ns")
                eff = f"{arrow} {sig}  ({LABELS[r]})"
            tstr = f"{t:>8.2f}" if not np.isnan(t) else f"{'—':>8}"
            pstr = f"{p:>10.4f}" if not np.isnan(p) else f"{'—':>10}"
            print(f"  {variant:<14}{r:>10}{tstr}{pstr}  {eff}", flush=True)

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "window_T": WINDOW_T,
            "tickers": TICKERS, "regime_mapping": FIN_TO_SYNTH,
            "all_runs": all_runs}
    json.dump(out, open("results/regime_world_sprint6_real_finance.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint6_real_finance.json", flush=True)


if __name__ == "__main__":
    main()
