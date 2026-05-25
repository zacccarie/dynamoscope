"""Sprint 7 : per-ingredient ablation + Sharpe metric.

Sprint 6B reframe identified 4 architectural ingredients (single_phi):
  (a) stable        : spectral_norm + tanh-bounded delta on Φ
  (b) per_timestep  : sliding-window descriptors per step
  (c) lyap_reg      : Jacobian σ_max penalty
  (d) multistep     : K-step rollout training loss

Sprint 7 turns each off in isolation to measure individual contribution.

Variants (n_regimes=1 = single_phi family) :
  A) full         — all 4 on (Sprint 6B winner reference)
  B) no_stable    — stable=False (raw residual MLP)
  C) no_per_ts    — per_timestep_router=False (global descriptors only)
  D) no_lyap      — w_lyap_reg=0
  E) no_multistep — w_multistep=0
  F) naive        — all 4 off (baseline single-Φ minus all stability)

Downstream metric : naive long/short trading Sharpe + crash recall@K=5
(days with |return| > 5% — does the model predict them K steps ahead?).
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
from experiments.regime_world_rollout_attractor import eval_model_per_regime

FIN_TO_SYNTH = {"bull": "smooth", "sideways": "periodic", "bear": "chaotic"}


def relabel(samples):
    for s in samples:
        dom_idx = int(np.bincount(s.regime_labels, minlength=3).argmax())
        s.regime = FIN_TO_SYNTH[FIN_REGIMES[dom_idx]]
    return samples


def build(variant, seed, d_in=5):
    torch.manual_seed(seed); np.random.seed(seed)
    # All variants are single_phi (n_regimes=1).
    base_kw = dict(d_in=d_in, d_fast=32, d_slow=32, slow_stride=4,
                    n_regimes=1)
    # Default config (all ingredients on) :
    base_cfg = dict(n_epochs=30, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                     w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                     w_regime_sup=0.0, w_multistep=2.0, K_multistep=4,
                     gamma_multistep=0.9, n_starts_multistep=4,
                     w_lyap_reg=0.5, lyap_targets=(0.9,))
    if variant == "full":
        m = RegimeWorldModel(**base_kw, stable=True,
                              per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg)
    elif variant == "no_stable":
        m = RegimeWorldModel(**base_kw, stable=False,
                              per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg)
    elif variant == "no_per_ts":
        m = RegimeWorldModel(**base_kw, stable=True,
                              per_timestep_router=False)
        cfg = TrainConfig(**base_cfg)
    elif variant == "no_lyap":
        cfg_kw = dict(base_cfg); cfg_kw["w_lyap_reg"] = 0.0
        m = RegimeWorldModel(**base_kw, stable=True,
                              per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**cfg_kw)
    elif variant == "no_multistep":
        cfg_kw = dict(base_cfg); cfg_kw["w_multistep"] = 0.0
        m = RegimeWorldModel(**base_kw, stable=True,
                              per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**cfg_kw)
    elif variant == "naive":
        # All 4 ingredients off
        cfg_kw = dict(base_cfg); cfg_kw["w_lyap_reg"] = 0.0; cfg_kw["w_multistep"] = 0.0
        m = RegimeWorldModel(**base_kw, stable=False,
                              per_timestep_router=False)
        cfg = TrainConfig(**cfg_kw)
    else:
        raise ValueError(variant)
    return m, cfg


def compute_sharpe_and_crash_DISABLED(model, samples, K_pred: int = 5,
                              crash_threshold: float = 0.05) -> dict:
    """Downstream metric DISABLED in Sprint 7 — turned out bogus.

    Signal extraction from z_slow_pred latent space doesn't actually
    decode to predicted log-return. The "long if pred > 0" strategy
    collapsed to "always long BTC" which yields ~2.15 Sharpe on the
    bull-biased 5y period regardless of model. Need a proper return-
    prediction head (z_slow → next-step log_return scalar) for valid
    Sharpe, deferred to future work.

    Kept as STUB to record the failure mode (F20).

    For each window :
      - At each time t with at least K_pred future steps :
        * Run model.rollout(context=samples up to t, K=K_pred)
        * pred_return_K = pred[K_pred-1][0] (feature 0 = log_return)
        * Position : +1 if pred_return_K > 0, else -1 (long/short)
        * Realized return at t+K_pred = sample.return_series[t+K_pred]
        * strategy_return = position × realized_return - tx_cost
      - Sharpe = mean(strategy_return) / std(strategy_return) * sqrt(252)
      - Crash recall : among days t+K with |actual_return| > threshold,
        what fraction was correctly predicted by the model (i.e., model
        predicted same sign of return)?
    """
    model.eval()
    all_strategy_rets = []
    crash_correct = 0
    crash_total = 0
    n_obs = 0
    T_ctx = 16  # context window = 16 frames for rollout
    tx_cost = 0.0005  # 5 bps per trade

    with torch.no_grad():
        for s in samples:
            x = torch.from_numpy(s.traj.astype(np.float32))
            x_norm = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
            T = x.shape[0]
            if T < T_ctx + K_pred * 4 + 4:
                continue
            # Sliding window predictions
            for t in range(T_ctx, T - K_pred * 4 - 4, K_pred * 4):
                try:
                    # Run model forward up to t for context
                    if hasattr(model, 'rollout'):
                        ctx = x_norm[:t]
                        out = model.rollout(ctx, K_pred)
                        pred_features = out["z_slow_pred"]  # (K_pred, d_slow)
                    else:
                        continue
                    # Decode z_slow to feature space via decoder, if available
                    if hasattr(model, 'decoder') and pred_features.shape[-1] != x.shape[-1]:
                        # Cannot directly invert encoder→decoder, use sign of mean
                        pred_signal = pred_features[K_pred - 1].mean().item()
                    else:
                        pred_signal = pred_features[K_pred - 1, 0].item()
                    actual_return = float(s.return_series[t + K_pred * 4])
                    position = 1.0 if pred_signal > 0 else -1.0
                    strategy_return = position * actual_return - tx_cost
                    all_strategy_rets.append(strategy_return)
                    n_obs += 1
                    if abs(actual_return) > crash_threshold:
                        crash_total += 1
                        # Correct = model predicted same sign
                        if (pred_signal > 0) == (actual_return > 0):
                            crash_correct += 1
                except Exception:
                    continue
    if n_obs == 0:
        return {"sharpe": float("nan"), "n_obs": 0,
                "crash_recall": float("nan"), "crash_total": 0,
                "mean_return": float("nan"), "std_return": float("nan")}
    rets = np.array(all_strategy_rets)
    mean_r = rets.mean()
    std_r = rets.std() + 1e-9
    sharpe = mean_r / std_r * np.sqrt(252)  # annualize for daily
    crash_recall = (crash_correct / max(crash_total, 1)) if crash_total > 0 else float("nan")
    return {
        "sharpe": float(sharpe), "n_obs": int(n_obs),
        "mean_return": float(mean_r), "std_return": float(std_r),
        "crash_recall": float(crash_recall),
        "crash_total": int(crash_total), "crash_correct": int(crash_correct),
    }


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
    for variant in ["full", "no_stable", "no_per_ts", "no_lyap",
                      "no_multistep", "naive"]:
        t0 = time.time()
        try:
            m, cfg = build(variant, seed)
            train(m, train_ds, cfg, verbose=False)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            mse_res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind="rwm")
            results[variant] = {"params": params, "eval": mse_res}
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
    print("SPRINT 7 : per-ingredient ablation + Sharpe metric on real BTC+ETH", flush=True)
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

    VARIANTS = ["full", "no_stable", "no_per_ts", "no_lyap", "no_multistep", "naive"]
    REGIMES = ["smooth", "periodic", "chaotic"]
    LABELS = {"smooth": "bull", "periodic": "sideways", "chaotic": "bear"}
    K = K_MAX - 1

    def collect_mse(variant, regime, k):
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
    print(f"K={K+1} MSE per regime (BTC+ETH real)", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'bull':>20}{'sideways':>20}{'bear':>20}", flush=True)
    for variant in VARIANTS:
        row = [f"  {variant:<14}"]
        for r in REGIMES:
            v = collect_mse(variant, r, K)
            if len(v) == 0: row.append(f"{'n/a':>20}")
            else: row.append(f"{v.mean():>10.3e}±{v.std():.0e}({len(v)})".rjust(20))
        print("".join(row), flush=True)

    print(f"\n{'=' * 96}", flush=True)
    print("Welch t-tests vs 'full' — MSE K=24 per regime", flush=True)
    print("=" * 96, flush=True)
    base = {r: collect_mse("full", r, K) for r in REGIMES}
    for variant in VARIANTS:
        if variant == "full": continue
        for r in REGIMES:
            v = collect_mse(variant, r, K)
            t, p = welch_ttest(v, base[r])
            if len(v) == 0: eff = "no data"
            elif np.isnan(p): eff = "—"
            else:
                worse = v.mean() > base[r].mean()
                arrow = "↑ worse" if worse else "↓ better"
                sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                        "*" if p < 0.05 else "ns")
                eff = f"{arrow} {sig}  ({LABELS[r]})"
            tstr = f"{t:>8.2f}" if not np.isnan(t) else f"{'—':>8}"
            pstr = f"{p:>10.4f}" if not np.isnan(p) else f"{'—':>10}"
            print(f"  {variant:<14}{LABELS[r]:>10}{tstr}{pstr}  {eff}", flush=True)

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "all_runs": all_runs}
    json.dump(out, open("results/regime_world_sprint7_ingredients.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint7_ingredients.json", flush=True)


if __name__ == "__main__":
    main()
