"""Sprint 8 : return decoder (F20 fix) + cross-asset transfer.

Adds a small return_head (z_slow → predicted next-window log-return) to
the single-expert RWM. Trained jointly with the world model on
BTC + ETH. Evaluated on :
  - in-sample (BTC + ETH held-out windows)
  - cross-asset (SPY, QQQ, GLD) — different market dynamics

Metrics per asset :
  - K=24 rollout MSE (latent-space, model's own decoded feature error)
  - Predicted-return MSE vs actual cumulative log return (downstream signal)
  - Sharpe of long/short strategy (sign of predicted return → ±1 position
    held for next slow window, multiplied by actual cumulative log return)
  - Crash recall = P(pred_return < q05_threshold | actual_return < -3σ)

Baselines :
  - constant 'always long' (BTC bull-market control)
  - random sign (Sharpe should be ~0)

This validates that the architecture's latent dynamics carry an actually
useful return signal AND that it transfers across asset classes.
"""
from __future__ import annotations
import json
import sys
import time
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.regime_world import RegimeWorldModel, TrainConfig, train
from backend.regime_world.finance_real import make_real_market_dataset
from backend.regime_world.finance import REGIMES as FIN_REGIMES

# Mapping reuse from Sprint 6B
FIN_TO_SYNTH = {"bull": "smooth", "sideways": "periodic", "bear": "chaotic"}


def relabel(samples):
    """Map per-step heuristic finance labels → dominant synth regime."""
    for s in samples:
        dom_idx = int(np.bincount(s.regime_labels, minlength=3).argmax())
        s.regime = FIN_TO_SYNTH[FIN_REGIMES[dom_idx]]
    return samples


def build_model(seed, d_in=5, slow_stride=4):
    torch.manual_seed(seed); np.random.seed(seed)
    # single_phi base (Sprint 6B winner) + return_decoder head
    m = RegimeWorldModel(d_in=d_in, d_fast=32, d_slow=32,
                          slow_stride=slow_stride, n_regimes=1,
                          stable=True, per_timestep_router=True, router_window=4,
                          return_decoder=True, return_dim=1)
    cfg = TrainConfig(
        n_epochs=30, lr=3e-4, w_dyn=1.0, w_slow=1.0,
        w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
        w_regime_sup=0.0, w_multistep=2.0, K_multistep=4,
        gamma_multistep=0.9, n_starts_multistep=4,
        w_lyap_reg=0.5, lyap_targets=(0.9,),
        w_return_decode=1.0, return_slow_stride=slow_stride,
    )
    return m, cfg


def split_train_eval(samples, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(len(idx) * train_ratio)
    return [samples[i] for i in idx[:n_tr]], [samples[i] for i in idx[n_tr:]]


@torch.no_grad()
def eval_return_metrics(model, dataset, K_max=24, slow_stride=4,
                         n_ctx_slow=4, device="cpu"):
    """Rolling 1-step return forecast : forward() once par sample → return_pred
    à chaque slow step. Strategy = sign(pred) × actual return de la prochaine
    fenêtre slow. Sharpe agrégé sur toutes les prédictions.

    Honest design: à chaque slow step t, z_slow[t] est issu d'un contexte
    raw [0, t*S) — equivalent à un 1-step forecast réel. Pas de rollout
    récursif (qui collapse vers la moyenne sur K=24).
    """
    model = model.eval().to(device)
    n_per_year_slow = 252.0 / slow_stride
    strat_returns = []
    pred_returns = []
    actual_returns = []
    pred_at_horizon = []  # for K=24 rollout-endpoint MSE
    actual_at_horizon = []
    for s in dataset:
        x = torch.from_numpy(s.traj).float().to(device)
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        T = x.shape[0]
        T_s = T // slow_stride
        if T_s < n_ctx_slow + 2:
            continue
        out = model(x)
        if "return_pred_from_z" not in out:
            continue
        # return_pred_from_z[t] = decoded(z_slow[t]) = predicted next-window
        # log-return given context [0, t*S]. Aligned with actual_return[t]
        # = sum(return_series[(t+1)*S : (t+2)*S]).
        r_pred = out["return_pred_from_z"].cpu().numpy()   # (T_s,)
        ret_series = s.return_series
        for t in range(n_ctx_slow, T_s - 1):
            lo = (t + 1) * slow_stride
            hi = lo + slow_stride
            if hi > len(ret_series):
                break
            actual = float(ret_series[lo:hi].sum())
            pred = float(r_pred[t])
            strat = float(np.sign(pred) * actual)
            strat_returns.append(strat)
            pred_returns.append(pred)
            actual_returns.append(actual)
        # K=K_max horizon : prédire à slow step (T_s - K_max - 1) le cumul
        # log-return des K_max prochaines fenêtres slow.
        t_anchor = T_s - K_max - 2
        if t_anchor > n_ctx_slow:
            lo_h = (t_anchor + 1) * slow_stride
            hi_h = lo_h + K_max * slow_stride
            if hi_h <= len(ret_series):
                actual_at_horizon.append(float(ret_series[lo_h:hi_h].sum()))
                pred_at_horizon.append(float(r_pred[t_anchor]))
    if len(strat_returns) == 0:
        return {"sharpe": float("nan"), "crash_recall": float("nan"),
                "horizon_mse": float("nan"), "n_samples": 0,
                "mean_strat_return_bp": float("nan"),
                "always_long_sharpe": float("nan"),
                "direction_acc": float("nan")}
    sr = np.asarray(strat_returns); pr = np.asarray(pred_returns)
    ar = np.asarray(actual_returns)
    mu, sigma = sr.mean(), sr.std() + 1e-12
    sharpe = float(mu / sigma * np.sqrt(n_per_year_slow))
    al_mu, al_sigma = ar.mean(), ar.std() + 1e-12
    al_sharpe = float(al_mu / al_sigma * np.sqrt(n_per_year_slow))
    # Direction accuracy : P(sign(pred) == sign(actual))
    dir_acc = float((np.sign(pr) == np.sign(ar)).mean())
    # Crash recall : crash = actual < mean - 3σ
    crash_thresh = ar.mean() - 3.0 * ar.std()
    crash_mask = ar < crash_thresh
    if crash_mask.sum() == 0:
        crash_recall = float("nan")
    else:
        pred_crash_thresh = np.percentile(pr, 5)
        pred_crash_mask = pr < pred_crash_thresh
        crash_recall = float((crash_mask & pred_crash_mask).sum()) / float(crash_mask.sum())
    # Horizon MSE (K=K_max rollout end-step)
    if len(pred_at_horizon) > 0:
        ph = np.asarray(pred_at_horizon); ah = np.asarray(actual_at_horizon)
        horizon_mse = float(((ph - ah) ** 2).mean())
    else:
        horizon_mse = float("nan")
    return {
        "sharpe": sharpe,
        "always_long_sharpe": al_sharpe,
        "direction_acc": dir_acc,
        "crash_recall": crash_recall,
        "horizon_mse": horizon_mse,
        "n_samples": int(len(strat_returns)),
        "mean_strat_return_bp": float(mu * 10000.0),
    }


def run_one_seed(seed, train_assets, eval_assets_map, K_max=24, slow_stride=4):
    print(f"\n=== SEED {seed} ===", flush=True)
    results = {"seed": seed}
    # ------- Train on BTC + ETH -------
    train_full = make_real_market_dataset(tickers=train_assets,
                                            period="5y", window_T=256, stride=64)
    train_full = relabel(train_full)
    train_ds, in_eval_ds = split_train_eval(train_full, 0.8, seed=seed)
    print(f"  train_assets {train_assets}: {len(train_ds)}/{len(in_eval_ds)} train/eval", flush=True)
    t0 = time.time()
    m, cfg = build_model(seed, slow_stride=slow_stride)
    train(m, train_ds, cfg, verbose=False)
    p = sum(x.numel() for x in m.parameters() if x.requires_grad)
    print(f"  train ({time.time()-t0:.1f}s, {p/1000:.1f}K params)", flush=True)
    # ------- Eval in-sample (held-out windows on train tickers) -------
    res_in = eval_return_metrics(m, in_eval_ds, K_max=K_max, slow_stride=slow_stride)
    results["in_sample"] = res_in
    print(f"  IN-SAMPLE (BTC+ETH held-out, n={res_in['n_samples']}) "
          f"Sharpe={res_in['sharpe']:+.2f} (long-only={res_in['always_long_sharpe']:+.2f}) "
          f"crash_recall={res_in['crash_recall']:.2f} dir_acc={res_in['direction_acc']:.2f} horizon_mse={res_in['horizon_mse']:.4f}",
          flush=True)
    # ------- Eval cross-asset -------
    for label, tickers in eval_assets_map.items():
        try:
            ds = make_real_market_dataset(tickers=tickers, period="5y",
                                            window_T=256, stride=64)
            ds = relabel(ds)
            res = eval_return_metrics(m, ds, K_max=K_max, slow_stride=slow_stride)
            results[f"cross_{label}"] = res
            print(f"  CROSS [{label:<5}] (n={res['n_samples']}) "
                  f"Sharpe={res['sharpe']:+.2f} (long-only={res['always_long_sharpe']:+.2f}) "
                  f"crash_recall={res['crash_recall']:.2f} dir_acc={res['direction_acc']:.2f} horizon_mse={res['horizon_mse']:.4f}",
                  flush=True)
        except Exception as e:
            print(f"  CROSS [{label}] FAILED: {e}", flush=True)
            results[f"cross_{label}"] = {"failed": str(e)}
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
    print("SPRINT 8 : return decoder + cross-asset transfer on real markets", flush=True)
    print("=" * 96, flush=True)
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "10"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    train_assets = ["BTC-USD", "ETH-USD"]
    eval_assets_map = {
        "SPY":  ["SPY"],
        "QQQ":  ["QQQ"],
        "GLD":  ["GLD"],
    }
    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, train_assets, eval_assets_map,
                                            K_max=K_MAX, slow_stride=4))
        except Exception as e:
            print(f"  seed {seed} fatal: {e}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n[time] total {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    if not all_runs: return

    DOMAINS = ["in_sample", "cross_SPY", "cross_QQQ", "cross_GLD"]
    METRICS = ["sharpe", "always_long_sharpe", "direction_acc", "crash_recall",
                "horizon_mse", "mean_strat_return_bp"]

    print(f"\n{'=' * 96}", flush=True)
    print(f"Aggregated metrics ({N_SEEDS} seeds, n_per_year_slow = 63)", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Domain':<15}{'Sharpe':>14}{'long-only':>14}"
          f"{'Dir acc':>12}{'Crash R':>10}{'μ ret bp':>12}", flush=True)
    for d in DOMAINS:
        vals = {m: [] for m in METRICS}
        for run in all_runs:
            r = run.get(d, {})
            if "failed" in r: continue
            for m in METRICS:
                v = r.get(m, None)
                if v is not None and np.isfinite(v):
                    vals[m].append(v)
        def fmt(arr, w=14, prec=3):
            if len(arr) == 0: return "n/a".rjust(w)
            return f"{np.mean(arr):+.{prec}f}±{np.std(arr):.2f}".rjust(w)
        row = (f"  {d:<15}{fmt(vals['sharpe'])}{fmt(vals['always_long_sharpe'])}"
                f"{fmt(vals['direction_acc'], 12)}"
                f"{fmt(vals['crash_recall'], 10, 2)}"
                f"{fmt(vals['mean_strat_return_bp'], 12, 1)}")
        print(row, flush=True)

    # Welch test : model Sharpe vs always-long Sharpe (per-domain)
    print(f"\n{'=' * 96}", flush=True)
    print("Welch t-tests : model Sharpe vs long-only Sharpe (same domain)", flush=True)
    print("=" * 96, flush=True)
    for d in DOMAINS:
        a, b = [], []
        for run in all_runs:
            r = run.get(d, {})
            if "failed" in r: continue
            sa, sb = r.get("sharpe"), r.get("always_long_sharpe")
            if sa is not None and np.isfinite(sa): a.append(sa)
            if sb is not None and np.isfinite(sb): b.append(sb)
        t, p = welch_ttest(a, b)
        if len(a) and len(b):
            diff = np.mean(a) - np.mean(b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            arrow = "↑" if diff > 0 else "↓"
            print(f"  {d:<15}t={t:>+7.2f}  p={p:>7.4f}  Δ={diff:+.2f} {arrow} {sig}",
                  flush=True)
        else:
            print(f"  {d:<15}— insufficient data", flush=True)

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "train_assets": train_assets,
           "eval_assets_map": eval_assets_map, "all_runs": all_runs}
    json.dump(out, open("results/regime_world_sprint8_return_decoder.json", "w"),
              indent=2, default=_d)
    print("\n[saved] results/regime_world_sprint8_return_decoder.json", flush=True)


if __name__ == "__main__":
    main()
