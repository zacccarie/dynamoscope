"""Sprint 6B diagnostic ablation : test if router actually contributes.

Findings to date :
- per_ts_lyap wins p<0.05 on real BTC+ETH bear (Sprint 6A)
- BUT visualization shows the router collapses to ~static 75/20/5 distribution
- The question : is the win driven by routing, or by averaged-mixture dynamics?

Three variants compared (matched protocol with Sprint 6A) :
  A) per_ts_lyap (Sprint 6A winner, reference)
  B) frozen_router : same architecture but router output forced to uniform 1/3
                      (ablates routing entirely)
  C) single_phi    : 1 expert only (n_regimes=1), capacity-matched via wider
                      d_fast/d_slow. Tests if mixture-of-3 even matters.

If (B) MSE ≈ (A) MSE → router is decoration, mixture does the work.
If (C) MSE ≈ (A) MSE → mixture is decoration, single expert does the work.
If (A) >> (B) and (A) >> (C) → both router AND mixture genuinely contribute.

Bonus : log per-trajectory router entropy H(r_t) for (A) to quantify collapse.
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
    base_kw = dict(d_in=d_in, d_fast=32, d_slow=32, slow_stride=4, stable=True,
                    per_timestep_router=True, router_window=4)
    base_cfg = dict(n_epochs=30, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                     w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                     w_regime_sup=5.0, w_multistep=2.0, K_multistep=4,
                     gamma_multistep=0.9, n_starts_multistep=4,
                     w_lyap_reg=0.5, lyap_targets=(0.9, 1.0, 1.0))
    if variant == "per_ts_lyap":
        m = RegimeWorldModel(**base_kw)
    elif variant == "frozen_router":
        m = RegimeWorldModel(**base_kw, frozen_router=True)
    elif variant == "single_phi":
        # n_regimes=1, no taxonomic CE (no labels), no Lyap targets per regime
        cfg_single = dict(base_cfg)
        cfg_single["w_regime_sup"] = 0.0
        cfg_single["lyap_targets"] = (0.9,)
        m = RegimeWorldModel(d_in=d_in, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True, n_regimes=1,
                              per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**cfg_single)
        return m, cfg
    else:
        raise ValueError(variant)
    cfg = TrainConfig(**base_cfg)
    return m, cfg


def split_train_eval(samples, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(len(idx) * train_ratio)
    return [samples[i] for i in idx[:n_tr]], [samples[i] for i in idx[n_tr:]]


def measure_router_entropy(model, samples, n_samples_max=10):
    """Mean entropy of router output across windows, for diagnosing collapse."""
    model.eval()
    if not hasattr(model, "router"):
        return float("nan")
    entropies = []
    with torch.no_grad():
        for s in samples[:n_samples_max]:
            x = torch.from_numpy(s.traj.astype(np.float32))
            x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
            out = model(x)
            r = out["r"]
            if r.dim() == 1: r = r.unsqueeze(0)
            # Skip first 5 steps (warm-up transient)
            r = r[5:] if r.shape[0] > 5 else r
            p = r.clamp_min(1e-9)
            H = -(p * p.log()).sum(dim=-1).mean()
            entropies.append(float(H.item()))
    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def measure_phi_diversity(model, samples, n_samples_max=5):
    """Mean pairwise ‖Φ_i(z) − Φ_j(z)‖ for n_regimes > 1.

    If Φ's are similar → expert collapse, mixture is redundant.
    """
    model.eval()
    if not hasattr(model, "dynamics") or len(model.dynamics.transitions) < 2:
        return float("nan")
    diffs = []
    with torch.no_grad():
        for s in samples[:n_samples_max]:
            x = torch.from_numpy(s.traj.astype(np.float32))
            x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
            out = model(x)
            z = out["z_slow"]
            # Compute Φ_k(z) for each k
            phis = [phi(z) for phi in model.dynamics.transitions]
            # Pairwise L2 differences
            K = len(phis)
            for i in range(K):
                for j in range(i + 1, K):
                    d = (phis[i] - phis[j]).pow(2).mean().sqrt().item()
                    diffs.append(d)
    return float(np.mean(diffs)) if diffs else float("nan")


def run_one_seed(seed, samples, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_ds, eval_ds = split_train_eval(samples, 0.8, seed=seed)
    results = {"seed": seed}
    for variant in ["per_ts_lyap", "frozen_router", "single_phi"]:
        t0 = time.time()
        try:
            m, cfg = build(variant, seed)
            train(m, train_ds, cfg, verbose=False)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            res = eval_model_per_regime(m, eval_ds, K_max=K_max, kind="rwm")
            ent = measure_router_entropy(m, eval_ds)
            div = measure_phi_diversity(m, eval_ds)
            results[variant] = {"params": params, "eval": res,
                                 "router_entropy": ent, "phi_diversity": div}
            print(f"  {variant:<14} ({time.time()-t0:.1f}s, {params/1000:.1f}K, "
                  f"H={ent:.3f}, ΔΦ={div:.4f})", flush=True)
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
    print("SPRINT 6B ABLATION : per_ts_lyap vs frozen_router vs single_phi", flush=True)
    print("Goal : determine if router and mixture actually contribute to wins", flush=True)
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

    VARIANTS = ["per_ts_lyap", "frozen_router", "single_phi"]
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
    print(f"  {'Variant':<14}{'bull (smooth)':>22}{'sideways (periodic)':>22}{'bear (chaotic)':>22}",
          flush=True)
    for variant in VARIANTS:
        row = [f"  {variant:<14}"]
        for r in REGIMES:
            v = collect(variant, r, K)
            if len(v) == 0: row.append(f"{'n/a':>22}")
            else: row.append(f"{v.mean():>12.3e}±{v.std():.0e}({len(v)})".rjust(22))
        print("".join(row), flush=True)

    print(f"\n{'=' * 96}", flush=True)
    print("Diagnostic metrics : router entropy H(r), Φ pairwise distance ΔΦ",
          flush=True)
    print(f"  max H = log(3) ≈ 1.099 (uniform);  ΔΦ ≈ 0 = expert collapse",
          flush=True)
    print("=" * 96, flush=True)
    for variant in VARIANTS:
        ents = [r[variant].get("router_entropy", float("nan")) for r in all_runs
                if variant in r and "failed" not in r[variant]]
        divs = [r[variant].get("phi_diversity", float("nan")) for r in all_runs
                if variant in r and "failed" not in r[variant]]
        ents = np.array([e for e in ents if not np.isnan(e)])
        divs = np.array([d for d in divs if not np.isnan(d)])
        if len(ents):
            print(f"  {variant:<14} H={ents.mean():.3f}±{ents.std():.3f}  "
                  f"ΔΦ={divs.mean():.4f}±{divs.std():.4f}  (n={len(ents)})",
                  flush=True)

    print(f"\n{'=' * 96}", flush=True)
    print("Welch t-tests : frozen_router and single_phi vs per_ts_lyap",
          flush=True)
    print("(if p > 0.05 → ablated variant is NOT significantly worse → ablation OK)",
          flush=True)
    print("=" * 96, flush=True)
    base = {r: collect("per_ts_lyap", r, K) for r in REGIMES}
    for variant in ["frozen_router", "single_phi"]:
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
                eff = f"{arrow} {sig}"
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
    json.dump(out, open("results/regime_world_sprint6b_ablation.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint6b_ablation.json", flush=True)


if __name__ == "__main__":
    main()
