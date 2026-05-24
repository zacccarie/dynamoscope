"""Sprint 2 architectural exploration.

Conditions (10 seeds, mixed-regime synth ou mono selon variant) :

  A) baseline           : Sprint 1 stab+ms (K=3, hand-crafted desc, mono-train)
  B) +learned_desc      : A but LearnedDescriptorHeads (d_out=8, MLP)
  C) +uncertainty       : A but UncertaintyWeighting wraps the loss components
  D) +mixed_train       : A but trained AND eval'd on MixedTrajSample data
                          (transitions smooth↔periodic↔chaotic mid-flow,
                           per-step regime labels available but not used
                           by current losses → tests purely arch handling)

Evaluation : Welch t-tests vs A baseline on K=24 rollout MSE.
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
    make_dataset, make_mixed_dataset,
    UncertaintyWeighting,
)
from backend.regime_world.losses import regime_conditional_loss
from backend.regime_world.synth import REGIME_TO_IDX
from experiments.regime_world_probe import pad_to_d_in
from experiments.regime_world_rollout_attractor import eval_model_per_regime


def make_dataset_for_variant(variant, seed, n_per_regime=12, T=128):
    if variant in ("baseline", "learned_desc", "uncertainty"):
        ds = make_dataset(n_per_regime=n_per_regime, T=T,
                          base_seed=seed * 100)
        for s in ds: pad_to_d_in(s, 3)
        return ds
    if variant == "mixed_train":
        # Mixed-regime trajectories. n_samples ≈ same as 3×n_per_regime.
        ds = make_mixed_dataset(n_samples=3 * n_per_regime, T=T,
                                 base_seed=seed * 100)
        # MixedTrajSample already 3-D
        return ds
    raise ValueError(variant)


def build(variant, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if variant == "baseline":
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=5.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg
    if variant == "learned_desc":
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True, learned_descriptors=True,
                              learned_desc_dim=8)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=5.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg
    if variant == "uncertainty":
        # Same architecture as baseline, but trainer wraps loss sum in
        # UncertaintyWeighting. We use a tiny custom training loop here
        # rather than threading UW through trainer.train() — keeps the
        # change local.
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=5.0, w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg
    if variant == "mixed_train":
        # Baseline architecture, but data is mixed-regime trajectories.
        # We don't have per-trajectory regime label any more (it's "mixed"),
        # so set w_regime_sup=0 — train fully unsupervised on regimes.
        m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                              stable=True)
        cfg = TrainConfig(n_epochs=40, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                           w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                           w_regime_sup=0.0,  # no labels for mixed
                           w_multistep=2.0,
                           K_multistep=4, gamma_multistep=0.9,
                           n_starts_multistep=4)
        return m, cfg
    raise ValueError(variant)


def train_uncertainty_variant(model, dataset, cfg):
    """Custom training loop with UncertaintyWeighting wrapping 5 loss terms :
    [recon, dyn, slow, recur, lyap]. Other terms (regime_sup, multistep,
    entropy) kept with fixed weights.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    uw = UncertaintyWeighting(n_tasks=5).to(device)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(uw.parameters()), lr=cfg.lr,
    )
    s1 = int(0.3 * cfg.n_epochs); s2 = int(0.6 * cfg.n_epochs)
    for epoch in range(cfg.n_epochs):
        # Stage weights (warmup mirrors trainer.stage_weights)
        if epoch < s1:
            w_ms = 0.0
            stage_slow, stage_recur, stage_lyap = 0.0, 0.0, 0.0
        elif epoch < s2:
            w_ms = 0.5 * cfg.w_multistep
            stage_slow, stage_recur, stage_lyap = 1.0, 1.0, 0.3
        else:
            w_ms = cfg.w_multistep
            stage_slow, stage_recur, stage_lyap = 1.0, 1.0, 1.0
        for sample in dataset:
            x = torch.from_numpy(sample.traj).float().to(device)
            x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
            out = model(x)
            regime_idx = REGIME_TO_IDX.get(sample.regime, 0)
            ld = regime_conditional_loss(
                out, x,
                w_recon=1.0, w_dyn=1.0, w_slow=stage_slow,
                w_recur=stage_recur * 0.5, w_lyap=stage_lyap * 0.3,
                w_entropy=0.0, w_regime_sup=cfg.w_regime_sup,
                w_multistep=w_ms, K_multistep=cfg.K_multistep,
                gamma_multistep=cfg.gamma_multistep,
                n_starts_multistep=cfg.n_starts_multistep,
                dynamics_module=model.dynamics, regime_idx=regime_idx,
                target_log_growth_chaotic=cfg.target_log_growth_chaotic,
                target_DET_periodic=cfg.target_DET_periodic,
            )
            # 5 base tasks for UncertaintyWeighting
            tasks = [ld["recon"], ld["dyn"], ld["slow_weighted"],
                      ld["recur_weighted"], ld["lyap_weighted"]]
            # Fixed-weight terms : regime_sup, multistep, entropy, load_balance
            loss = uw(tasks) + ld["regime_sup"] + ld["multistep"] \
                   + ld["entropy_reg"] + ld["load_balance"]
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
    return uw


def train_one(variant, model, cfg, dataset):
    if variant == "uncertainty":
        uw = train_uncertainty_variant(model, dataset, cfg)
        return model, uw
    train(model, dataset, cfg, verbose=False)
    return model, None


def run_one_seed(seed, n_per_regime=12, n_epochs=40, T=128, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    # Common eval set : mono-regime held out
    eval_ds_mono = make_dataset(n_per_regime=8, T=T, base_seed=seed * 100 + 5000)
    for s in eval_ds_mono: pad_to_d_in(s, 3)
    # Mixed-regime eval set (out-of-distribution for variants A-C)
    eval_ds_mixed = make_mixed_dataset(n_samples=24, T=T,
                                         base_seed=seed * 100 + 9000)
    results = {"seed": seed}
    for variant in ["baseline", "learned_desc", "uncertainty", "mixed_train"]:
        t0 = time.time()
        try:
            train_ds = make_dataset_for_variant(variant, seed,
                                                  n_per_regime=n_per_regime, T=T)
            m, cfg = build(variant, seed)
            m, _ = train_one(variant, m, cfg, train_ds)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            # Eval on BOTH mono and mixed
            res_mono = eval_model_per_regime(m, eval_ds_mono, K_max=K_max, kind="rwm")
            res_mixed = eval_model_per_regime(m, eval_ds_mixed, K_max=K_max, kind="rwm")
            results[variant] = {
                "params": params, "mono": res_mono, "mixed": res_mixed,
            }
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
    print("SPRINT 2 : baseline vs learned_desc vs uncertainty vs mixed_train", flush=True)
    print("=" * 96, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "10"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    N_EPOCHS = int(os.environ.get("RWM_N_EPOCHS", "40"))

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, n_epochs=N_EPOCHS, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} fatal: {e}", flush=True)
    print(f"\n[time] total {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    if not all_runs: return

    VARIANTS = ["baseline", "learned_desc", "uncertainty", "mixed_train"]
    REGIMES = ["smooth", "periodic", "chaotic", "mixed"]
    K = K_MAX - 1

    def collect(variant, eval_set, regime, k):
        vals = []
        for run in all_runs:
            res = run.get(variant, {})
            if "failed" in res: continue
            inner = res.get(eval_set, {}).get("mse_per_regime", {})
            mse = inner.get(regime, None)
            if mse is None or k >= len(mse): continue
            v = mse[k]
            if np.isfinite(v) and abs(v) < 1e15: vals.append(v)
        return np.array(vals)

    print("\n" + "=" * 96, flush=True)
    print(f"K={K+1} MSE per regime, eval set = mono", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'smooth':>20}{'periodic':>20}{'chaotic':>20}",
          flush=True)
    for variant in VARIANTS:
        row = [f"  {variant:<14}"]
        for regime in REGIMES[:3]:
            v = collect(variant, "mono", regime, K)
            if len(v) == 0: row.append(f"{'n/a':>20}")
            else: row.append(f"{v.mean():>10.3e}±{v.std():.0e}({len(v)})".rjust(20))
        print("".join(row), flush=True)

    print("\n" + "=" * 96, flush=True)
    print(f"K={K+1} MSE per regime, eval set = mixed (OOD for A-C)", flush=True)
    print("=" * 96, flush=True)
    for variant in VARIANTS:
        v = collect(variant, "mixed", "mixed", K)
        if len(v) == 0:
            print(f"  {variant:<14} mixed: n/a", flush=True)
        else:
            print(f"  {variant:<14} mixed K{K+1}: {v.mean():.3e} ± {v.std():.0e}"
                  f"  ({len(v)} seeds)", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("Welch t-test vs baseline (mono-eval, K=24)", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'regime':>10}{'t':>8}{'p':>10}  effect", flush=True)
    base = {r: collect("baseline", "mono", r, K) for r in REGIMES[:3]}
    for variant in VARIANTS:
        if variant == "baseline": continue
        for regime in REGIMES[:3]:
            v = collect(variant, "mono", regime, K)
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

    # Save full summary
    summary = {}
    for variant in VARIANTS:
        s = {"mono": {}, "mixed": {}}
        for eval_set in ["mono", "mixed"]:
            for regime in REGIMES if eval_set == "mixed" else REGIMES[:3]:
                stack = []
                for run in all_runs:
                    res = run.get(variant, {})
                    if "failed" in res: continue
                    mse = res.get(eval_set, {}).get("mse_per_regime", {}).get(regime, None)
                    if mse is None: continue
                    arr = np.asarray(mse, dtype=float)
                    if np.isfinite(arr).all() and (np.abs(arr) < 1e15).all():
                        stack.append(arr)
                if stack:
                    a = np.array(stack)
                    s[eval_set][regime] = {
                        "mean": a.mean(axis=0).tolist(),
                        "std": a.std(axis=0).tolist(),
                        "n": int(a.shape[0]),
                    }
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
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX, "n_epochs": N_EPOCHS,
            "all_runs": all_runs, "summary": summary}
    json.dump(out, open("results/regime_world_sprint2.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_sprint2.json", flush=True)


if __name__ == "__main__":
    main()
