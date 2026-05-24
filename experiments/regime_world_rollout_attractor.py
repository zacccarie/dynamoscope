"""Validation profonde du fix F10. Reviewer-grade.

Quatre modèles, 8 seeds, MSE + attractor preservation per regime.

Modèles :
  RWM-orig    : stable=False  w_ms=0  (paper baseline = diverge)
  RWM-stab+ms : stable=True   w_ms=2  (le fix proposé)
  Flat-56     : matched-cap   w_ms=0  (capacity baseline)
  Flat-56+ms  : matched-cap   w_ms=2  (CAPACITY-FAIRNESS CONTROL —
                                       isole architecture de training loss)

Métriques :
  MSE per K ∈ {1..24} per regime
  Attractor : bounded, spectral_match, autocorr_match, DET_match, lyap_match
  Tous reportés mean ± std sur 8 seeds, breakdown par régime.

Pré-empte les critiques reviewer :
  R1. "more stable because larger" → matched params (30K).
  R2. "fix is the multi-step loss, not architecture" → Flat-56+ms control.
  R3. "low MSE but wrong attractor" → attractor preservation reported.
  R4. "lucky init" → 8 seeds + variance reported.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
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
from experiments.attractor_metrics import (
    attractor_score, aggregate_attractor,
)


def train_rwm(seed, train_ds, n_epochs, *, stable, w_multistep):
    torch.manual_seed(seed); np.random.seed(seed)
    m = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4,
                          stable=stable)
    cfg = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                       w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                       w_entropy=0.0, w_regime_sup=5.0,
                       w_multistep=w_multistep, K_multistep=4,
                       gamma_multistep=0.9, n_starts_multistep=4)
    train(m, train_ds, cfg, verbose=False)
    return m


def train_flat(seed, train_ds, n_epochs, *, w_multistep):
    torch.manual_seed(seed); np.random.seed(seed)
    f = FlatBaseline(d_in=3, d_h=56)
    train_baseline(f, train_ds, n_epochs=n_epochs,
                    w_multistep=w_multistep, K_multistep=4,
                    gamma_multistep=0.9, n_starts_multistep=4)
    return f


@torch.no_grad()
def eval_model_per_regime(model, dataset, K_max=24, n_ctx_slow=4, kind="rwm"
                           ) -> dict:
    """Per-sample : MSE per K + attractor metrics. Aggregated per regime."""
    T_ctx = n_ctx_slow * 4
    mse_per_regime = defaultdict(lambda: np.zeros(K_max))
    cnt_per_regime = defaultdict(lambda: np.zeros(K_max))
    attractor_per_regime = defaultdict(list)
    for sample in dataset:
        regime = sample.regime
        x = torch.from_numpy(sample.traj).float()
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        T = x.shape[0]
        if T < T_ctx + K_max * 4: continue
        x_context = x[:T_ctx]
        x_full = x[:T_ctx + K_max * 4]

        # GT slow trajectory
        if kind == "rwm":
            out_full = model(x_full)
            gt_full = out_full["z_slow"]
            n_ctx_steps = T_ctx // 4
            gt_target = gt_full[n_ctx_steps:n_ctx_steps + K_max]
        else:
            out_full = model(x_full)
            gt_full = out_full["z_seq"]
            gt_target = gt_full[T_ctx:T_ctx + K_max]

        # Rollout
        if kind == "rwm":
            out = model.rollout(x_context, K_max)
            pred = out["z_slow_pred"]
        else:
            out = model(x_context)
            h = out["z_seq"][-1]
            preds = []
            for _ in range(K_max):
                h_next = model.proj_next(h)
                preds.append(h_next)
                h = h_next
            pred = torch.stack(preds)

        K_actual = min(pred.shape[0], gt_target.shape[0])
        for k in range(K_actual):
            mse_k = ((pred[k] - gt_target[k]) ** 2).mean().item()
            mse_per_regime[regime][k] += mse_k
            cnt_per_regime[regime][k] += 1

        # Attractor metrics on full K_max rollout
        pred_np = pred.cpu().numpy()
        gt_np = gt_target.cpu().numpy()
        if pred_np.shape[0] >= 8 and gt_np.shape[0] >= 8:
            sc = attractor_score(pred_np, gt_np)
            attractor_per_regime[regime].append(sc)

    out_mse = {}
    for regime in mse_per_regime:
        out_mse[regime] = (mse_per_regime[regime]
                            / np.maximum(cnt_per_regime[regime], 1)).tolist()
    out_attr = {regime: aggregate_attractor(scs) if scs else {}
                for regime, scs in attractor_per_regime.items()}
    return {"mse_per_regime": out_mse, "attractor_per_regime": out_attr}


def run_one_seed(seed, n_per_regime=12, n_epochs=40, T=128, K_max=24):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_ds = make_dataset(n_per_regime=n_per_regime, T=T, base_seed=seed * 100)
    for s in train_ds: pad_to_d_in(s, 3)
    eval_ds = make_dataset(n_per_regime=8, T=T, base_seed=seed * 100 + 5000)
    for s in eval_ds: pad_to_d_in(s, 3)

    results = {"seed": seed}
    models = []

    t0 = time.time()
    rwm_orig = train_rwm(seed, train_ds, n_epochs, stable=False, w_multistep=0.0)
    print(f"  RWM-orig    ({time.time()-t0:.1f}s)", flush=True)
    models.append(("rwm_orig", rwm_orig, "rwm"))

    t0 = time.time()
    rwm_stab_ms = train_rwm(seed, train_ds, n_epochs, stable=True, w_multistep=2.0)
    print(f"  RWM-stab+ms ({time.time()-t0:.1f}s)", flush=True)
    models.append(("rwm_stab_ms", rwm_stab_ms, "rwm"))

    t0 = time.time()
    flat = train_flat(seed, train_ds, n_epochs, w_multistep=0.0)
    print(f"  Flat-56     ({time.time()-t0:.1f}s)", flush=True)
    models.append(("flat56", flat, "flat"))

    t0 = time.time()
    flat_ms = train_flat(seed, train_ds, n_epochs, w_multistep=2.0)
    print(f"  Flat-56+ms  ({time.time()-t0:.1f}s)", flush=True)
    models.append(("flat56_ms", flat_ms, "flat"))

    for name, m, kind in models:
        results[name] = eval_model_per_regime(m, eval_ds, K_max=K_max, kind=kind)

    return results


def aggregate_runs(all_runs, model_key, K_max):
    """Aggregate per-regime MSE + attractor across seeds."""
    regimes = ["smooth", "periodic", "chaotic"]
    mse_agg = {r: {"mean": np.zeros(K_max), "std": np.zeros(K_max)} for r in regimes}
    attr_agg = {r: defaultdict(list) for r in regimes}
    for r in regimes:
        mse_stack = []
        for run in all_runs:
            if model_key not in run: continue
            res = run[model_key]
            if r in res["mse_per_regime"]:
                mse_stack.append(res["mse_per_regime"][r])
            if r in res["attractor_per_regime"]:
                for k, v in res["attractor_per_regime"][r].items():
                    attr_agg[r][k].append(v)
        if mse_stack:
            arr = np.array(mse_stack)
            mse_agg[r]["mean"] = arr.mean(axis=0)
            mse_agg[r]["std"] = arr.std(axis=0)
    attr_out = {r: {k: (float(np.mean(v)), float(np.std(v)))
                    for k, v in d.items()} for r, d in attr_agg.items()}
    return mse_agg, attr_out


def main():
    print("=" * 90, flush=True)
    print("REVIEWER-GRADE rollout : 8 seeds × 4 models × per-regime × attractor", flush=True)
    print("=" * 90, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "8"))
    K_MAX = int(os.environ.get("RWM_K_MAX", "24"))
    N_EPOCHS = int(os.environ.get("RWM_N_EPOCHS", "40"))

    print(f"  Seeds: {N_SEEDS}  K_max: {K_MAX}  epochs: {N_EPOCHS}", flush=True)
    print(f"  Models: RWM-orig | RWM-stab+ms | Flat-56 | Flat-56+ms", flush=True)
    print("=" * 90, flush=True)

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, n_epochs=N_EPOCHS, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} failed : {e}", flush=True)
            import traceback; traceback.print_exc()
    elapsed = time.time() - t0
    print(f"\n[time] total {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    if not all_runs: return

    summary = {}
    for key in ["rwm_orig", "rwm_stab_ms", "flat56", "flat56_ms"]:
        mse_agg, attr_agg = aggregate_runs(all_runs, key, K_MAX)
        summary[key] = {"mse": mse_agg, "attractor": attr_agg}

    print("\n" + "=" * 90, flush=True)
    print("MSE per K per regime (mean ± std across seeds)", flush=True)
    print("=" * 90, flush=True)
    for regime in ["smooth", "periodic", "chaotic"]:
        print(f"\n--- {regime.upper()} ---", flush=True)
        print(f"  {'K':>3}{'RWM-orig':>16}{'RWM-stab+ms':>16}"
              f"{'Flat-56':>16}{'Flat-56+ms':>16}", flush=True)
        for k in [0, 3, 7, 15, K_MAX - 1]:
            row = [f"  {k+1:>3}"]
            for key in ["rwm_orig", "rwm_stab_ms", "flat56", "flat56_ms"]:
                m = summary[key]["mse"][regime]["mean"]
                s = summary[key]["mse"][regime]["std"]
                if k < len(m):
                    row.append(f"{m[k]:>8.3e}±{s[k]:.0e}".rjust(16))
                else:
                    row.append("-".rjust(16))
            print("".join(row), flush=True)

    print("\n" + "=" * 90, flush=True)
    print("Attractor preservation (mean across seeds × samples per regime)", flush=True)
    print("=" * 90, flush=True)
    for regime in ["smooth", "periodic", "chaotic"]:
        print(f"\n--- {regime.upper()} ---", flush=True)
        for key in ["rwm_orig", "rwm_stab_ms", "flat56", "flat56_ms"]:
            a = summary[key]["attractor"].get(regime, {})
            b = a.get("bounded_frac", (None, None))
            sm = a.get("spectral_match_mean", (None, None))
            ac = a.get("autocorr_match_mean", (None, None))
            dm = a.get("recurrence_DET_match_mean", (None, None))
            lm = a.get("lyap_match_mean", (None, None))
            def fmt(v):
                if v is None or v[0] is None: return "  n/a"
                if isinstance(v, tuple) and len(v) == 2:
                    return f"{v[0]:.3f}"
                return f"{v:.3f}"
            print(f"  {key:<14} bnd={fmt(b)}  spec={fmt(sm)}  "
                  f"acf={fmt(ac)}  DET={fmt(dm)}  lyap={fmt(lm)}", flush=True)

    # Verdict
    print("\n" + "=" * 90, flush=True)
    print("CAPACITY-FAIRNESS VERDICT", flush=True)
    print("=" * 90, flush=True)
    K = K_MAX - 1
    rwm_ms_K = np.mean([summary["rwm_stab_ms"]["mse"][r]["mean"][K]
                          for r in ["smooth", "periodic", "chaotic"]])
    flat_K = np.mean([summary["flat56"]["mse"][r]["mean"][K]
                       for r in ["smooth", "periodic", "chaotic"]])
    flat_ms_K = np.mean([summary["flat56_ms"]["mse"][r]["mean"][K]
                          for r in ["smooth", "periodic", "chaotic"]])
    print(f"  K={K+1} mean MSE :")
    print(f"    RWM-stab+ms : {rwm_ms_K:.4f}")
    print(f"    Flat-56     : {flat_K:.4f}")
    print(f"    Flat-56+ms  : {flat_ms_K:.4f}")
    print(f"  Ratios:")
    print(f"    RWM-stab+ms / Flat-56    : {rwm_ms_K/max(flat_K, 1e-9):.3f}")
    print(f"    RWM-stab+ms / Flat-56+ms : {rwm_ms_K/max(flat_ms_K, 1e-9):.3f}   "
          f"← FAIR comparison")
    print(f"    Flat-56+ms / Flat-56     : {flat_ms_K/max(flat_K, 1e-9):.3f}   "
          f"← effect of multi-step loss alone")

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")

    # Convert summary to JSON-serializable
    def serialize(d):
        if isinstance(d, dict):
            return {str(k): serialize(v) for k, v in d.items()}
        if isinstance(d, np.ndarray): return d.tolist()
        if isinstance(d, (np.integer, np.floating)): return float(d)
        if isinstance(d, tuple): return [serialize(x) for x in d]
        if isinstance(d, list): return [serialize(x) for x in d]
        return d

    out_json = {
        "n_seeds": N_SEEDS, "K_max": K_MAX, "n_epochs": N_EPOCHS,
        "all_runs": serialize(all_runs),
        "summary": serialize(summary),
        "capacity_verdict": {
            "rwm_stab_ms_K_last": float(rwm_ms_K),
            "flat56_K_last": float(flat_K),
            "flat56_ms_K_last": float(flat_ms_K),
            "ratio_vs_flat": float(rwm_ms_K / max(flat_K, 1e-9)),
            "ratio_vs_flat_ms": float(rwm_ms_K / max(flat_ms_K, 1e-9)),
            "multistep_effect_on_flat": float(flat_ms_K / max(flat_K, 1e-9)),
        },
    }
    json.dump(out_json, open("results/regime_world_rollout_attractor.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_rollout_attractor.json", flush=True)


if __name__ == "__main__":
    main()
