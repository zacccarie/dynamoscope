"""Penn Action real video : test Sprint winners on real action sequences.

Mini setup (Sprint test) :
  - 3 action classes mapped to dynamics regimes (rough proxy) :
      smooth   : push_ups (controlled, slow tempo)
      periodic : jumping_jacks (cyclic)
      chaotic  : tennis_serve (one-shot explosive)
  - 20 videos per class → 60 total
  - 64 frames per video → DINOv2 features (T=64, D=384)
  - Window slicing : 32 frames, stride 16 → ~120 training windows
  - Eval split : 80 / 20

Variants compared (Sprint winners) :
  A) baseline      (stab+ms)
  B) per_timestep  (Sprint 3 winner)
  C) per_ts_lyap   (Sprint 4 winner)
  D) Flat-56+ms    (capacity-fair control)

3 seeds (cheap on real video, encoding cached after first run).
Welch t-tests on rollout K=24 MSE per action.

Encoder DINOv2 frozen → d_fast = 384 (not 32 like synth).
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
from backend.regime_world.penn_action import (
    build_penn_action_dataset, windows_from_penn_sample,
)
from experiments.regime_world_probe import FlatBaseline, train_baseline
from experiments.regime_world_rollout_attractor import eval_model_per_regime
from backend.regime_world.synth import REGIME_TO_IDX


# Map Penn Action class → our 3-regime taxonomy.
# Best heuristic guess based on motion type :
ACTION_TO_REGIME = {
    "bench_press": "smooth",     # slow controlled lifting (n=115 ≥48 frames)
    "squat": "periodic",         # cyclic repetition (n=227 ≥48 frames)
    "tennis_serve": "chaotic",   # explosive one-shot (n=161 ≥48 frames)
}


def relabel_to_regime(samples):
    """Convert Penn Action samples (.action) to regime-tagged samples."""
    out = []
    for s in samples:
        regime = ACTION_TO_REGIME.get(s.action, None)
        if regime is None:
            continue
        # Mutate action → regime for trainer compat.
        s.action = regime
        out.append(s)
    return out


def build(variant, seed, d_in=384):
    torch.manual_seed(seed); np.random.seed(seed)
    base_kw = dict(d_in=d_in, d_fast=64, d_slow=64, slow_stride=4, stable=True)
    base_cfg = dict(n_epochs=20, lr=3e-4, w_dyn=1.0, w_slow=1.0,
                     w_recur=0.5, w_lyap=0.3, w_entropy=0.0,
                     w_regime_sup=5.0, w_multistep=2.0, K_multistep=4,
                     gamma_multistep=0.9, n_starts_multistep=4)
    if variant == "baseline":
        m = RegimeWorldModel(**base_kw)
        cfg = TrainConfig(**base_cfg)
    elif variant == "per_timestep":
        m = RegimeWorldModel(**base_kw, per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg)
    elif variant == "per_ts_lyap":
        m = RegimeWorldModel(**base_kw, per_timestep_router=True, router_window=4)
        cfg = TrainConfig(**base_cfg, w_lyap_reg=0.5,
                           lyap_targets=(0.9, 1.0, 1.0))
    elif variant == "flat56_ms":
        m = FlatBaseline(d_in=d_in, d_h=128)
        return m, None, "flat"
    else:
        raise ValueError(variant)
    return m, cfg, "rwm"


def train_one(variant, m, cfg, train_ds, kind):
    if kind == "rwm":
        train(m, train_ds, cfg, verbose=False)
    else:
        train_baseline(m, train_ds, n_epochs=20, w_multistep=2.0,
                        K_multistep=4, gamma_multistep=0.9,
                        n_starts_multistep=4)


def split_train_eval(samples, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(len(idx) * train_ratio)
    return [samples[i] for i in idx[:n_tr]], [samples[i] for i in idx[n_tr:]]


def run_one_seed(seed, samples, K_max=24, window=32, stride=16):
    print(f"\n=== SEED {seed} ===", flush=True)
    train_full, eval_full = split_train_eval(samples, train_ratio=0.8, seed=seed)
    # Windowing for training
    train_ds = []
    for s in train_full:
        train_ds.extend(windows_from_penn_sample(s, window=window, stride=stride))
    eval_ds = eval_full  # keep full eval videos for rollout assessment
    print(f"  train windows: {len(train_ds)}  eval videos: {len(eval_ds)}", flush=True)
    results = {"seed": seed}
    for variant in ["baseline", "per_timestep", "per_ts_lyap", "flat56_ms"]:
        t0 = time.time()
        try:
            m, cfg, kind = build(variant, seed, d_in=samples[0].features.shape[1])
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
    print("PENN ACTION mini : baseline vs per_timestep vs per_ts_lyap vs flat56_ms",
          flush=True)
    print("=" * 96, flush=True)
    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "3"))
    # Penn Action videos are short (pushup median 45, jumping_jacks median 30,
    # tennis_serve median 64). K_max=8 + T_ctx=16 → need T ≥ 48 per video.
    K_MAX = int(os.environ.get("RWM_K_MAX", "8"))
    N_PER_CLASS = int(os.environ.get("PENN_N_PER_CLASS", "20"))
    MAX_FRAMES = int(os.environ.get("PENN_MAX_FRAMES", "128"))

    print(f"  N_per_class={N_PER_CLASS}  max_frames={MAX_FRAMES}  seeds={N_SEEDS}",
          flush=True)
    print(f"  Action→regime mapping: {ACTION_TO_REGIME}", flush=True)

    # Load + cache features (done once, reused across seeds).
    t0 = time.time()
    raw = build_penn_action_dataset(
        actions=list(ACTION_TO_REGIME.keys()),
        n_per_class=N_PER_CLASS, max_frames=MAX_FRAMES,
        seed=0, verbose=True,
    )
    samples = relabel_to_regime(raw)
    print(f"  encoding+caching took {time.time()-t0:.1f}s", flush=True)
    # Filter videos too short for K_max rollout : need T ≥ T_ctx + K_max*4.
    min_T = 16 + K_MAX * 4
    before = len(samples)
    samples = [s for s in samples if s.features.shape[0] >= min_T]
    print(f"  filtered T < {min_T}: {before} → {len(samples)} samples", flush=True)
    from collections import Counter
    reg_counts = Counter(s.action for s in samples)
    print(f"  per-regime counts: {dict(reg_counts)}", flush=True)

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, samples, K_max=K_MAX))
        except Exception as e:
            print(f"  seed {seed} fatal: {e}", flush=True)
    print(f"\n[time] training+eval total {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)",
          flush=True)

    if not all_runs: return

    VARIANTS = ["baseline", "per_timestep", "per_ts_lyap", "flat56_ms"]
    REGIMES = ["smooth", "periodic", "chaotic"]
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
    print(f"K={K+1} MSE per (action→regime) on Penn Action", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'Variant':<14}{'smooth (bench_p)':>22}"
          f"{'periodic (squat)':>22}{'chaotic (tennis_s)':>22}",
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

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        raise TypeError(f"{type(o)}")
    out = {"n_seeds": N_SEEDS, "K_max": K_MAX,
            "n_per_class": N_PER_CLASS, "max_frames": MAX_FRAMES,
            "action_mapping": ACTION_TO_REGIME, "all_runs": all_runs}
    json.dump(out, open("results/regime_world_penn_action.json", "w"),
              indent=2, default=_d)
    print(f"\n[saved] results/regime_world_penn_action.json", flush=True)


if __name__ == "__main__":
    main()
