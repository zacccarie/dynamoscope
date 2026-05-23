"""Phase D′ exp 2 : training ablation rigoureuse.

Pour valider claim "Phase C losses shape latents toward structured dynamics" :

- N_SEEDS seeds × N_CONFIGS loss weight settings × N_REGIMES held-out videos
- Bootstrap 95% CI sur metrics (DNA, RQA DET, smoothness, lyapunov estimé)
- Welch's t-test entre baseline et dynamoscope par regime

Configs ablation :
- baseline : recon + dyn only
- slow_only : + slowness 1.0
- causal_only : + causal 0.3
- full : slowness 0.5 + causal 0.3 + lyap-target 0

Régimes held-out :
- smooth : color_morph (low-frequency)
- periodic : rotating_shapes
- chaotic : double_pendulum_render

Output : results/training_ablation.json + ASCII summary avec CIs.
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

from backend.datasets import (
    gen_bouncing_balls,
    gen_rotating_shapes,
    gen_color_morph,
    gen_double_pendulum_render,
    gen_reaction_diffusion,
)
from backend.training import RSSMTrainer, TrainConfig
from backend.training.trainer import video_to_tensor
from backend.reducer import reduce_3d
from backend.dna import compute_dna
from backend.dynamics import analyse_trajectory


import os
N_SEEDS = int(os.environ.get("DYNAMOSCOPE_N_SEEDS", "5"))
N_EPOCHS = int(os.environ.get("DYNAMOSCOPE_N_EPOCHS", "12"))
TARGET_SIZE = 64
OUTPUT_TAG = os.environ.get("DYNAMOSCOPE_OUTPUT_TAG", "")


CONFIGS = {
    "baseline": dict(w_recon=1.0, w_dynamics=1.0, w_slowness=0.0, w_causal=0.0, w_lyapunov=0.0),
    "slow_only": dict(w_recon=1.0, w_dynamics=1.0, w_slowness=1.0, w_causal=0.0, w_lyapunov=0.0),
    "causal_only": dict(w_recon=1.0, w_dynamics=1.0, w_slowness=0.0, w_causal=0.3, w_lyapunov=0.0),
    "full": dict(w_recon=1.0, w_dynamics=1.0, w_slowness=0.5, w_causal=0.3, w_lyapunov=0.1),
}


def make_train_clips(seed: int) -> list[torch.Tensor]:
    """Stable training set across seeds."""
    rng = np.random.default_rng(seed)
    clips = []
    clips.append(video_to_tensor(np.stack(gen_bouncing_balls(n_frames=24, n_balls=3, seed=seed)), TARGET_SIZE))
    clips.append(video_to_tensor(np.stack(gen_bouncing_balls(n_frames=24, n_balls=5, seed=seed + 1)), TARGET_SIZE))
    clips.append(video_to_tensor(np.stack(gen_rotating_shapes(n_frames=24)), TARGET_SIZE))
    clips.append(video_to_tensor(np.stack(gen_color_morph(n_frames=24)), TARGET_SIZE))
    return clips


HELD_OUT_REGIMES = {
    "smooth": lambda: gen_color_morph(n_frames=60),
    "periodic": lambda: gen_rotating_shapes(n_frames=60),
    "chaotic": lambda: gen_double_pendulum_render(n_frames=60),
}


def prep_held_out(frames: list) -> np.ndarray:
    import cv2
    arr = np.stack(frames).astype(np.float32) / 255.0
    return np.stack([cv2.resize(f, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for f in arr])


def compute_metrics(z: np.ndarray) -> dict:
    coords = reduce_3d(z, method="pca")
    try:
        dna = compute_dna(z, coords)
        dna_score = dna["composite_score"]
    except Exception:
        dna_score = float("nan")
    try:
        dyn = analyse_trajectory(coords)
        lyap = dyn["lyapunov"]
        cdim = dyn["correlation_dim"]
        rqa = dyn["rqa"]["DET"]
    except Exception:
        lyap, cdim, rqa = float("nan"), float("nan"), float("nan")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    units = z / np.maximum(norms, 1e-9)
    cs = np.sum(units[:-1] * units[1:], axis=1).clip(-1, 1)
    smooth = float(np.mean(1.0 - cs))
    return {"dna": dna_score, "lyap": lyap, "cdim": cdim, "rqa_det": rqa, "smooth": smooth}


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, low, high) percentile bootstrap CI."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    means = np.sort(np.array(means))
    return float(arr.mean()), float(means[int(alpha / 2 * n_boot)]), float(means[int((1 - alpha / 2) * n_boot)])


def welch_t_test(a: list[float], b: list[float]) -> dict:
    """Welch's t-test (unequal variances). Return t, p (two-sided approx)."""
    a = np.array([v for v in a if not np.isnan(v)])
    b = np.array([v for v in b if not np.isnan(v)])
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "p_approx": float("nan"), "n_a": len(a), "n_b": len(b)}
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se < 1e-12:
        return {"t": float("nan"), "p_approx": 1.0, "n_a": na, "n_b": nb}
    t = (ma - mb) / se
    # df via Welch-Satterthwaite
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # Approx p via normal (df > 5 is decent)
    from math import erf, sqrt
    p_two = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return {"t": round(float(t), 4), "p_approx": round(float(p_two), 4), "n_a": int(na), "n_b": int(nb), "df": round(float(df), 1)}


def run_one(config_name: str, config_kwargs: dict, seed: int, held_out_dict: dict) -> dict:
    """Train 1 model, eval on all 3 held-out videos."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = TrainConfig(n_epochs=N_EPOCHS, embed_dim=64, hidden_dim=128, **config_kwargs)
    trainer = RSSMTrainer(cfg)
    clips = make_train_clips(seed)
    trainer.train(clips, verbose=False)

    out = {}
    for regime, frames in held_out_dict.items():
        z = trainer.encode_video(frames)
        out[regime] = compute_metrics(z)
    return out


def main():
    print("=" * 75)
    print(f"PHASE D' EXP 2 : TRAINING ABLATION ({N_SEEDS} seeds × {len(CONFIGS)} configs × {len(HELD_OUT_REGIMES)} regimes)")
    print("=" * 75)
    t_start = time.time()

    # Generate held-out videos once
    print("[setup] generating held-out videos...")
    held_out = {}
    for name, gen in HELD_OUT_REGIMES.items():
        held_out[name] = prep_held_out(gen())
    print(f"  regimes: {list(held_out.keys())}")

    # Results : results[config][regime][metric] = list across seeds
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    n_total = N_SEEDS * len(CONFIGS)
    n_done = 0
    for seed in range(N_SEEDS):
        for cname, ckwargs in CONFIGS.items():
            n_done += 1
            t0 = time.time()
            run_results = run_one(cname, ckwargs, seed, held_out)
            elapsed = time.time() - t0
            for regime, metrics in run_results.items():
                for k, v in metrics.items():
                    results[cname][regime][k].append(v)
            print(f"  [{n_done}/{n_total}] seed={seed} cfg={cname} ({elapsed:.1f}s)")

    # Build summary with bootstrap CI per (config, regime, metric)
    summary = {}
    METRICS = ["dna", "lyap", "rqa_det", "smooth"]
    for cname in CONFIGS:
        summary[cname] = {}
        for regime in HELD_OUT_REGIMES:
            summary[cname][regime] = {}
            for m in METRICS:
                vals = results[cname][regime][m]
                mean, lo, hi = bootstrap_ci(vals)
                summary[cname][regime][m] = {
                    "mean": round(mean, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "n": len(vals),
                }

    # Statistical test : full vs baseline per (regime, metric)
    tests = {}
    for regime in HELD_OUT_REGIMES:
        tests[regime] = {}
        for m in METRICS:
            tests[regime][m] = welch_t_test(
                results["full"][regime][m],
                results["baseline"][regime][m],
            )

    elapsed_total = time.time() - t_start
    print(f"\n[time] total {elapsed_total:.1f}s")

    # ASCII table
    print("\n" + "=" * 75)
    print("SUMMARY : mean [95% CI]")
    print("=" * 75)
    for regime in HELD_OUT_REGIMES:
        print(f"\n--- regime: {regime} ---")
        print(f"{'config':<14}{'DNA':>20}{'RQA_DET':>20}{'smooth':>22}")
        for cname in CONFIGS:
            s = summary[cname][regime]
            dna_str = f"{s['dna']['mean']:.2f} [{s['dna']['ci95_low']:.2f},{s['dna']['ci95_high']:.2f}]"
            rqa_str = f"{s['rqa_det']['mean']:.3f} [{s['rqa_det']['ci95_low']:.3f},{s['rqa_det']['ci95_high']:.3f}]"
            sm_str = f"{s['smooth']['mean']:.5f} [{s['smooth']['ci95_low']:.5f},{s['smooth']['ci95_high']:.5f}]"
            print(f"{cname:<14}{dna_str:>20}{rqa_str:>20}{sm_str:>22}")

    print("\n" + "=" * 75)
    print("WELCH'S T-TEST : full vs baseline")
    print("=" * 75)
    print(f"{'regime':<12}{'metric':<10}{'t':>10}{'p_approx':>12}{'significant':>15}")
    for regime in HELD_OUT_REGIMES:
        for m in METRICS:
            tt = tests[regime][m]
            sig = "yes (*)" if tt.get("p_approx", 1) < 0.05 else "no"
            print(f"{regime:<12}{m:<10}{tt['t']:>10}{tt['p_approx']:>12}{sig:>15}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "summary": summary,
        "tests_full_vs_baseline": tests,
        "config_settings": CONFIGS,
        "n_seeds": N_SEEDS,
        "n_epochs": N_EPOCHS,
        "raw_results": {
            cname: {regime: dict(results[cname][regime]) for regime in HELD_OUT_REGIMES}
            for cname in CONFIGS
        },
    }

    def _json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"non-serializable {type(o)}")

    suffix = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
    out_path = f"results/training_ablation{suffix}.json"
    json.dump(out, open(out_path, "w"), indent=2, default=_json_default)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
