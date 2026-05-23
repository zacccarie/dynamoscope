"""Phase D′ exp 3 : baselines comparison.

Compare 4 encoders sur 3 held-out video regimes :
- MiniRSSM (Dynamoscope-trained, full Phase C losses)
- MiniRSSM (baseline, recon+dynamics only)
- DINOv2-S frozen pretrained (Oquab 2024)
- ResNet50 frozen pretrained (He 2016)

Sur chaque (encoder, regime) on calcule DNA, lyap, RQA DET, smoothness.
N_REPEATS répétitions des regimes (différents seeds pour generators
quand applicable) → bootstrap 95% CI.

Question : trained mini-RSSM peut-il rivaliser avec gros encoders sur metrics ?
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
    gen_color_morph,
    gen_rotating_shapes,
    gen_double_pendulum_render,
    gen_bouncing_balls,
)
from backend.training import RSSMTrainer, TrainConfig
from backend.training.trainer import video_to_tensor
from backend.encoder import get_encoder
from backend.reducer import reduce_3d
from backend.dna import compute_dna
from backend.dynamics import analyse_trajectory


N_REPEATS = 5
N_EPOCHS = 12
TARGET_SIZE = 64
HELD_OUT_FRAMES = 60


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
        rqa = dyn["rqa"]["DET"]
    except Exception:
        lyap, rqa = float("nan"), float("nan")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    units = z / np.maximum(norms, 1e-9)
    cs = np.sum(units[:-1] * units[1:], axis=1).clip(-1, 1)
    smooth = float(np.mean(1.0 - cs))
    return {"dna": dna_score, "lyap": lyap, "rqa_det": rqa, "smooth": smooth}


def prep_for_rssm(frames_list: list) -> np.ndarray:
    """List → (N, 64, 64, 3) [0,1]."""
    import cv2
    arr = np.stack(frames_list).astype(np.float32) / 255.0
    return np.stack([cv2.resize(f, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for f in arr])


def prep_for_pretrained(frames_list: list, target_size: int = 224) -> np.ndarray:
    """List → (N, 224, 224, 3) [0,1] pour DINOv2/ResNet50."""
    import cv2
    arr = np.stack(frames_list).astype(np.float32) / 255.0
    return np.stack([cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_AREA) for f in arr])


def make_train_clips(seed: int) -> list[torch.Tensor]:
    clips = []
    clips.append(video_to_tensor(np.stack(gen_bouncing_balls(n_frames=24, n_balls=3, seed=seed)), TARGET_SIZE))
    clips.append(video_to_tensor(np.stack(gen_bouncing_balls(n_frames=24, n_balls=5, seed=seed + 1)), TARGET_SIZE))
    clips.append(video_to_tensor(np.stack(gen_rotating_shapes(n_frames=24)), TARGET_SIZE))
    clips.append(video_to_tensor(np.stack(gen_color_morph(n_frames=24)), TARGET_SIZE))
    return clips


def train_rssm(seed: int, use_dynamoscope: bool) -> RSSMTrainer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_dynamoscope:
        cfg = TrainConfig(
            n_epochs=N_EPOCHS,
            embed_dim=64,
            hidden_dim=128,
            w_slowness=0.5,
            w_causal=0.3,
            w_lyapunov=0.1,
            lyapunov_target=0.0,
        )
    else:
        cfg = TrainConfig(
            n_epochs=N_EPOCHS,
            embed_dim=64,
            hidden_dim=128,
            w_slowness=0.0,
            w_causal=0.0,
            w_lyapunov=0.0,
        )
    trainer = RSSMTrainer(cfg)
    trainer.train(make_train_clips(seed), verbose=False)
    return trainer


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    means = np.sort(np.array(means))
    return float(arr.mean()), float(means[int(alpha / 2 * n_boot)]), float(means[int((1 - alpha / 2) * n_boot)])


def main():
    print("=" * 75)
    print(f"PHASE D' EXP 3 : BASELINES ({N_REPEATS} repeats × 4 encoders × 3 regimes)")
    print("=" * 75)
    t_start = time.time()

    print("[setup] loading pretrained encoders (DINOv2, ResNet50)...")
    dino = get_encoder("dinov2_vits14")
    resnet = get_encoder("resnet50")
    print(f"  dinov2 dim: {dino.latent_dim}, resnet50 dim: {resnet.latent_dim}")

    # results[encoder][regime][metric] = list
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for rep in range(N_REPEATS):
        print(f"\n--- repeat {rep + 1}/{N_REPEATS} ---")
        # Generate fresh held-out for this repeat
        held_out_raw = {
            "smooth": gen_color_morph(n_frames=HELD_OUT_FRAMES),
            "periodic": gen_rotating_shapes(n_frames=HELD_OUT_FRAMES),
            "chaotic": gen_double_pendulum_render(n_frames=HELD_OUT_FRAMES),
        }
        held_out_rssm = {k: prep_for_rssm(v) for k, v in held_out_raw.items()}
        held_out_big = {k: prep_for_pretrained(v) for k, v in held_out_raw.items()}

        # Train 2 RSSMs (baseline + dynamoscope) for this seed
        print("  training mini_rssm_baseline...")
        rssm_base = train_rssm(rep, use_dynamoscope=False)
        print("  training mini_rssm_dynamoscope...")
        rssm_dyn = train_rssm(rep, use_dynamoscope=True)

        for regime in held_out_raw:
            # mini_rssm_baseline
            z = rssm_base.encode_video(held_out_rssm[regime])
            for k, v in compute_metrics(z).items():
                results["mini_rssm_baseline"][regime][k].append(v)

            # mini_rssm_dynamoscope
            z = rssm_dyn.encode_video(held_out_rssm[regime])
            for k, v in compute_metrics(z).items():
                results["mini_rssm_dynamoscope"][regime][k].append(v)

            # DINOv2 frozen
            z = dino.encode(held_out_big[regime])
            for k, v in compute_metrics(z).items():
                results["dinov2_vits14"][regime][k].append(v)

            # ResNet50 frozen
            z = resnet.encode(held_out_big[regime])
            for k, v in compute_metrics(z).items():
                results["resnet50"][regime][k].append(v)

        print(f"  rep {rep + 1} done")

    # Bootstrap CIs
    ENCODERS = ["mini_rssm_baseline", "mini_rssm_dynamoscope", "resnet50", "dinov2_vits14"]
    REGIMES = ["smooth", "periodic", "chaotic"]
    METRICS = ["dna", "lyap", "rqa_det", "smooth"]

    summary = {}
    for enc in ENCODERS:
        summary[enc] = {}
        for regime in REGIMES:
            summary[enc][regime] = {}
            for m in METRICS:
                vals = results[enc][regime][m]
                mean, lo, hi = bootstrap_ci(vals)
                summary[enc][regime][m] = {
                    "mean": round(mean, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "n": len(vals),
                }

    elapsed = time.time() - t_start
    print(f"\n[time] total {elapsed:.1f}s")

    # ASCII summary : DNA only for compactness
    print("\n" + "=" * 75)
    print("DNA mean [95% CI] PAR REGIME")
    print("=" * 75)
    print(f"{'encoder':<28}{'smooth':>15}{'periodic':>15}{'chaotic':>15}")
    for enc in ENCODERS:
        cols = []
        for regime in REGIMES:
            s = summary[enc][regime]["dna"]
            cols.append(f"{s['mean']:.1f} [{s['ci95_low']:.1f},{s['ci95_high']:.1f}]")
        print(f"{enc:<28}{cols[0]:>15}{cols[1]:>15}{cols[2]:>15}")

    print("\n" + "=" * 75)
    print("SMOOTHNESS mean × 10³ [95% CI] PAR REGIME")
    print("=" * 75)
    print(f"{'encoder':<28}{'smooth':>18}{'periodic':>18}{'chaotic':>18}")
    for enc in ENCODERS:
        cols = []
        for regime in REGIMES:
            s = summary[enc][regime]["smooth"]
            cols.append(f"{s['mean']*1000:.2f} [{s['ci95_low']*1000:.2f},{s['ci95_high']*1000:.2f}]")
        print(f"{enc:<28}{cols[0]:>18}{cols[1]:>18}{cols[2]:>18}")

    Path("results").mkdir(exist_ok=True)

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

    out = {
        "summary": summary,
        "n_repeats": N_REPEATS,
        "n_epochs_rssm": N_EPOCHS,
        "raw_results": {
            enc: {regime: dict(results[enc][regime]) for regime in REGIMES}
            for enc in ENCODERS
        },
    }
    json.dump(out, open("results/baselines.json", "w"), indent=2, default=_json_default)
    print(f"\n[saved] results/baselines.json")


if __name__ == "__main__":
    main()
