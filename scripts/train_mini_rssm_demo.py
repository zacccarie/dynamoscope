"""Phase D demo : feedback loop Dynamoscope → mini RSSM.

Pipeline :
1. Génère procedural videos (bouncing balls + rotating shapes)
2. Train 2 modèles :
   - baseline (recon + dynamics only)
   - dynamoscope (+ slowness + causal + lyapunov target)
3. Encode held-out video avec chaque modèle
4. Compare DNA scores + dynamics metrics

Usage :
    python scripts/train_mini_rssm_demo.py [--epochs N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ajout root pour import backend
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from backend.datasets import (
    gen_bouncing_balls,
    gen_rotating_shapes,
    gen_color_morph,
)
from backend.training import MiniRSSM, RSSMTrainer, TrainConfig
from backend.training.trainer import video_to_tensor
from backend.reducer import reduce_3d
from backend.dna import compute_dna
from backend.dynamics import analyse_trajectory


def make_dataset(n_clips: int = 6) -> tuple[list[torch.Tensor], np.ndarray]:
    """Génère train clips (tensors) + held-out video (numpy)."""
    train_clips = []
    print(f"[data] generating {n_clips} training clips...")
    for i in range(n_clips // 3):
        frames = gen_bouncing_balls(n_frames=32, n_balls=3 + i, seed=i)
        train_clips.append(video_to_tensor(np.stack(frames), target_size=64))
    for i in range(n_clips // 3):
        frames = gen_rotating_shapes(n_frames=32)
        train_clips.append(video_to_tensor(np.stack(frames), target_size=64))
    for i in range(n_clips - 2 * (n_clips // 3)):
        frames = gen_color_morph(n_frames=32)
        train_clips.append(video_to_tensor(np.stack(frames), target_size=64))

    print("[data] held-out : bouncing 60 frames")
    held_out_frames = gen_bouncing_balls(n_frames=60, n_balls=5, seed=99)
    held_out = np.stack(held_out_frames).astype(np.float32) / 255.0
    # Resize to 64
    import cv2

    held_out = np.stack(
        [cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in held_out]
    )
    return train_clips, held_out


def evaluate(trainer: RSSMTrainer, held_out: np.ndarray, label: str) -> dict:
    """Encode held-out video + compute metrics."""
    z = trainer.encode_video(held_out)
    coords = reduce_3d(z, method="pca")
    dna = compute_dna(z, coords)
    dyn = analyse_trajectory(coords)

    # Smoothness : mean cosine velocity
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    units = z / np.maximum(norms, 1e-9)
    cos_sim = np.sum(units[:-1] * units[1:], axis=1).clip(-1, 1)
    smooth = float(np.mean(1.0 - cos_sim))

    res = {
        "label": label,
        "dna_score": dna["composite_score"],
        "dna_label": dna["label"],
        "lyapunov": dyn["lyapunov"],
        "corr_dim": dyn["correlation_dim"],
        "rqa_det": dyn["rqa"]["DET"],
        "smoothness": round(smooth, 6),
    }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n_clips", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    train_clips, held_out = make_dataset(args.n_clips)

    # --- Baseline : recon + dynamics only ---
    print("\n" + "=" * 60)
    print("[baseline] train recon + dynamics only")
    print("=" * 60)
    cfg_base = TrainConfig(
        n_epochs=args.epochs,
        w_recon=1.0,
        w_dynamics=1.0,
        w_slowness=0.0,
        w_causal=0.0,
        w_lyapunov=0.0,
    )
    trainer_base = RSSMTrainer(cfg_base)
    trainer_base.train(train_clips, verbose=not args.quiet)
    metrics_base = evaluate(trainer_base, held_out, "baseline")

    # --- Dynamoscope : + slowness + causal + lyapunov target 0 ---
    print("\n" + "=" * 60)
    print("[dynamoscope] train + Phase C losses (target Lyap=0)")
    print("=" * 60)
    cfg_dyn = TrainConfig(
        n_epochs=args.epochs,
        w_recon=1.0,
        w_dynamics=1.0,
        w_slowness=0.1,
        w_causal=0.05,
        w_lyapunov=0.05,
        lyapunov_target=0.0,
    )
    torch.manual_seed(0)
    trainer_dyn = RSSMTrainer(cfg_dyn)
    trainer_dyn.train(train_clips, verbose=not args.quiet)
    metrics_dyn = evaluate(trainer_dyn, held_out, "dynamoscope")

    # --- Compare ---
    print("\n" + "=" * 60)
    print("HELD-OUT EVAL")
    print("=" * 60)
    fields = ["dna_score", "dna_label", "lyapunov", "corr_dim", "rqa_det", "smoothness"]
    print(f"{'metric':<14}{'baseline':>16}{'dynamoscope':>16}{'delta':>14}")
    print("-" * 60)
    for f in fields:
        b = metrics_base[f]
        d = metrics_dyn[f]
        if isinstance(b, (int, float)) and isinstance(d, (int, float)):
            delta = d - b
            print(f"{f:<14}{b:>16.4f}{d:>16.4f}{delta:>+14.4f}")
        else:
            print(f"{f:<14}{str(b):>16}{str(d):>16}{'':>14}")

    # Save JSON
    import json

    out = Path("outputs/phase_d_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"baseline": metrics_base, "dynamoscope": metrics_dyn}, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
