"""Phase 2 : video data pipeline pour RegimeWorldModel.

Génère corpus procédural multi-régime, encode via DINOv2 frozen,
cache features numpy pour itération rapide.

Régimes mapping (procedurals → regime labels) :
  smooth     : color_morph  (transitions lentes couleur)
  periodic   : rotating_shapes (rotations cycliques)
  chaotic    : bouncing_balls, double_pendulum, reaction_diffusion, game_of_life

Chaque video → DINOv2 features (T, 384) → cached.
"""
from __future__ import annotations
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from ..datasets import (
    gen_bouncing_balls, gen_rotating_shapes, gen_color_morph,
    gen_double_pendulum_render, gen_reaction_diffusion, gen_game_of_life,
)
from .synth import REGIME_TO_IDX


CACHE_DIR = Path("cache/regime_world_video")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoSample:
    """Video latent trajectory + regime label."""
    features: np.ndarray  # (T, D_enc) DINOv2 features
    regime: str
    generator_name: str
    params: dict

    @property
    def traj(self) -> np.ndarray:
        """Alias for trainer compatibility (expects .traj attribute)."""
        return self.features

    @traj.setter
    def traj(self, value: np.ndarray) -> None:
        self.features = value


# Generators per regime — each callable(n_frames, seed) → list of frames
def _gen_bouncing(n_frames, seed):
    return gen_bouncing_balls(n_frames=n_frames, n_balls=4, seed=seed)

def _gen_rotating(n_frames, seed):
    return gen_rotating_shapes(n_frames=n_frames)

def _gen_color(n_frames, seed):
    return gen_color_morph(n_frames=n_frames)

def _gen_pendulum(n_frames, seed):
    return gen_double_pendulum_render(n_frames=n_frames)

def _gen_reaction(n_frames, seed):
    return gen_reaction_diffusion(n_frames=n_frames)

def _gen_gol(n_frames, seed):
    return gen_game_of_life(n_frames=n_frames, seed=seed)


GEN_TABLE = {
    "smooth": [_gen_color],
    "periodic": [_gen_rotating],
    "chaotic": [_gen_bouncing, _gen_pendulum, _gen_reaction, _gen_gol],
}


def _cache_key(gen_name: str, seed: int, n_frames: int, encoder: str) -> str:
    h = hashlib.md5(f"{gen_name}_{seed}_{n_frames}_{encoder}".encode()).hexdigest()[:16]
    return f"{gen_name}_{seed}_{n_frames}_{encoder}_{h}.npy"


def encode_video_dinov2(frames: np.ndarray, encoder, target_size: int = 224
                         ) -> np.ndarray:
    """Encode frames (N, H, W, 3) via DINOv2 → (N, 384).

    Resize to target_size×target_size (must be divisible by patch=14 → use 224).
    """
    import cv2
    # Normalize to float32 [0,1]
    if frames.dtype != np.float32:
        if frames.max() > 1.5:
            frames = frames.astype(np.float32) / 255.0
        else:
            frames = frames.astype(np.float32)
    # Resize if needed
    if frames.shape[1] != target_size or frames.shape[2] != target_size:
        resized = np.stack([
            cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_AREA)
            for f in frames
        ])
    else:
        resized = frames
    return encoder.encode(resized)


def gen_and_encode(gen_fn, gen_name: str, seed: int, n_frames: int,
                    encoder, encoder_name: str = "dinov2_vits14") -> np.ndarray:
    """Génère video frames + encode + cache."""
    cache_path = CACHE_DIR / _cache_key(gen_name, seed, n_frames, encoder_name)
    if cache_path.exists():
        return np.load(cache_path)
    raw_frames = gen_fn(n_frames, seed)
    arr = np.stack(raw_frames)
    feats = encode_video_dinov2(arr, encoder)
    np.save(cache_path, feats)
    return feats


def build_video_dataset(n_videos_per_regime: int = 4, n_frames_per_video: int = 64,
                       encoder_name: str = "dinov2_vits14",
                       seed_base: int = 0, verbose: bool = True) -> list[VideoSample]:
    """Build full video dataset : N videos × 3 regimes."""
    from ..encoder import get_encoder
    if verbose:
        print(f"  loading encoder {encoder_name}...")
    encoder = get_encoder(encoder_name)
    samples = []
    seed = seed_base
    for regime, gens in GEN_TABLE.items():
        for v_idx in range(n_videos_per_regime):
            gen_fn = gens[v_idx % len(gens)]
            gen_name = gen_fn.__name__.replace("_gen_", "")
            if verbose:
                print(f"    [{regime:<8}] {gen_name} seed={seed}")
            feats = gen_and_encode(gen_fn, gen_name, seed, n_frames_per_video,
                                    encoder, encoder_name)
            samples.append(VideoSample(
                features=feats, regime=regime,
                generator_name=gen_name, params={"seed": seed},
            ))
            seed += 1
    return samples


def windows_from_sample(sample: VideoSample, window: int = 32, stride: int = 8
                         ) -> list[VideoSample]:
    """Sub-sample window slices from one full sample."""
    out = []
    T = sample.features.shape[0]
    for start in range(0, T - window + 1, stride):
        feats_w = sample.features[start:start + window]
        out.append(VideoSample(
            features=feats_w, regime=sample.regime,
            generator_name=sample.generator_name,
            params={**sample.params, "window_start": start},
        ))
    return out


def build_windowed_dataset(n_videos_per_regime: int = 4,
                           n_frames_per_video: int = 64,
                           window: int = 32, stride: int = 8,
                           encoder_name: str = "dinov2_vits14",
                           seed_base: int = 0, verbose: bool = True) -> list[VideoSample]:
    base = build_video_dataset(n_videos_per_regime, n_frames_per_video,
                                encoder_name, seed_base, verbose)
    out = []
    for s in base:
        out.extend(windows_from_sample(s, window=window, stride=stride))
    if verbose:
        print(f"  {len(base)} base videos → {len(out)} windows of length {window}")
    return out
