"""Segmentation temporelle : detection de boundaries via peaks de velocite perceptuelle.
Algos : percentile-threshold, prominence-based, adaptive (median-MAD)."""
from __future__ import annotations
import numpy as np


def perceptual_velocity_cosine(latents: np.ndarray) -> np.ndarray:
    """Vitesse perceptuelle = cosine distance frame-a-frame."""
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    units = latents / norms
    cos_sim = np.sum(units[:-1] * units[1:], axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.concatenate([[0.0], 1.0 - cos_sim]).astype(np.float32)


def detect_peaks(velocity: np.ndarray, threshold: float, min_distance: int = 3) -> list[int]:
    """Trouve indices ou velocity > threshold avec contrainte de separation minimale."""
    peaks: list[int] = []
    last = -min_distance - 1
    for i, v in enumerate(velocity):
        if v >= threshold and (i - last) >= min_distance:
            peaks.append(i)
            last = i
    return peaks


def find_boundaries(
    latents: np.ndarray,
    method: str = "adaptive",
    sensitivity: float = 1.0,
    min_segment: int = 4,
) -> dict:
    """Détection de boundaries par analyse de velocity.
    Methods :
      - 'percentile' : seuil = (95 - 10*sens) percentile
      - 'adaptive'   : median + sensitivity * MAD (median absolute deviation)
      - 'std'        : mean + sensitivity * std
    """
    velocity = perceptual_velocity_cosine(latents)
    n = len(velocity)

    if method == "percentile":
        p = max(50.0, min(99.0, 100.0 - 10.0 * sensitivity))
        threshold = float(np.percentile(velocity, p))
    elif method == "std":
        threshold = float(velocity.mean() + sensitivity * velocity.std())
    else:  # adaptive median + MAD
        med = np.median(velocity)
        mad = np.median(np.abs(velocity - med))
        threshold = float(med + sensitivity * 2.5 * mad)

    boundaries = detect_peaks(velocity, threshold, min_distance=max(2, min_segment // 2))
    # Construit segments : [0, b1), [b1, b2), ..., [bK, n)
    cuts = [0] + boundaries + [n]
    segments: list[dict] = []
    for k in range(len(cuts) - 1):
        s, e = cuts[k], cuts[k + 1]
        if e - s < min_segment and k > 0 and segments:
            # Fusionne dans segment precedent si trop court
            segments[-1]["end"] = e
        else:
            seg_vel = velocity[s:e]
            segments.append({
                "start": int(s),
                "end": int(e),
                "length": int(e - s),
                "mean_velocity": float(np.mean(seg_vel)) if len(seg_vel) else 0.0,
                "peak_velocity": float(np.max(seg_vel)) if len(seg_vel) else 0.0,
            })

    return {
        "method": method,
        "sensitivity": sensitivity,
        "threshold": round(threshold, 4),
        "n_boundaries": len(boundaries),
        "n_segments": len(segments),
        "boundaries": boundaries,
        "segments": segments,
        "velocity": velocity.tolist(),
        "velocity_stats": {
            "mean": float(velocity.mean()),
            "std": float(velocity.std()),
            "median": float(np.median(velocity)),
            "max": float(velocity.max()),
        },
    }
