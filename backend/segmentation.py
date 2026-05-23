"""Détection automatique de shot/event boundaries dans vidéo.

Principe : peaks de vélocité perceptuelle (cosine distance frame-à-frame dans
espace latent original 2048d) = transitions sémantiques abruptes = cuts/dissolves/morphs.

3 méthodes seuil :
- adaptive : median + sens × MAD (robuste outliers)
- std : mean + sens × std (gaussian assumption)
- percentile : top X% (simple)

Output : segments [start, end) + boundaries indices.
"""
from __future__ import annotations
import numpy as np


def perceptual_velocity_cosine(latents: np.ndarray) -> np.ndarray:
    """Vélocité perceptuelle : v[t] = 1 - cos(z[t], z[t-1]).

    Cosine distance ∈ [0, 2] : 0 = identique, 1 = orthogonal, 2 = opposé.
    Insensible à la magnitude des features (ne dépend que de la direction).
    Mesure mieux la "nouveauté visuelle" que distance euclidienne.
    """
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    units = latents / norms
    cos_sim = np.sum(units[:-1] * units[1:], axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.concatenate([[0.0], 1.0 - cos_sim]).astype(np.float32)


def detect_peaks(velocity: np.ndarray, threshold: float, min_distance: int = 3) -> list[int]:
    """Trouve indices où velocity[i] ≥ threshold AVEC séparation min entre peaks.

    Si 2 peaks consécutifs trop proches, garde seulement premier (évite double-detection
    d'une même transition).
    """
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
    """Pipeline : latents → vélocités → seuil → boundaries → segments.

    Méthodes :
    - 'adaptive' (robuste) : threshold = median + sens × 2.5 × MAD (Median Absolute Deviation)
                              MAD = écart-type robuste, insensible outliers
    - 'std' : threshold = mean + sens × std (gaussian)
    - 'percentile' : threshold = (95 - 10·sens)e percentile

    Petits segments (< min_segment) sont fusionnés avec précédent pour éviter
    sur-segmentation.

    Returns: segments list + boundaries + velocity profile + threshold info.
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
