"""Observables raw : 12 features par-frame, calculées directement sur pixels.

Port de js/video.js (phase-space-video). Pipeline classique, sans encoder
neuronal : utile comme baseline interpretable ET comme source pour reconstructeur
Takens (delay embedding) en lieu de latent neuronal.

12 canaux :
  brightness  : moyenne luminance L = 0.2126R + 0.7152G + 0.0722B
  contrast    : écart-type luminance
  meanR, meanG, meanB : moyennes canaux RGB
  motion      : moyenne |L_t - L_{t-1}| (différence inter-frame)
  flowX, flowY: centroïde de la différence (proxy optical flow)
  entropy     : Shannon sur histogramme luminance 32 bins
  edges       : moyenne |∇L| (gradient Sobel)
  centroidX, centroidY : centroïde pondéré par luminance

Toutes valeurs normalisées dans [0, 1] approximativement, comparables.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

from .ingestion import sample_frames


OBSERVABLE_KEYS = [
    "brightness", "contrast", "meanR", "meanG", "meanB",
    "motion", "flowX", "flowY", "entropy", "edges",
    "centroidX", "centroidY",
]

OBSERVABLE_LABELS = {
    "brightness": "luminance",
    "contrast": "contraste",
    "meanR": "rouge",
    "meanG": "vert",
    "meanB": "bleu",
    "motion": "mouvement",
    "flowX": "flux X",
    "flowY": "flux Y",
    "entropy": "entropie",
    "edges": "contours",
    "centroidX": "centroïde X",
    "centroidY": "centroïde Y",
}


def _frame_features(frame: np.ndarray, prev_gray: np.ndarray | None) -> tuple[dict, np.ndarray]:
    """Calcule 12 scalaires + retourne gray pour next-frame motion.

    Args:
        frame: (H, W, 3) RGB uint8 ou float [0,1].
        prev_gray: (H, W) float ou None pour première frame.
    Returns:
        (features_dict, gray_array)
    """
    if frame.dtype != np.uint8:
        frame_u8 = (frame * 255).clip(0, 255).astype(np.uint8)
    else:
        frame_u8 = frame
    h, w = frame_u8.shape[:2]
    n = h * w

    r = frame_u8[..., 0].astype(np.float32)
    g = frame_u8[..., 1].astype(np.float32)
    b = frame_u8[..., 2].astype(np.float32)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b  # luminance [0,255]

    mean_L = float(gray.mean())
    contrast = float(gray.std())

    # entropie Shannon sur 32 bins
    bins = np.histogram(gray, bins=32, range=(0.0, 256.0))[0].astype(np.float64)
    p = bins / max(n, 1)
    nz = p > 0
    entropy = float(-(p[nz] * np.log2(p[nz])).sum())

    # gradient Sobel intérieur (évite bords)
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    inner = min(gx.shape[1], gy.shape[1]) - 0
    # éviter shape mismatch : tronquer
    H, W = gray.shape
    gx_i = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy_i = gray[2:, 1:-1] - gray[:-2, 1:-1]
    edge_sum = float(np.sqrt(gx_i**2 + gy_i**2).sum())
    edges = edge_sum / max(1, (W - 2) * (H - 2))

    # centroïde pondéré luminance
    ys, xs = np.mgrid[0:H, 0:W]
    sum_L = float(gray.sum())
    if sum_L > 1e-6:
        cx = float((xs * gray).sum()) / sum_L
        cy = float((ys * gray).sum()) / sum_L
    else:
        cx, cy = W / 2, H / 2

    # motion + flux
    if prev_gray is not None:
        diff = np.abs(gray - prev_gray)
        m_sum = float(diff.sum())
        motion = float(diff.mean())
        if m_sum > 1e-6:
            mfx = float((xs * diff).sum()) / m_sum
            mfy = float((ys * diff).sum()) / m_sum
        else:
            mfx, mfy = W / 2, H / 2
    else:
        motion = 0.0
        mfx, mfy = W / 2, H / 2

    features = {
        "brightness": mean_L / 255.0,
        "contrast": contrast / 128.0,
        "meanR": float(r.mean()) / 255.0,
        "meanG": float(g.mean()) / 255.0,
        "meanB": float(b.mean()) / 255.0,
        "motion": motion / 255.0,
        "flowX": mfx / W,
        "flowY": mfy / H,
        "entropy": entropy / 5.0,  # max entropy log2(32)=5
        "edges": min(1.0, edges / 96.0),
        "centroidX": cx / W,
        "centroidY": cy / H,
    }
    return features, gray


def compute_observables(
    frames: np.ndarray,
    downscale_width: int = 96,
) -> dict[str, np.ndarray]:
    """Calcule 12 observables sur séquence de frames.

    Args:
        frames: (N, H, W, 3) RGB float [0,1] ou uint8.
        downscale_width: largeur de travail (96 par défaut, comme js/video.js).
            Hauteur ajustée pour préserver ratio.
    Returns:
        dict {key: (N,) array} pour chaque clé OBSERVABLE_KEYS.
    """
    n = frames.shape[0]
    if n == 0:
        return {k: np.zeros(0) for k in OBSERVABLE_KEYS}

    # resize down pour stabiliser + accélérer
    orig_h, orig_w = frames.shape[1:3]
    if orig_w != downscale_width:
        target_w = downscale_width
        target_h = max(2, round(orig_h * target_w / orig_w))
        # cv2 attend uint8 ou float32
        f_resized = np.stack([
            cv2.resize(
                (f if f.dtype == np.uint8 else (f * 255).clip(0, 255).astype(np.uint8)),
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
            for f in frames
        ])
    else:
        f_resized = frames

    out = {k: np.zeros(n, dtype=np.float32) for k in OBSERVABLE_KEYS}
    prev_gray = None
    for i in range(n):
        feats, gray = _frame_features(f_resized[i], prev_gray)
        for k in OBSERVABLE_KEYS:
            out[k][i] = feats[k]
        prev_gray = gray

    # First-frame motion : copie de frame 1 (pas de "previous")
    if n > 1:
        for k in ("motion", "flowX", "flowY"):
            out[k][0] = out[k][1]
    return out


def observables_to_matrix(
    observables: dict[str, np.ndarray],
    keys: list[str] | None = None,
) -> np.ndarray:
    """Stack observables en matrice (N, K) pour PCA ou viz.

    Args:
        observables: dict from compute_observables.
        keys: subset des canaux. Default = tous.
    Returns:
        (N, K) array.
    """
    if keys is None:
        keys = OBSERVABLE_KEYS
    return np.stack([observables[k] for k in keys], axis=1)


def compute_observables_from_video(
    video_path: str | Path,
    max_frames: int = 240,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pipeline : video → frames → observables.

    Returns:
        (observables_dict, frame_indices)
    """
    frames, indices = sample_frames(video_path, max_frames=max_frames)
    obs = compute_observables(frames)
    return obs, indices
