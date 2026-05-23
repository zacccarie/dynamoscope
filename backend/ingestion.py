"""Ingestion vidéo : transforme fichier vidéo → tableau de frames numériques.

Étapes : décodage (OpenCV) → sampling temporel → conversion RGB normalisée → thumbnails JPG.

Le sampling uniforme garantit couverture équilibrée du contenu. Resize 224×224 = standard
attendu par les encoders ImageNet/CLIP/DINOv2.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator


def sample_frames(
    video_path: str | Path,
    max_frames: int = 200,
    target_size: int = 224,
) -> tuple[np.ndarray, list[int]]:
    """Échantillonne N frames uniformément le long de la vidéo.

    Pourquoi uniforme : couverture temporelle équilibrée, évite biais début/fin.
    Pourquoi resize 224 : taille standard ResNet/ViT/CLIP/DINOv2 — entraînés sur ImageNet 224×224.
    Pourquoi RGB float [0,1] : forme attendue par tous les encoders downstream.

    Args:
        video_path: chemin vers fichier vidéo (mp4, mov, etc.)
        max_frames: nombre cible de frames à extraire (typique 100-200)
        target_size: dimension carrée des frames sorties

    Returns:
        (frames_NHWC, kept_indices) : tensor (N, H, W, 3) float32 + liste indices originaux
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Empty video")

    n = min(max_frames, total)
    # linspace = échantillonnage uniforme, garantit endpoints inclus
    indices = np.linspace(0, total - 1, n).astype(int).tolist()
    indices_set = set(indices)

    frames: list[np.ndarray] = []
    kept_indices: list[int] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices_set:
            # OpenCV utilise BGR par défaut, encoders attendent RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # INTER_AREA = best pour downsampling (anti-aliasing)
            resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
            # Normalise [0,255] → [0,1] pour stabilité numérique encoders
            frames.append(resized.astype(np.float32) / 255.0)
            kept_indices.append(idx)
        idx += 1

    cap.release()
    return np.stack(frames, axis=0), kept_indices


def video_meta(video_path: str | Path) -> dict:
    """Extrait métadonnées vidéo : fps, frame_count, dimensions.

    Pourquoi : fps utilisé pour alignement audio-frame, frame_count pour stats UI.
    """
    cap = cv2.VideoCapture(str(video_path))
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def save_thumbnails(
    frames_nhwc: np.ndarray,
    out_dir: Path,
    width: int = 200,
    quality: int = 70,
) -> list[str]:
    """Sauvegarde mini-JPEG par frame pour le player vidéo synchronisé.

    Pourquoi small JPG (200px wide, qualité 70) : trade-off taille/lisibilité.
    80 frames × ~11KB = ~900KB total. Servis statiquement par FastAPI à `/frames/{key}/`.

    Args:
        frames_nhwc: tensor frames (N, H, W, 3) float [0,1]
        out_dir: dossier destination (auto-créé)
        width: largeur cible miniature
        quality: qualité JPEG 0-100

    Returns:
        Liste noms fichiers (relatifs à out_dir) pour construction URLs frontend.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, frame in enumerate(frames_nhwc):
        # NHWC float [0,1] → BGR uint8 pour cv2.imwrite
        # [..., ::-1] inverse axe canal (RGB → BGR car cv2 attend BGR)
        bgr = (frame[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
        h, w = bgr.shape[:2]
        if w > width:
            new_h = int(h * width / w)
            bgr = cv2.resize(bgr, (width, new_h), interpolation=cv2.INTER_AREA)
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(out_dir / name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        names.append(name)
    return names
