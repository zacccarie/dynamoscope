"""Video ingestion : frame sampling + GPU-ready tensors."""
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
    """Echantillonne frames uniformement. Retourne (frames NHWC float32 [0,1], indices)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Empty video")

    n = min(max_frames, total)
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
            # BGR -> RGB, resize, normalize
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
            frames.append(resized.astype(np.float32) / 255.0)
            kept_indices.append(idx)
        idx += 1

    cap.release()
    return np.stack(frames, axis=0), kept_indices


def video_meta(video_path: str | Path) -> dict:
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
    """Sauvegarde JPG miniature par frame, renvoie noms relatifs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, frame in enumerate(frames_nhwc):
        # NHWC float [0,1] -> BGR uint8
        bgr = (frame[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
        h, w = bgr.shape[:2]
        if w > width:
            new_h = int(h * width / w)
            bgr = cv2.resize(bgr, (width, new_h), interpolation=cv2.INTER_AREA)
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(out_dir / name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        names.append(name)
    return names
