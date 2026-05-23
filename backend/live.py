"""Live webcam sessions : streaming temps réel pour growing trajectory.

Architecture session-based : chaque connexion webcam = 1 session avec UUID.
Buffer FIFO 300 frames max. Encode chaque frame entrant, append latent.
Re-project trajectory entière à chaque update (UMAP rapide sur petits N).
"""
from __future__ import annotations
import time
import uuid
from pathlib import Path
from typing import Any
import numpy as np


class LiveSession:
    """État d'une session webcam en cours.

    Attributs :
    - id : UUID 12 chars
    - encoder_name : encoder utilisé pour cette session (verrouillé)
    - latents : list growing de feature vectors
    - thumb_paths : noms JPGs sauvés disque
    - frames_dir : dossier dédié `cache/frames/live_{id}/`
    - max_frames : FIFO buffer size (drop oldest si dépassé)
    """

    def __init__(self, encoder_name: str, frames_dir: Path, max_frames: int = 300):
        self.id = uuid.uuid4().hex[:12]
        self.encoder_name = encoder_name
        self.created_at = time.time()
        self.max_frames = max_frames
        self.latents: list[np.ndarray] = []
        self.thumb_paths: list[str] = []
        self.frames_dir = frames_dir / f"live_{self.id}"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.last_refit_n = 0

    @property
    def n(self) -> int:
        return len(self.latents)

    def append(self, latent: np.ndarray, thumb_name: str) -> None:
        if self.n >= self.max_frames:
            # FIFO : drop oldest
            self.latents.pop(0)
            self.thumb_paths.pop(0)
        self.latents.append(latent.flatten())
        self.thumb_paths.append(thumb_name)

    def stacked_latents(self) -> np.ndarray:
        return np.stack(self.latents, axis=0)


_SESSIONS: dict[str, LiveSession] = {}


def create_session(encoder_name: str, frames_dir: Path) -> LiveSession:
    """Crée nouvelle session live + enregistre dans registry global _SESSIONS."""
    s = LiveSession(encoder_name=encoder_name, frames_dir=frames_dir)
    _SESSIONS[s.id] = s
    return s


def get_session(sid: str) -> LiveSession | None:
    """Retrieve session par UUID. Returns None si session expirée ou inexistante."""
    return _SESSIONS.get(sid)


def end_session(sid: str) -> None:
    """Cleanup session : retire du registry. Frames sur disque préservés pour history."""
    _SESSIONS.pop(sid, None)
