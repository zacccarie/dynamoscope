"""TrajectoryProducer : protocol unifié pour plug models dans pipeline.

Tout model qui prend frames vidéo → latent trajectory doit implémenter
cette interface. Permet swap encoders, world models, generative models
de manière interchangeable.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class TrajectoryProducer(ABC):
    """Abstract base class for any model producing latent trajectory from video.

    Subclasses :
    - EncoderWrapper : frozen feature extractor (per-frame)
    - WorldModelWrapper : encoder + learned dynamics
    - GenerativeWrapper : video generator (analyse en sortie)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier unique pour benchmark + UI."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionalité output latent space."""

    @property
    @abstractmethod
    def category(self) -> str:
        """'encoder' | 'world_model' | 'generative'."""

    @abstractmethod
    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        """Input frames (N, H, W, 3) float [0,1] → output (N, D) latents.

        Convention : 1 latent vector par frame (downsample/upsample
        si model nativement opère sur clips).
        """

    def info(self) -> dict:
        """Metadata sur le model (params, paper ref, etc.)."""
        return {
            "name": self.name,
            "category": self.category,
            "latent_dim": self.latent_dim,
        }

    # Optional : pour generative models
    def sample_trajectory(self, n_frames: int, seed: int = 0) -> np.ndarray | None:
        """Génère trajectory aléatoire/conditionnée. None si non-applicable."""
        return None

    # Optional : forward dynamics in latent space (world models)
    def rollout_latent(self, initial_z: np.ndarray, horizon: int) -> np.ndarray | None:
        """Iterate model dynamics depuis initial_z. None si non-applicable."""
        return None
