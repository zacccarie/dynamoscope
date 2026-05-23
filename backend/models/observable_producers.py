"""Observable-based TrajectoryProducers : classical phase-space embedding.

Pipeline classique (sans encoder neuronal) :
  frames → 12 raw observables → choix mode embedding → trajectoire (N, m)

3 modes :
- DelayEmbedProducer : Takens delay sur 1 observable scalaire choisi
- PCAObservableProducer : PCA sur K observables → m dim
- DirectObservableProducer : pick 3 (ou m) observables comme axes directs

Catégorie : 'observable'. Complementaire aux encoders neuronaux ('encoder')
et world models ('world_model'). Plug into MODEL_REGISTRY normalement,
benchmarkable cross-model.
"""
from __future__ import annotations
import numpy as np
from .base import TrajectoryProducer
from ..observables import compute_observables, OBSERVABLE_KEYS
from ..embedding import (
    delay_embed,
    pca_embed,
    direct_embed,
    auto_tau,
)


class DelayEmbedProducer(TrajectoryProducer):
    """Takens delay embedding sur 1 observable scalaire.

    Args:
        observable: clé parmi OBSERVABLE_KEYS (ex: 'motion', 'brightness').
        tau: délai. Si None → auto_tau().
        m: dimension de plongement.
    """

    def __init__(
        self,
        observable: str = "motion",
        tau: int | None = None,
        m: int = 3,
    ):
        if observable not in OBSERVABLE_KEYS:
            raise ValueError(f"unknown observable {observable}, choose from {OBSERVABLE_KEYS}")
        self._observable = observable
        self._tau = tau
        self._m = m
        self._effective_tau: int | None = None

    @property
    def name(self) -> str:
        tau_str = self._tau if self._tau is not None else "auto"
        return f"delay_{self._observable}_t{tau_str}_m{self._m}"

    @property
    def latent_dim(self) -> int:
        return self._m

    @property
    def category(self) -> str:
        return "observable"

    @property
    def effective_tau(self) -> int | None:
        return self._effective_tau

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        observables = compute_observables(frames)
        series = observables[self._observable]
        tau = self._tau if self._tau is not None else auto_tau(series)
        self._effective_tau = int(tau)
        return delay_embed(series, m=self._m, tau=tau).astype(np.float32)

    def info(self) -> dict:
        return {
            **super().info(),
            "observable": self._observable,
            "tau": self._tau,
            "m": self._m,
            "mode": "delay",
        }


class PCAObservableProducer(TrajectoryProducer):
    """PCA sur K observables → m components.

    Args:
        keys: liste de clés observables à considérer (default = toutes 12).
        m: nombre de components principales.
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        m: int = 3,
    ):
        self._keys = keys if keys is not None else OBSERVABLE_KEYS
        for k in self._keys:
            if k not in OBSERVABLE_KEYS:
                raise ValueError(f"unknown observable {k}")
        self._m = m

    @property
    def name(self) -> str:
        return f"pca_obs_m{self._m}"

    @property
    def latent_dim(self) -> int:
        return self._m

    @property
    def category(self) -> str:
        return "observable"

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        observables = compute_observables(frames)
        points, _vals, _expl = pca_embed(observables, self._keys, m=self._m)
        return points.astype(np.float32)

    def info(self) -> dict:
        return {
            **super().info(),
            "keys": self._keys,
            "m": self._m,
            "mode": "pca",
        }


class DirectObservableProducer(TrajectoryProducer):
    """Pick K observables comme axes directs (no embedding).

    Args:
        keys: liste de clés (ex: ['brightness', 'motion', 'entropy']).
    """

    def __init__(self, keys: list[str] | None = None):
        if keys is None:
            keys = ["brightness", "motion", "entropy"]
        for k in keys:
            if k not in OBSERVABLE_KEYS:
                raise ValueError(f"unknown observable {k}")
        self._keys = keys

    @property
    def name(self) -> str:
        return "direct_" + "_".join(self._keys)

    @property
    def latent_dim(self) -> int:
        return len(self._keys)

    @property
    def category(self) -> str:
        return "observable"

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        observables = compute_observables(frames)
        return direct_embed(observables, self._keys).astype(np.float32)

    def info(self) -> dict:
        return {
            **super().info(),
            "keys": self._keys,
            "mode": "direct",
        }


def list_observable_producers() -> list[dict]:
    """Default presets utiles pour MODEL_REGISTRY + UI."""
    return [
        {
            "name": "delay_motion_auto_m3",
            "category": "observable",
            "latent_dim": 3,
            "mode": "delay",
            "observable": "motion",
            "tau": "auto",
            "m": 3,
            "description": "Takens delay sur 'motion' avec τ auto",
        },
        {
            "name": "delay_brightness_auto_m3",
            "category": "observable",
            "latent_dim": 3,
            "mode": "delay",
            "observable": "brightness",
            "tau": "auto",
            "m": 3,
            "description": "Takens delay sur luminance",
        },
        {
            "name": "pca_obs_m3",
            "category": "observable",
            "latent_dim": 3,
            "mode": "pca",
            "description": "PCA sur 12 observables → 3D",
        },
        {
            "name": "direct_bme",
            "category": "observable",
            "latent_dim": 3,
            "mode": "direct",
            "keys": ["brightness", "motion", "entropy"],
            "description": "Direct : luminance × mouvement × entropie",
        },
    ]
