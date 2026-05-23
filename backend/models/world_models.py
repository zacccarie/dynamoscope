"""World model wrappers : Dreamer / V-JEPA / RSSM stubs.

Architecture : world models = encoder + learned dynamics. Pour benchmark
Dynamoscope, on dump latent trajectory produit par encoder partie du world model.

Current status : stubs avec interface définie. Implementations concrètes
demandent fetch de checkpoints externes (HuggingFace ou repo officiels)
qui peut prendre plusieurs GB. Provided as research extensibility points.

To enable :
1. Dreamer (Hafner) : checkpoint via dreamer-pytorch ou JAX original
2. V-JEPA (Bardes Meta) : weights via HuggingFace facebook/vjepa
3. RSSM stand-alone : custom small model trainable on procedural videos
"""
from __future__ import annotations
import numpy as np
from .base import TrajectoryProducer


class WorldModelWrapper(TrajectoryProducer):
    """Base abstrait pour world models. Subclasses concrètes ci-dessous."""

    @property
    def category(self) -> str:
        return "world_model"

    def rollout_latent(self, initial_z: np.ndarray, horizon: int) -> np.ndarray | None:
        """World models savent prédire futur dans latent space."""
        raise NotImplementedError


class DreamerWrapper(WorldModelWrapper):
    """Dreamer (Hafner et al. 2021) RSSM world model.

    NOT IMPLEMENTED YET — requires installation :
        pip install dreamer-pytorch
        + download checkpoint depuis https://github.com/danijar/dreamerv3

    Provided as extensibility template. Workflow attendu :
    1. Load encoder + dynamics from checkpoint
    2. produce_trajectory : run encoder on frames
    3. rollout_latent : iterate dynamics for imagination
    """
    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self._available = False
        # TODO : try import dreamer_pytorch, load checkpoint
        # self.model = ...

    @property
    def name(self) -> str:
        return "dreamer_v3_stub"

    @property
    def latent_dim(self) -> int:
        return 1024  # RSSM standard dim

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        if not self._available:
            raise NotImplementedError(
                "DreamerWrapper stub. Install dreamer-pytorch + provide checkpoint to enable."
            )
        # placeholder pour implementation future
        return np.zeros((frames.shape[0], self.latent_dim), dtype=np.float32)


class VJEPAWrapper(WorldModelWrapper):
    """V-JEPA (Bardes et al. 2024, Meta) joint embedding predictive arch.

    NOT IMPLEMENTED YET — requires :
        Weights via HuggingFace facebook/vjepa (~1-2GB)
        transformers latest + custom decoder

    Provided as extensibility template.
    """
    def __init__(self, model_id: str = "facebook/vjepa-vit-h"):
        self.model_id = model_id
        self._available = False

    @property
    def name(self) -> str:
        return "vjepa_stub"

    @property
    def latent_dim(self) -> int:
        return 1280  # ViT-H

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        if not self._available:
            raise NotImplementedError(
                "VJEPAWrapper stub. Download V-JEPA weights + transformers latest to enable."
            )
        return np.zeros((frames.shape[0], self.latent_dim), dtype=np.float32)


class RSSMSmallWrapper(WorldModelWrapper):
    """RSSM small : trainable in-tool sur procedural videos.

    Plus pragmatique : architecture custom légère, training sur procedural
    datasets (datasets.py), pas de download externe.

    Encoder = DinoV2-S frozen, dynamics = 2-layer GRU avec deterministic +
    stochastic state. Trainable end-to-end.

    Status : skeleton — implementation training loop = future work.
    """
    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self._available = False
        # TODO : implement RSSM small + training loop
        # self.encoder = DinoV2Encoder()
        # self.dynamics = GRUDynamics(input_dim=384, hidden=256, det=128, stoch=64)

    @property
    def name(self) -> str:
        return "rssm_small_stub"

    @property
    def latent_dim(self) -> int:
        return 192  # det 128 + stoch 64

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        if not self._available:
            raise NotImplementedError(
                "RSSMSmallWrapper stub. Training loop not implemented yet — see TODO."
            )
        return np.zeros((frames.shape[0], self.latent_dim), dtype=np.float32)


def list_world_model_wrappers() -> list[dict]:
    """Liste world model stubs avec metadata."""
    return [
        {
            "name": "dreamer_v3_stub",
            "category": "world_model",
            "latent_dim": 1024,
            "paper": "Hafner 2021",
            "status": "stub",
            "to_enable": "pip install dreamer-pytorch + checkpoint",
        },
        {
            "name": "vjepa_stub",
            "category": "world_model",
            "latent_dim": 1280,
            "paper": "Bardes 2024",
            "status": "stub",
            "to_enable": "download facebook/vjepa weights",
        },
        {
            "name": "rssm_small_stub",
            "category": "world_model",
            "latent_dim": 192,
            "paper": "Hafner 2018 (PlaNet)",
            "status": "stub",
            "to_enable": "training loop on procedural videos (in-tool)",
        },
    ]
