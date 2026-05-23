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


class VJEPA2Wrapper(WorldModelWrapper):
    """V-JEPA 2 (Bardes et al. 2024, Meta) — REAL implementation via transformers.

    ViT-L backbone, 326M params, output 1024d per clip after spatio-temporal patches.
    Auto-download depuis HuggingFace facebook/vjepa2-vitl-fpc64-256 (~1.3GB).

    Per-frame strategy : pour chaque frame i, clip 16-frame centered window
    (similar to VideoMAEWrapper) → mean pool → 1 vector 1024d per frame.
    """
    def __init__(self, model_id: str = "facebook/vjepa2-vitl-fpc64-256", device: str | None = None):
        import torch
        from transformers import AutoModel
        self.model_id = model_id
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().to(self.device)
        # ImageNet stats par défaut pour normalisation
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 1, 3, 1, 1)
        self._available = True

    @property
    def name(self) -> str:
        return "vjepa2_vitl"

    @property
    def latent_dim(self) -> int:
        return 1024

    def produce_trajectory(self, frames: np.ndarray, batch_size: int = 2, clip_size: int = 16) -> np.ndarray:
        """Per-frame V-JEPA 2 encoding via 16-frame centered window.

        Args:
            frames: (N, H, W, 3) float [0,1]
            batch_size: clips per forward (V-JEPA gros, garder petit)
            clip_size: 16 = V-JEPA standard
        """
        import torch
        import cv2

        n = frames.shape[0]
        # Resize 256×256 attendu par V-JEPA 2 (peut differer selon variant)
        target_size = 256
        if frames.shape[1] != target_size:
            frames_resized = np.stack([
                cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_AREA)
                for f in frames
            ])
        else:
            frames_resized = frames

        frames_u8 = (frames_resized * 255).clip(0, 255).astype(np.uint8)
        half = clip_size // 2
        out: list[np.ndarray] = []

        with torch.no_grad():
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch_clips = []
                for i in range(batch_start, batch_end):
                    idxs = [min(max(0, i - half + 1 + k), n - 1) for k in range(clip_size)]
                    clip = np.stack([frames_u8[j] for j in idxs])  # (16, H, W, 3)
                    batch_clips.append(clip)
                batch_arr = np.stack(batch_clips, axis=0)  # (B, 16, H, W, 3)
                t = torch.from_numpy(batch_arr).permute(0, 1, 4, 2, 3).float().to(self.device) / 255.0
                t = (t - self._mean) / self._std
                output = self.model(t)
                # last_hidden_state : (B, T_patches, 1024)
                h = output.last_hidden_state
                feats = h.mean(dim=1).float().cpu().numpy()
                out.append(feats)
        return np.concatenate(out, axis=0)


# Backward-compat alias for registry
VJEPAWrapper = VJEPA2Wrapper


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
            "name": "vjepa2_vitl",
            "category": "world_model",
            "latent_dim": 1024,
            "paper": "Bardes 2024 (Meta)",
            "status": "available",
            "to_enable": "auto-download facebook/vjepa2-vitl-fpc64-256 (~1.3GB)",
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
