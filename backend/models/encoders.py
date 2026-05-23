"""Encoder wrappers : adapte existing encoder.py classes au TrajectoryProducer protocol.

Wraps frozen feature extractors (ResNet50, ViT-B, DINOv2, CLIP, VideoMAE)
sans modifier code existant. Single-frame encoders : 1 latent per frame.
VideoMAE : 1 latent per frame via 16-frame centered window.
"""
from __future__ import annotations
import numpy as np
from .base import TrajectoryProducer
from ..encoder import ENCODER_REGISTRY, get_encoder


class EncoderWrapper(TrajectoryProducer):
    """Wraps backend.encoder classes en TrajectoryProducer."""

    def __init__(self, encoder_name: str):
        if encoder_name not in ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(ENCODER_REGISTRY)}")
        self._name = encoder_name
        self._encoder = get_encoder(encoder_name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def latent_dim(self) -> int:
        return self._encoder.latent_dim

    @property
    def category(self) -> str:
        return "encoder"

    def produce_trajectory(self, frames: np.ndarray) -> np.ndarray:
        return self._encoder.encode(frames)

    def info(self) -> dict:
        info_map = {
            "resnet50": {"params_M": 25, "paper": "He 2016", "pretrain": "ImageNet supervised"},
            "vit_b_16": {"params_M": 86, "paper": "Dosovitskiy 2020", "pretrain": "ImageNet supervised"},
            "dinov2_vits14": {"params_M": 22, "paper": "Oquab 2024", "pretrain": "self-supervised distillation"},
            "clip_vit_b32": {"params_M": 151, "paper": "Radford 2021", "pretrain": "contrastive image-text"},
            "videomae_base": {"params_M": 86, "paper": "Tong 2022", "pretrain": "self-supervised 16-frame clips"},
        }
        meta = info_map.get(self._name, {})
        return {
            "name": self.name,
            "category": self.category,
            "latent_dim": self.latent_dim,
            **meta,
        }


def list_encoder_wrappers() -> list[dict]:
    """Liste tous encoders disponibles avec metadata."""
    return [EncoderWrapper(name).info() for name in ENCODER_REGISTRY.keys()]
