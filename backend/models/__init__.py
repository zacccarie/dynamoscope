"""Model plugin protocol : unified interface for trajectory producers.

3 catégories :
- encoders : frozen feature extractors (ResNet, ViT, DINOv2, CLIP, VideoMAE)
- world_models : encoder + dynamics (Dreamer, V-JEPA, RSSM)
- generative : video generators (SORA, AnimateDiff, CogVideoX)

Tous implémentent TrajectoryProducer protocol.
"""
from .base import TrajectoryProducer
from .encoders import (
    EncoderWrapper, list_encoder_wrappers,
)
from .world_models import WorldModelWrapper, list_world_model_wrappers
from .registry import MODEL_REGISTRY, get_model

__all__ = [
    "TrajectoryProducer",
    "EncoderWrapper",
    "WorldModelWrapper",
    "list_encoder_wrappers",
    "list_world_model_wrappers",
    "MODEL_REGISTRY",
    "get_model",
]
