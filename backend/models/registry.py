"""Model registry global : factory function pour instancier par nom."""
from __future__ import annotations
from .base import TrajectoryProducer
from .encoders import EncoderWrapper
from .world_models import DreamerWrapper, VJEPA2Wrapper, RSSMSmallWrapper
from ..encoder import ENCODER_REGISTRY


# Build full registry : encoders + world models
MODEL_REGISTRY: dict[str, type] = {
    # Encoders (via EncoderWrapper) - all 5 from encoder.py
    **{name: lambda n=name: EncoderWrapper(n) for name in ENCODER_REGISTRY.keys()},
    # World models stubs
    "dreamer_v3_stub": DreamerWrapper,
    "vjepa2_vitl": VJEPA2Wrapper,
    "rssm_small_stub": RSSMSmallWrapper,
}


def get_model(name: str) -> TrajectoryProducer:
    """Factory : instancie model par nom.

    Encoders disponibles immédiatement.
    World model stubs lèvent NotImplementedError au produce_trajectory().
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    factory = MODEL_REGISTRY[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()


def list_all_models() -> list[dict]:
    """Liste full des models disponibles + leur info."""
    from .encoders import list_encoder_wrappers
    from .world_models import list_world_model_wrappers
    return list_encoder_wrappers() + list_world_model_wrappers()
