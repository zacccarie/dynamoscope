"""Model registry global : factory function pour instancier par nom."""
from __future__ import annotations
from .base import TrajectoryProducer
from .encoders import EncoderWrapper
from .world_models import DreamerWrapper, VJEPA2Wrapper, RSSMSmallWrapper
from .observable_producers import (
    DelayEmbedProducer,
    PCAObservableProducer,
    DirectObservableProducer,
)
from ..encoder import ENCODER_REGISTRY


# Build full registry : encoders + world models
MODEL_REGISTRY: dict[str, type] = {
    # Encoders (via EncoderWrapper) - all 5 from encoder.py
    **{name: lambda n=name: EncoderWrapper(n) for name in ENCODER_REGISTRY.keys()},
    # World models stubs
    "dreamer_v3_stub": DreamerWrapper,
    "vjepa2_vitl": VJEPA2Wrapper,
    "rssm_small_stub": RSSMSmallWrapper,
    # Observable-based (classical phase-space embedding)
    "delay_motion_auto_m3": lambda: DelayEmbedProducer("motion", tau=None, m=3),
    "delay_brightness_auto_m3": lambda: DelayEmbedProducer("brightness", tau=None, m=3),
    "delay_entropy_auto_m3": lambda: DelayEmbedProducer("entropy", tau=None, m=3),
    "pca_obs_m3": lambda: PCAObservableProducer(m=3),
    "direct_bme": lambda: DirectObservableProducer(["brightness", "motion", "entropy"]),
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
    from .observable_producers import list_observable_producers
    return (
        list_encoder_wrappers()
        + list_world_model_wrappers()
        + list_observable_producers()
    )
