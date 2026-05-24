"""RegimeWorldModel — MVP Phase 1.

Regime-aware dynamical world model.
State : (z_fast, z_slow, d_dyn, r) over {smooth, periodic, chaotic}.
"""
from .model import (
    RegimeWorldModel, PerceptualEncoder, FastSlowSSM, DescriptorHeads,
    RegimeRouter, RegimeMixtureDynamics, TransitionModule, REGIMES,
)
from .losses import regime_conditional_loss
from .trainer import TrainConfig, train, encode_eval
from .synth import (
    make_dataset, split_train_eval, TrajSample, REGIME_TO_IDX,
    GENERATORS_PER_REGIME,
)

__all__ = [
    "RegimeWorldModel", "PerceptualEncoder", "FastSlowSSM", "DescriptorHeads",
    "RegimeRouter", "RegimeMixtureDynamics", "TransitionModule", "REGIMES",
    "regime_conditional_loss",
    "TrainConfig", "train", "encode_eval",
    "make_dataset", "split_train_eval", "TrajSample", "REGIME_TO_IDX",
    "GENERATORS_PER_REGIME",
]
