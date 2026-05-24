"""RegimeWorldModel — MVP Phase 1.

Regime-aware dynamical world model.
State : (z_fast, z_slow, d_dyn, r) over {smooth, periodic, chaotic}.
"""
from .model import (
    RegimeWorldModel, PerceptualEncoder, FastSlowSSM, DescriptorHeads,
    RegimeRouter, RegimeMixtureDynamics, TransitionModule,
    KoopmanTransitionModule, REGIMES,
)
from .losses import (
    regime_conditional_loss, UncertaintyWeighting,
    loss_contrastive_descriptors,
)
from .trainer import TrainConfig, train, encode_eval
from .synth import (
    make_dataset, split_train_eval, TrajSample, REGIME_TO_IDX,
    GENERATORS_PER_REGIME,
    MixedTrajSample, gen_mixed_regime, make_mixed_dataset,
    make_harder_dataset,
)

__all__ = [
    "RegimeWorldModel", "PerceptualEncoder", "FastSlowSSM", "DescriptorHeads",
    "RegimeRouter", "RegimeMixtureDynamics", "TransitionModule",
    "KoopmanTransitionModule", "REGIMES",
    "regime_conditional_loss", "UncertaintyWeighting",
    "loss_contrastive_descriptors",
    "TrainConfig", "train", "encode_eval",
    "make_dataset", "split_train_eval", "TrajSample", "REGIME_TO_IDX",
    "GENERATORS_PER_REGIME",
    "MixedTrajSample", "gen_mixed_regime", "make_mixed_dataset",
    "make_harder_dataset",
]
