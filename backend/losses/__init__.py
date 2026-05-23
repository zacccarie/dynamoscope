"""Differentiable losses pour entraînement de world models / encoders.

Objectif Dynamoscope research contribution : transformer les métriques
diagnostiques classiques (Lyapunov, persistence homology, slowness, causality)
en signaux gradient-based pour shaping de latent space.

Chaque loss est un nn.Module avec forward(z) → scalar diff. w.r.t. encoder params.

Module        | Réf classique           | Proxy différentiable
--------------|-------------------------|--------------------------------
lyapunov.py   | Rosenstein 1993         | log-growth de paires proches
topology.py   | Cohen-Steiner 2007 PD   | distance-matrix preservation +
              |                         | sliced Wasserstein optionnel
slowness.py   | Wiskott-Sejnowski 2002  | Δz² + variance penalty
causal.py     | Granger 1969            | L1 sur matrice autorégressive

Usage typique (training loop) :
    encoder = MyEncoder()
    losses = [SFASlownessRegularizer(weight=0.1),
              CausalSparsityLoss(weight=0.05)]
    z = encoder(video)  # (T, D)
    loss = task_loss(z) + sum(L(z) for L in losses)
    loss.backward()
"""
from .lyapunov import LyapunovMatchingLoss
from .topology import TopologyPreservationLoss
from .slowness import SFASlownessRegularizer
from .causal import CausalSparsityLoss

__all__ = [
    "LyapunovMatchingLoss",
    "TopologyPreservationLoss",
    "SFASlownessRegularizer",
    "CausalSparsityLoss",
]
