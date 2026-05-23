"""Phase D : mini world model + training loop avec Phase C losses.

Architecture pédagogique :
- mini_rssm.py : CNN encoder + GRU dynamics + tied decoder (deterministic RSSM)
- trainer.py : training loop combinant reconstruction + Dynamoscope losses

Goal : demonstrer feedback loop Dynamoscope → world model :
1. Train baseline (recon only) sur procedural videos
2. Train avec Dynamoscope losses (SFA + Causal + Lyapunov target)
3. Compare metrics avant/après → DNA score, smoothness, topology
"""
from .mini_rssm import MiniRSSM
from .trainer import RSSMTrainer, TrainConfig

__all__ = ["MiniRSSM", "RSSMTrainer", "TrainConfig"]
