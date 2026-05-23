"""Tests pour backend.training : MiniRSSM forward + Trainer smoke + eval."""
from __future__ import annotations
import numpy as np
import torch
import pytest

from backend.training import MiniRSSM, RSSMTrainer, TrainConfig


def test_mini_rssm_forward():
    m = MiniRSSM(in_channels=3, embed_dim=64, hidden_dim=128)
    frames = torch.rand(6, 3, 64, 64)
    z, z_pred, recon = m(frames)
    assert z.shape == (6, 64)
    assert z_pred.shape == (6, 64)
    assert recon.shape == (6, 3, 64, 64)
    assert torch.isfinite(z).all()
    assert torch.isfinite(z_pred).all()
    assert torch.isfinite(recon).all()


def test_mini_rssm_params_count():
    m = MiniRSSM(embed_dim=64, hidden_dim=128)
    n = m.count_params()
    # Should be a few hundred thousand to single millions
    assert 100_000 < n < 10_000_000, f"unexpected param count {n}"


def test_trainer_runs_one_epoch():
    cfg = TrainConfig(
        n_epochs=1,
        embed_dim=32,
        hidden_dim=64,
        w_slowness=0.1,
        w_causal=0.05,
        device="cpu",
    )
    t = RSSMTrainer(cfg)
    clip = torch.rand(8, 3, 64, 64)
    hist = t.train([clip], verbose=False)
    assert len(hist) == 1
    assert torch.isfinite(torch.tensor(hist[0].loss_total))
    assert hist[0].loss_total > 0


def test_trainer_loss_decreases():
    """Avec recon only + assez d'epochs, recon loss doit descendre."""
    cfg = TrainConfig(
        n_epochs=8,
        embed_dim=32,
        hidden_dim=64,
        w_slowness=0.0,
        w_causal=0.0,
        device="cpu",
    )
    t = RSSMTrainer(cfg)
    # Repeat same frames so model can memorize
    clip = torch.rand(8, 3, 64, 64)
    hist = t.train([clip], verbose=False)
    assert hist[-1].loss_recon < hist[0].loss_recon, (
        f"recon loss should decrease: start={hist[0].loss_recon}, "
        f"end={hist[-1].loss_recon}"
    )


def test_trainer_encode_video():
    cfg = TrainConfig(
        n_epochs=1, embed_dim=32, hidden_dim=64, device="cpu"
    )
    t = RSSMTrainer(cfg)
    frames_np = np.random.rand(10, 64, 64, 3).astype(np.float32)
    z = t.encode_video(frames_np)
    assert z.shape == (10, 32)
    assert np.isfinite(z).all()


def test_trainer_all_losses_active():
    """Vérifier que toutes les Phase C losses peuvent être combinées."""
    cfg = TrainConfig(
        n_epochs=1,
        embed_dim=32,
        hidden_dim=64,
        w_recon=1.0,
        w_dynamics=1.0,
        w_slowness=0.1,
        w_causal=0.05,
        w_lyapunov=0.05,
        w_topology=0.05,
        device="cpu",
    )
    t = RSSMTrainer(cfg)
    clip = torch.rand(16, 3, 64, 64)
    hist = t.train([clip], verbose=False)
    # All loss components should be finite and non-zero
    s = hist[0]
    assert torch.isfinite(torch.tensor(s.loss_recon))
    assert torch.isfinite(torch.tensor(s.loss_dynamics))
    assert torch.isfinite(torch.tensor(s.loss_slowness))
    assert s.loss_slowness > 0
    assert torch.isfinite(torch.tensor(s.loss_topology))
