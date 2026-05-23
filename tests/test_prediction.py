"""Tests prediction module : MLP + rollout + counterfactual."""
import numpy as np
from backend.prediction import (
    LatentMLP, train_model, fit_predictor, rollout, counterfactual_rollouts,
    _project_via_knn,
)


def test_latent_mlp_forward_shape():
    """Residual MLP : output shape == input shape."""
    import torch
    model = LatentMLP(dim=64, hidden=128)
    x = torch.randn(10, 64)
    y = model(x)
    assert y.shape == x.shape


def test_train_model_converges(fake_latents_2048):
    """Training MLP doit converger (loss diminue)."""
    model, mean, std = train_model(fake_latents_2048, epochs=50, hidden=64, device="cpu")
    assert model is not None
    assert mean.shape == (1, 2048)
    assert std.shape == (1, 2048)


def test_fit_predictor_returns_predictability(fake_latents_2048):
    """fit_predictor retourne predictability + surprise + loss curve."""
    out = fit_predictor(fake_latents_2048, epochs=30, hidden=64, device="cpu")
    assert "predictability" in out
    assert "surprise" in out
    assert "loss_curve" in out
    assert 0 <= out["predictability"] <= 1


def test_rollout_returns_ghost_3d(fake_latents_2048):
    """Rollout produit ghost_3d longueur = horizon."""
    coords_3d = np.random.randn(fake_latents_2048.shape[0], 3).astype(np.float32)
    out = rollout(fake_latents_2048, coords_3d, start_idx=20, horizon=10,
                  epochs=20, hidden=64, device="cpu")
    assert len(out["ghost_3d"]) == 10
    assert len(out["step_errors"]) <= 10


def test_counterfactual_n_samples(fake_latents_2048):
    """Counterfactual rollouts produit N samples."""
    coords_3d = np.random.randn(fake_latents_2048.shape[0], 3).astype(np.float32)
    out = counterfactual_rollouts(
        fake_latents_2048, coords_3d, start_idx=20, horizon=8,
        n_samples=5, noise_scale=0.05, epochs=20, hidden=64, device="cpu",
    )
    assert len(out["rollouts"]) == 5
    for r in out["rollouts"]:
        assert len(r) == 8


def test_project_via_knn():
    """Projection knn weighted retourne (N, 3) coords valides."""
    rng = np.random.RandomState(0)
    all_lat = rng.randn(50, 128)
    coords_3d = rng.randn(50, 3)
    pred = rng.randn(10, 128)
    out = _project_via_knn(pred, all_lat, coords_3d, k=5)
    assert len(out) == 10
    assert all(len(c) == 3 for c in out)
