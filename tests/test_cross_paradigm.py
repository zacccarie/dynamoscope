"""Tests pour backend.cross_paradigm."""
from __future__ import annotations
import numpy as np
import pytest

from backend.cross_paradigm import observable_probe, observable_probe_transfer
from backend.observables import OBSERVABLE_KEYS


def _synthetic_obs(T: int = 80, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(T).astype(np.float32) for k in OBSERVABLE_KEYS}


def test_probe_linear_combo_high_r2():
    """Si z = linear combo d'observables, R² ≈ 1."""
    obs = _synthetic_obs(T=200)
    X = np.stack([obs[k] for k in OBSERVABLE_KEYS], axis=1)
    # Construct z as exact linear combo
    rng = np.random.default_rng(42)
    W = rng.standard_normal((12, 4))
    z = X @ W
    out = observable_probe(z, obs)
    assert out["mean_r2"] > 0.95, f"expected high R², got {out['mean_r2']}"
    assert out["n_dims_r2_above_0.8"] == 4


def test_probe_random_z_low_r2():
    """z random non-corrélé aux observables → R² faible."""
    obs = _synthetic_obs(T=200, seed=0)
    rng = np.random.default_rng(99)
    z = rng.standard_normal((200, 4))
    out = observable_probe(z, obs)
    assert out["mean_r2"] < 0.3


def test_probe_top_obs_makes_sense():
    """Top observable doit être celui le plus pondéré."""
    obs = _synthetic_obs(T=200)
    X = np.stack([obs[k] for k in OBSERVABLE_KEYS], axis=1)
    # z[0] = motion + small noise
    motion_idx = OBSERVABLE_KEYS.index("motion")
    rng = np.random.default_rng(0)
    z0 = X[:, motion_idx] + 0.01 * rng.standard_normal(200)
    z = z0[:, None]
    out = observable_probe(z, obs)
    top_obs = out["per_dim_top_obs"][0][0][0]
    assert top_obs == "motion", f"expected 'motion' top, got {top_obs}"


def test_probe_shape():
    obs = _synthetic_obs(T=100)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((100, 8))
    out = observable_probe(z, obs)
    assert len(out["per_dim_r2"]) == 8
    assert len(out["per_dim_top_obs"]) == 8
    assert 0 <= out["n_dims_r2_above_0.5"] <= 8


def test_probe_transfer():
    """Train sur set 1, eval sur set 2."""
    obs1 = _synthetic_obs(T=120, seed=1)
    obs2 = _synthetic_obs(T=120, seed=2)
    X1 = np.stack([obs1[k] for k in OBSERVABLE_KEYS], axis=1)
    X2 = np.stack([obs2[k] for k in OBSERVABLE_KEYS], axis=1)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((12, 3))
    z1 = X1 @ W
    z2 = X2 @ W
    out = observable_probe_transfer(z1, obs1, z2, obs2)
    # Same generative process → high train AND test
    assert out["train_r2_mean"] > 0.95
    assert out["test_r2_mean"] > 0.90
    assert out["transfer_ratio"] > 0.9
