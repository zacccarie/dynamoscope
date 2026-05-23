"""Pytest fixtures partagées : trajectoires synthétiques pour tests modules."""
import numpy as np
import pytest

from backend.reducer import lorenz_trajectory
from backend.systems import lorenz, rossler, henon


@pytest.fixture(scope="session")
def lorenz_traj():
    """Trajectoire Lorenz normalisée (N, 3) ∈ [-1, 1]³."""
    return lorenz_trajectory(n=1000)


@pytest.fixture(scope="session")
def lorenz_raw():
    """Trajectoire Lorenz brute non-normalisée (N, 3) — pour SINDy."""
    _, raw, dt = lorenz_trajectory(n=1500, return_raw=True)
    return raw, dt


@pytest.fixture(scope="session")
def fake_latents_2048():
    """Latents synthétiques (N, 2048) avec structure trajectoire pour tests pipeline."""
    rng = np.random.RandomState(42)
    n = 100
    # Trajectoire sous-jacente 3D Lorenz embedded en 2048 via projection aléatoire
    base = lorenz_trajectory(n=n)
    W = rng.randn(3, 2048).astype(np.float32)
    latents = (base @ W) + 0.05 * rng.randn(n, 2048).astype(np.float32)
    return latents
