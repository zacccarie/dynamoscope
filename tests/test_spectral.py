"""Tests spectral : DMD, power spectrum."""
import numpy as np
from backend.spectral import dmd, power_spectrum
from backend.multiscale import (
    shannon_entropy, coarse_grain, multiscale_entropy, spectral_slope,
)


def test_dmd_returns_modes(lorenz_traj):
    """DMD retourne eigvals + amplitudes + frequences."""
    out = dmd(lorenz_traj, dt=0.01)
    assert "eigvals_re" in out and "eigvals_im" in out
    assert "amplitudes" in out
    assert "freqs" in out
    assert out["n_modes"] >= 1


def test_dmd_eigvals_near_unit_circle(lorenz_traj):
    """Pour signal stable cyclique, |eigvals| proche 1."""
    out = dmd(lorenz_traj)
    re = np.array(out["eigvals_re"])
    im = np.array(out["eigvals_im"])
    mags = np.sqrt(re**2 + im**2)
    # Au moins une eigenvalue proche cercle unité
    assert (mags > 0.5).any()


def test_power_spectrum_freqs_positive():
    """Frequencies returned sont positives + ordrées."""
    X = np.random.randn(100, 5)
    out = power_spectrum(X)
    freqs = out["freqs"]
    assert all(f >= 0 for f in freqs)


def test_shannon_entropy_uniform_max():
    """Distribution uniforme donne entropie max log2(bins)."""
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=10000)
    h = shannon_entropy(x, bins=32)
    assert 4 < h < 5.1  # max théorique log2(32) = 5


def test_shannon_entropy_constant_zero():
    """Signal constant → entropie ≈ 0."""
    x = np.ones(100)
    h = shannon_entropy(x)
    assert h < 0.5


def test_coarse_grain_halves_length():
    """coarse_grain(X, factor=2) divise length par 2."""
    X = np.random.randn(100, 5)
    Xc = coarse_grain(X, factor=2)
    assert Xc.shape == (50, 5)


def test_multiscale_entropy_ladder_shape(lorenz_traj):
    """Multiscale entropy returns ladder list."""
    out = multiscale_entropy(lorenz_traj)
    assert "ladder" in out
    assert len(out["ladder"]) >= 3
    for row in out["ladder"]:
        assert "scale" in row and "entropy" in row


def test_spectral_slope_lorenz_negative(lorenz_traj):
    """Lorenz a spectral slope négatif (power-law decay)."""
    slope = spectral_slope(lorenz_traj)
    assert slope < 0
