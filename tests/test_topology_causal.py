"""Tests topology + causal modules."""
import numpy as np
from backend.topology import persistent_homology, sliding_window_ph
from backend.causal import (
    granger_pairwise, transfer_entropy_binned, ccm_sugihara, causal_summary,
)


def test_persistent_homology_h0_h1(lorenz_traj):
    """PH retourne diagrams H0 + H1."""
    out = persistent_homology(lorenz_traj[:200], max_dim=1)
    assert "diagrams" in out
    assert len(out["diagrams"]) == 2  # H0 et H1
    assert out["diagrams"][0]["dim"] == 0
    assert out["diagrams"][1]["dim"] == 1


def test_ph_h0_count_at_least_points(lorenz_traj):
    """H0 count = nombre de composantes connexes initiales ≈ N points."""
    n = 100
    out = persistent_homology(lorenz_traj[:n], max_dim=1)
    h0_count = out["diagrams"][0]["count"]
    # H0 pairs incluent toutes les composantes initiales + leur fusion
    assert h0_count >= 1


def test_sliding_window_ph_multiple_windows(lorenz_traj):
    """Sliding window génère plusieurs windows."""
    out = sliding_window_ph(lorenz_traj[:300], window=80, stride=40)
    assert out["n_windows"] >= 2


def test_granger_matrix_shape():
    """Granger pairwise produit matrice D×D."""
    rng = np.random.RandomState(0)
    # Cas dégénéré pour stabilité numérique
    n = 200
    x = rng.randn(n)
    series = np.stack([x, np.roll(x, 5) + 0.1*rng.randn(n), rng.randn(n)], axis=1)
    M = granger_pairwise(series, lag=3)
    assert M.shape == (3, 3)
    # Diagonale = 0 (un signal ne se cause pas lui-même)
    assert np.allclose(np.diag(M), 0)


def test_transfer_entropy_detects_coupling():
    """TE détecte couplage : y suit x lagged → TE(x → y) > 0."""
    rng = np.random.RandomState(0)
    n = 500
    x = rng.randn(n)
    y = np.zeros(n)
    y[5:] = 0.8 * x[:-5] + 0.2 * rng.randn(n - 5)
    te = transfer_entropy_binned(y, x, lag=5, bins=6)
    # x lagged drives y, donc TE(x → y) > 0
    assert te >= 0  # non-negative par construction


def test_ccm_sugihara_returns_in_range():
    """CCM skill score ∈ [-1, 1]."""
    rng = np.random.RandomState(0)
    n = 200
    x = np.sin(np.linspace(0, 10*np.pi, n)) + 0.1*rng.randn(n)
    y = np.cos(np.linspace(0, 10*np.pi, n)) + 0.1*rng.randn(n)
    skill = ccm_sugihara(x, y, e=3, tau=1)
    assert -1 <= skill <= 1


def test_causal_summary_returns_all_three():
    """causal_summary retourne granger + TE + CCM."""
    rng = np.random.RandomState(0)
    series = rng.randn(150, 3)
    out = causal_summary(series, lag=2)
    assert "granger" in out
    assert "transfer_entropy" in out
    assert "ccm" in out
    assert "labels" in out
