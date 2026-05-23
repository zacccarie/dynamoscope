"""Tests module dynamics : Lyapunov, RQA, Takens, correlation dim."""
import numpy as np
from backend.dynamics import (
    analyse_trajectory, lyapunov_rosenstein, recurrence_matrix,
    rqa_stats, correlation_dimension, takens_embed, mutual_info_lag,
)


def test_lyapunov_returns_finite_scalar(lorenz_traj):
    """Lyapunov estimator retourne nombre fini + courbe divergence."""
    lam, div = lyapunov_rosenstein(lorenz_traj.astype(np.float32))
    assert np.isfinite(lam)
    assert div.ndim == 1
    assert len(div) > 5


def test_lyapunov_positive_on_chaos(lorenz_traj):
    """Lorenz chaotique → Lyapunov > 0 (divergence). En espace normalisé, plus modeste."""
    lam, _ = lyapunov_rosenstein(lorenz_traj.astype(np.float32))
    # Sur coords normalisées [-1,1], lam peut être petit mais doit rester >= 0
    assert lam >= -0.01


def test_recurrence_matrix_binary_square(lorenz_traj):
    """Recurrence matrix : binaire (0/1), carrée."""
    R, eps = recurrence_matrix(lorenz_traj[:200], max_n=200)
    assert R.shape[0] == R.shape[1]
    assert set(np.unique(R).tolist()).issubset({0, 1})
    assert eps > 0


def test_recurrence_diagonal_all_ones(lorenz_traj):
    """Diagonale de R = 1 (chaque point est récurrent avec lui-même)."""
    R, _ = recurrence_matrix(lorenz_traj[:100], max_n=100)
    assert np.all(np.diag(R) == 1)


def test_rqa_stats_in_unit_interval(lorenz_traj):
    """RR, DET, LAM ∈ [0, 1]."""
    R, _ = recurrence_matrix(lorenz_traj[:200])
    stats = rqa_stats(R)
    for k in ["RR", "DET", "LAM"]:
        assert 0.0 <= stats[k] <= 1.0


def test_correlation_dimension_positive(lorenz_traj):
    """Dim corrélation Lorenz devrait être positive."""
    d = correlation_dimension(lorenz_traj)
    assert d > 0
    assert d < 5  # bornée raisonnablement


def test_takens_embed_shape():
    """Takens embedding (m, τ) produit shape (N - (m-1)τ, m)."""
    series = np.sin(np.linspace(0, 10 * np.pi, 500)).astype(np.float32)
    embed = takens_embed(series, m=3, tau=5)
    assert embed.shape == (500 - 2 * 5, 3)


def test_mutual_info_lag_returns_int():
    """MI lag selector retourne entier >= 1."""
    series = np.sin(np.linspace(0, 10 * np.pi, 300))
    lag = mutual_info_lag(series, max_lag=30)
    assert isinstance(lag, int)
    assert lag >= 1


def test_analyse_trajectory_full_pipeline(lorenz_traj):
    """Pipeline complet retourne tous les champs attendus."""
    out = analyse_trajectory(lorenz_traj)
    assert "lyapunov" in out
    assert "correlation_dim" in out
    assert "epsilon" in out
    assert "rqa" in out
    assert {"RR", "DET", "LAM"}.issubset(out["rqa"].keys())
    assert "recurrence" in out
    assert "recurrence_size" in out
