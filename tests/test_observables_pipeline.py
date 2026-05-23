"""Tests pour observables + embedding + regime classifier + producers."""
from __future__ import annotations
import numpy as np
import pytest

from backend.observables import (
    compute_observables,
    observables_to_matrix,
    OBSERVABLE_KEYS,
)
from backend.embedding import auto_tau, delay_embed, pca_embed, direct_embed, zscore
from backend.regime_classifier import classify_regime, classify_from_analysis


# ---------------------- observables ----------------------


def _synthetic_frames(n: int = 30, h: int = 32, w: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = rng.uniform(0, 1, (n, h, w, 3)).astype(np.float32)
    return frames


def test_observables_keys_present():
    frames = _synthetic_frames()
    obs = compute_observables(frames)
    for k in OBSERVABLE_KEYS:
        assert k in obs, f"missing {k}"
        assert obs[k].shape == (frames.shape[0],)


def test_observables_values_finite_and_bounded():
    frames = _synthetic_frames()
    obs = compute_observables(frames)
    for k in OBSERVABLE_KEYS:
        vals = obs[k]
        assert np.isfinite(vals).all(), f"{k} has non-finite values"
        # All normalized to [0, ~1.5]
        assert vals.min() >= -0.01
        assert vals.max() <= 1.5


def test_observables_first_motion_eq_second():
    """First frame motion should be copied from frame 1 (no t-1)."""
    frames = _synthetic_frames(n=10)
    obs = compute_observables(frames)
    assert obs["motion"][0] == obs["motion"][1]
    assert obs["flowX"][0] == obs["flowX"][1]


def test_observables_to_matrix():
    frames = _synthetic_frames()
    obs = compute_observables(frames)
    M = observables_to_matrix(obs)
    assert M.shape == (frames.shape[0], len(OBSERVABLE_KEYS))


# ---------------------- embedding ----------------------


def test_zscore_zero_mean_unit_var():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(arr)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9


def test_auto_tau_returns_positive():
    rng = np.random.default_rng(0)
    series = np.sin(np.linspace(0, 6 * np.pi, 200)) + 0.1 * rng.standard_normal(200)
    tau = auto_tau(series)
    assert tau >= 1


def test_delay_embed_shape():
    series = np.sin(np.linspace(0, 6 * np.pi, 100))
    out = delay_embed(series, m=3, tau=2)
    expected_n = 100 - 2 * 2
    assert out.shape == (expected_n, 3)


def test_delay_embed_too_short_raises():
    with pytest.raises(ValueError):
        delay_embed(np.array([1.0, 2.0, 3.0]), m=3, tau=2)


def test_pca_embed_dim_capping():
    rng = np.random.default_rng(0)
    channels = {f"c{i}": rng.standard_normal(80) for i in range(5)}
    points, vals, expl = pca_embed(channels, list(channels.keys()), m=3)
    assert points.shape == (80, 3)
    assert vals.shape == (3,)
    assert expl.shape == (3,)


def test_pca_explained_sums_close_to_one():
    rng = np.random.default_rng(0)
    channels = {f"c{i}": rng.standard_normal(80) for i in range(4)}
    _pts, _vals, expl = pca_embed(channels, list(channels.keys()), m=4)
    # Power iter on noise → eigenvalues all ≈1 → fractions sum to ~1
    assert abs(expl.sum() - 1.0) < 0.1


def test_direct_embed_shape():
    rng = np.random.default_rng(0)
    channels = {"a": rng.standard_normal(50), "b": rng.standard_normal(50), "c": rng.standard_normal(50)}
    out = direct_embed(channels, ["a", "b", "c"])
    assert out.shape == (50, 3)


# ---------------------- regime classifier ----------------------


def test_regime_fixed_point():
    """High convergence → fixed point."""
    v = classify_regime(
        embedding_dim=3,
        correlation_dim=0.5,
        max_diag_ratio=0.5,
        lyapunov=-0.1,
        convergence=0.9,
    )
    assert v.kind == "fixed"
    assert v.confidence > 0.5


def test_regime_noise():
    """Low max_diag_ratio + high corr_dim → noise."""
    v = classify_regime(
        embedding_dim=3,
        correlation_dim=2.8,
        max_diag_ratio=0.01,
        lyapunov=0.05,
        convergence=0.0,
    )
    assert v.kind == "noise"


def test_regime_strange_attractor():
    """Intermediate diag + corrDim non-saturated → strange."""
    v = classify_regime(
        embedding_dim=4,
        correlation_dim=2.1,
        max_diag_ratio=0.3,
        lyapunov=0.5,
        convergence=0.0,
    )
    assert v.kind == "strange"


def test_regime_cycle_limit():
    """High diag + low corr_dim → cycle limit."""
    v = classify_regime(
        embedding_dim=3,
        correlation_dim=1.05,
        max_diag_ratio=0.85,
        lyapunov=0.0,
        convergence=0.0,
    )
    assert v.kind == "cycle"


def test_regime_torus():
    """High diag + corr_dim ~ 2 → torus."""
    v = classify_regime(
        embedding_dim=4,
        correlation_dim=2.0,
        max_diag_ratio=0.85,
        lyapunov=0.0,
        convergence=0.0,
    )
    assert v.kind == "torus"


def test_regime_from_analysis_dict():
    """classify_from_analysis avec dict synthétique."""
    analysis = {
        "lyapunov": 0.0,
        "correlation_dim": 1.0,
        "max_diag_ratio": 0.9,
        "convergence_rate": 0.0,
        "rqa": {"DET": 0.9},
    }
    v = classify_from_analysis(analysis, embedding_dim=3)
    assert v.kind == "cycle"
    assert v.confidence > 0


# ---------------------- TrajectoryProducers ----------------------


def test_delay_embed_producer_basic():
    from backend.models.observable_producers import DelayEmbedProducer
    frames = _synthetic_frames(n=40)
    p = DelayEmbedProducer("motion", tau=2, m=3)
    out = p.produce_trajectory(frames)
    assert out.shape[0] == 40 - 2 * 2
    assert out.shape[1] == 3
    assert np.isfinite(out).all()


def test_delay_embed_producer_auto_tau():
    from backend.models.observable_producers import DelayEmbedProducer
    frames = _synthetic_frames(n=60)
    p = DelayEmbedProducer("brightness", tau=None, m=3)
    out = p.produce_trajectory(frames)
    assert p.effective_tau is not None
    assert p.effective_tau >= 1


def test_pca_observable_producer():
    from backend.models.observable_producers import PCAObservableProducer
    frames = _synthetic_frames(n=40)
    p = PCAObservableProducer(m=3)
    out = p.produce_trajectory(frames)
    assert out.shape == (40, 3)


def test_direct_observable_producer():
    from backend.models.observable_producers import DirectObservableProducer
    frames = _synthetic_frames(n=40)
    p = DirectObservableProducer(["brightness", "motion", "entropy"])
    out = p.produce_trajectory(frames)
    assert out.shape == (40, 3)


def test_registry_observable_models_loadable():
    from backend.models.registry import get_model
    for name in ["delay_motion_auto_m3", "pca_obs_m3", "direct_bme"]:
        m = get_model(name)
        assert m.category == "observable"
        assert m.latent_dim == 3
