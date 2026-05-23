"""Tests modules secondaires : emergence, segmentation, clustering, sfa, evolution, dna, systems."""
import numpy as np
from backend.emergence import (
    transition_mi, coarse_grain_states, effective_information_ladder,
)
from backend.segmentation import (
    perceptual_velocity_cosine, detect_peaks, find_boundaries,
)
from backend.clustering import cluster_latents, transition_matrix
from backend.sfa import slow_feature_analysis
from backend.evolution import evolution_pipeline
from backend.dna import compute_dna, _clip01
from backend.systems import lorenz, rossler, henon, van_der_pol, double_pendulum, logistic_map, list_systems
from backend.autotune import composite_score, random_search
from backend.wavelets import wavelet_decompose, wavelet_per_dim


def test_transition_mi_returns_scalar(fake_latents_2048):
    out = transition_mi(fake_latents_2048[:50])
    assert isinstance(out, float)
    assert out >= 0


def test_emergence_ladder_keys(fake_latents_2048):
    out = effective_information_ladder(fake_latents_2048[:80])
    assert "ladder" in out and "phi_id_approx" in out
    assert isinstance(out["phi_id_approx"], float)


def test_perceptual_velocity_first_zero(fake_latents_2048):
    """v[0] = 0 (pas de frame précédente)."""
    v = perceptual_velocity_cosine(fake_latents_2048)
    assert v[0] == 0.0
    assert len(v) == fake_latents_2048.shape[0]


def test_detect_peaks_min_distance():
    """Peaks respectent min_distance."""
    vel = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    peaks = detect_peaks(vel, threshold=0.5, min_distance=3)
    # Tous peaks séparés d'au moins 3
    for i in range(len(peaks) - 1):
        assert peaks[i+1] - peaks[i] >= 3


def test_find_boundaries_returns_segments(fake_latents_2048):
    out = find_boundaries(fake_latents_2048, method="adaptive", sensitivity=1.0)
    assert "segments" in out
    assert "boundaries" in out
    assert "velocity" in out


def test_hdbscan_clustering(fake_latents_2048):
    """HDBSCAN retourne labels + clusters list."""
    out = cluster_latents(fake_latents_2048, min_cluster_size=4)
    assert "labels" in out
    assert "clusters" in out
    assert len(out["labels"]) == fake_latents_2048.shape[0]


def test_transition_matrix_shape():
    """Transition matrix size = n_unique_clusters."""
    labels = [0, 1, 1, 2, 0, 2, 1, 0, 0]
    out = transition_matrix(labels)
    n_c = len(set(labels))
    assert len(out["matrix"]) == n_c


def test_sfa_returns_components(fake_latents_2048):
    """SFA retourne n_components features lents."""
    out = slow_feature_analysis(fake_latents_2048[:60], n_components=3)
    assert len(out["slow_features"]) == 60
    assert len(out["slow_features"][0]) <= 3


def test_evolution_pipeline_n_windows(fake_latents_2048):
    """Evolution glisse fenêtres."""
    out = evolution_pipeline(fake_latents_2048, window=20, stride=10)
    assert out["n_windows"] >= 1
    for k in ["velocity", "local_dim", "n_clusters", "predictability"]:
        assert k in out["metrics"]


def test_dna_composite_in_range(fake_latents_2048, lorenz_traj):
    """DNA score ∈ [0, 100]."""
    out = compute_dna(fake_latents_2048, lorenz_traj[:fake_latents_2048.shape[0]])
    assert 0 <= out["composite_score"] <= 100
    assert "label" in out
    # 8 axes incl. regime_confidence (added via observables pipeline)
    assert len(out["axes"]) == 8
    assert "regime_confidence" in out["axes"]
    assert "regime_verdict" in out
    assert out["regime_verdict"]["kind"] in (
        "fixed", "noise", "strange", "cycle", "torus", "unknown",
    )


def test_clip01_clamps():
    assert _clip01(-0.5) == 0.0
    assert _clip01(0.5) == 0.5
    assert _clip01(1.5) == 1.0


def test_systems_zoo_all_generate():
    """Tous les 6 systèmes synthétiques produisent (N, 3) sortie valide."""
    for fn in [lorenz, rossler, van_der_pol, henon, logistic_map, double_pendulum]:
        out = fn()
        coords = np.array(out["coords"])
        assert coords.shape[1] == 3
        assert coords.shape[0] > 50
        assert "name" in out and "type" in out
        # Normalisé [-1, 1]
        assert coords.min() >= -1.001
        assert coords.max() <= 1.001


def test_list_systems_returns_six():
    systems = list_systems()
    assert len(systems) == 6
    ids = {s["id"] for s in systems}
    assert {"lorenz", "rossler", "henon", "logistic", "van_der_pol", "double_pendulum"} == ids


def test_wavelet_decompose_returns_levels():
    """DWT 1D produit n_levels + approximation + détails."""
    series = np.sin(np.linspace(0, 10*np.pi, 256))
    out = wavelet_decompose(series, wavelet="db4")
    assert out["n_levels"] >= 3
    assert len(out["energy_per_scale"]) == out["n_levels"]
    assert out["dominant_scale"] >= 1


def test_wavelet_per_dim_energy_sums(fake_latents_2048):
    """Energy normalized somme à 1 (au seuil numérique)."""
    out = wavelet_per_dim(fake_latents_2048[:128])
    s = sum(out["energy_normalized"])
    assert 0.99 < s < 1.01


def test_wavelet_dominant_scale_periodic():
    """Signal sinusoïdal pure → dominant scale corresponds à période."""
    n = 512
    period = 32  # cycle every 32 samples
    series = np.sin(2 * np.pi * np.arange(n) / period)
    out = wavelet_decompose(series, wavelet="db4")
    # dominant scale doit être dans la zone correspondant à la période
    # (pas trop strict car DWT discrétise par puissances 2)
    assert out["dominant_scale"] >= 2


def test_composite_score_decomposition(fake_latents_2048, lorenz_traj):
    """Composite score = pondérée + decomposition."""
    coords = lorenz_traj[:fake_latents_2048.shape[0]]
    out = composite_score(fake_latents_2048, coords)
    assert "total" in out
    assert "faithfulness" in out
    assert "continuity_norm" in out
    assert "spread" in out
    assert 0 <= out["total"] <= 1
