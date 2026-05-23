"""Sliding window analytics : évolution temporelle des métriques.

Au lieu d'agréger sur toute la vidéo, calcule métriques par fenêtre glissante.
Révèle non-stationnarité : où vidéo change-t-elle de régime ?

Métriques per-window :
- velocity : taux de changement
- local_dim : effective rank PCA local (complexité instantanée)
- n_clusters : modes locaux
- predictability : déterminisme local
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize


def _windowed_velocity(latents: np.ndarray) -> float:
    """Vitesse moyenne dans window : mean(1 - cos(z[t], z[t-1])).

    Cosine distance robuste à magnitude, sensible direction features.
    """
    n = latents.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    units = latents / np.maximum(norms, 1e-9)
    cos_sim = np.sum(units[:-1] * units[1:], axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.mean(1.0 - cos_sim))


def _windowed_local_dim(latents: np.ndarray) -> float:
    """Effective dimensionality via entropie des eigvalues PCA locales.

    PCA(window) → eigvalues λ_i. Compute p_i = λ_i / Σλ.
    Effective dim = exp(-Σ p_i log p_i) = entropie exponentielle.

    Si window vit dans sous-espace 1D, eff_dim ≈ 1. Si full rank, ≈ window size.
    Mesure complexité locale du signal.
    """
    n = latents.shape[0]
    if n < 4:
        return 0.0
    X = latents - latents.mean(axis=0, keepdims=True)
    cov = X @ X.T / max(n - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-9]
    if eig.size == 0:
        return 0.0
    p = eig / eig.sum()
    H = -np.sum(p * np.log(p))
    return float(math.exp(H))


def _windowed_n_clusters(latents: np.ndarray) -> int:
    """Compte modes locaux via mini-HDBSCAN (min_cluster_size=3).

    Window avec 1 cluster = phase homogène.
    Window avec multiple clusters = transition entre régimes.
    """
    n = latents.shape[0]
    if n < 6:
        return 0
    try:
        X = normalize(latents)
        clusterer = HDBSCAN(min_cluster_size=3, cluster_selection_method="eom")
        labels = clusterer.fit_predict(X)
        return len([u for u in set(labels) if u != -1])
    except Exception:
        return 0


def _windowed_predictability(latents: np.ndarray) -> float:
    """Proxy prédictibilité : ratio signal/bruit des vélocités.

    1 - std(vel)/mean(vel) : si vitesse régulière (low std) → prédictible.
    Si vitesse erratique (high std) → chaotique/imprévisible.
    """
    n = latents.shape[0]
    if n < 4:
        return 0.0
    diffs = np.linalg.norm(latents[1:] - latents[:-1], axis=1)
    m = diffs.mean()
    if m < 1e-9:
        return 1.0
    return float(max(0.0, 1.0 - diffs.std() / m))


def evolution_pipeline(
    latents: np.ndarray,
    window: int = 24,
    stride: int = 8,
) -> dict:
    """Glisse fenêtre window avec stride, calcule 4 métriques par window.

    Returns centers (frame index milieu de chaque window) + métriques aligned.
    Frontend trace 4 courbes superposées révélant évolution temporelle.
    """
    n = latents.shape[0]
    if n < window + stride:
        # Pas assez : compute on full
        return {
            "centers": [n // 2],
            "metrics": {
                "velocity": [_windowed_velocity(latents)],
                "local_dim": [_windowed_local_dim(latents)],
                "n_clusters": [_windowed_n_clusters(latents)],
                "predictability": [_windowed_predictability(latents)],
            },
            "window": window,
            "stride": stride,
            "n_windows": 1,
        }

    centers: list[int] = []
    metrics = {
        "velocity": [],
        "local_dim": [],
        "n_clusters": [],
        "predictability": [],
    }
    for start in range(0, n - window + 1, stride):
        chunk = latents[start : start + window]
        centers.append(start + window // 2)
        metrics["velocity"].append(_windowed_velocity(chunk))
        metrics["local_dim"].append(_windowed_local_dim(chunk))
        metrics["n_clusters"].append(_windowed_n_clusters(chunk))
        metrics["predictability"].append(_windowed_predictability(chunk))

    return {
        "centers": centers,
        "metrics": metrics,
        "window": window,
        "stride": stride,
        "n_windows": len(centers),
    }
