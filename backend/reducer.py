"""Réduction dimensionnelle : latents haute-dim (768-2048d) → 3D pour visualisation.

3 algorithmes au choix :
- UMAP : non-linéaire topologique, préserve voisinages, distord distances globales
- PCA : linéaire orthogonale, préserve variance + distances euclidiennes
- Isomap : géodésique k-NN, préserve distances sur manifold

Plus : trajectoire Lorenz synthétique pour démos.
"""
from __future__ import annotations
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


def reduce_3d(
    latents: np.ndarray,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    smoothing: int = 0,
) -> np.ndarray:
    """Projette latents (N, D) → coords 3D (N, 3) ∈ [-1, 1]³.

    Choix méthode :
    - umap : non-linéaire, clusters séparés visuellement, mais distances déformées
    - pca : linéaire, isométrique, frames proches en D restent proches en 3D
    - isomap : compromis géodésique, bon sur manifolds courbés

    Args:
        latents: (N, D) features
        method: 'umap' | 'pca' | 'isomap'
        n_neighbors: échelle structure locale (UMAP/Isomap). Petit = détail, grand = global.
        min_dist: compacité cluster UMAP. 0 = clusters serrés, 1 = uniforme.
        metric: 'cosine' | 'euclidean' | 'manhattan' | 'correlation' (UMAP).
        smoothing: moyenne glissante temporelle post-projection (lisse jumps UMAP).

    Returns:
        (N, 3) float32 normalisé [-1, 1] par axe (esthétique Three.js cube).
    """
    n = latents.shape[0]
    d_target = min(3, n, latents.shape[1])

    if method == "umap" and n >= 5:
        # n_neighbors auto-clip si n trop petit
        nn = min(max(2, n_neighbors), max(2, n - 1))
        reducer = umap.UMAP(
            n_components=3, n_neighbors=nn, min_dist=min_dist,
            metric=metric, random_state=42,  # reproductibilité
        )
        coords = reducer.fit_transform(latents)
    elif method == "isomap" and n >= 10:
        nn = min(max(5, n_neighbors), max(5, n - 1))
        reducer = Isomap(n_components=3, n_neighbors=nn)
        coords = reducer.fit_transform(latents)
    else:
        # PCA : linéaire, garantit distances proportionnelles
        pca = PCA(n_components=d_target)
        coords = pca.fit_transform(latents)
        if coords.shape[1] < 3:
            # Pad zéros si data 1D ou 2D
            pad = np.zeros((n, 3 - coords.shape[1]))
            coords = np.concatenate([coords, pad], axis=1)

    coords = coords.astype(np.float32)

    # Smoothing optionnel : moyenne glissante centrée pour lisser jumps UMAP artefactuels
    if smoothing > 1 and coords.shape[0] > smoothing:
        k = int(smoothing)
        pad = k // 2
        kernel = np.ones(k, dtype=np.float32) / k
        smoothed = np.zeros_like(coords)
        for j in range(coords.shape[1]):
            smoothed[:, j] = np.convolve(coords[:, j], kernel, mode="same")
        # Conserve bords non-lissés pour éviter régression aux extrêmes
        smoothed[:pad] = coords[:pad]
        smoothed[-pad:] = coords[-pad:]
        coords = smoothed

    # Min-max normalisation par axe → cube [-1, 1]³ (esthétique scene Three.js)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    coords = 2 * (coords - mins) / span - 1
    return coords


def lorenz_trajectory(
    n: int = 2000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    return_raw: bool = False,
) -> np.ndarray:
    """Intègre système Lorenz (1963) : équations chaos déterministe canonique.

    Équations :
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz

    Pour σ=10, ρ=28, β=8/3 → attracteur en papillon, exposant Lyapunov ≈ 0.906.
    Méthode : Euler explicite, dt=0.01 (suffisant pour ce régime).

    Args:
        n: nombre de steps temporels
        dt: pas d'intégration
        sigma, rho, beta: paramètres Lorenz canoniques
        return_raw: si True, retourne aussi coords brutes (utile SINDy avec vraies équations)

    Returns:
        coords normalisées (N, 3) ou (norm, raw, dt) si return_raw
    """
    x, y, z = 0.1, 0.0, 0.0
    raw = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        raw[i] = [x, y, z]
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = 2 * (raw - mins) / span - 1
    if return_raw:
        return norm, raw, dt
    return norm
