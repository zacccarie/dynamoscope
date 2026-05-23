"""Reduction dimensionnelle : UMAP / PCA / Isomap 3D."""
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
    """(N, D) -> (N, 3) float32, normalise dans [-1, 1].
    methods : umap, pca, isomap.
    n_neighbors : taille voisinage local (UMAP/Isomap).
    min_dist : compacité cluster (UMAP).
    metric : cosine / euclidean / manhattan / correlation (UMAP).
    smoothing : fenetre moyenne glissante post-projection (0 = off)."""
    n = latents.shape[0]
    d_target = min(3, n, latents.shape[1])

    if method == "umap" and n >= 5:
        nn = min(max(2, n_neighbors), max(2, n - 1))
        reducer = umap.UMAP(
            n_components=3, n_neighbors=nn, min_dist=min_dist,
            metric=metric, random_state=42,
        )
        coords = reducer.fit_transform(latents)
    elif method == "isomap" and n >= 10:
        nn = min(max(5, n_neighbors), max(5, n - 1))
        reducer = Isomap(n_components=3, n_neighbors=nn)
        coords = reducer.fit_transform(latents)
    else:
        # PCA = linéaire orthogonale, préserve distances proportionnellement
        pca = PCA(n_components=d_target)
        coords = pca.fit_transform(latents)
        if coords.shape[1] < 3:
            pad = np.zeros((n, 3 - coords.shape[1]))
            coords = np.concatenate([coords, pad], axis=1)

    coords = coords.astype(np.float32)
    # Smoothing temporel post-projection (moyenne glissante centrée)
    if smoothing > 1 and coords.shape[0] > smoothing:
        k = int(smoothing)
        pad = k // 2
        kernel = np.ones(k, dtype=np.float32) / k
        smoothed = np.zeros_like(coords)
        for j in range(coords.shape[1]):
            smoothed[:, j] = np.convolve(coords[:, j], kernel, mode="same")
        # Conserver bords non-lisses pour eviter regression aux extremes
        smoothed[:pad] = coords[:pad]
        smoothed[-pad:] = coords[-pad:]
        coords = smoothed
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
    """Genere trajectoire Lorenz pour demo. (N, 3) normalise [-1, 1].
    Si return_raw=True, retourne (normalised, raw, dt)."""
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
