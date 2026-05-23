"""Reconstruction d'espace des phases : Takens delay + PCA + direct embed.

Port de js/pipeline.js (phase-space-video). Trois modes :

1. delayEmbed(series, m, tau) — théorème de Takens 1981 :
   x(t) → [s(t), s(t+τ), s(t+2τ), ..., s(t+(m-1)τ)]
   Reconstruit attracteur d'un système dynamique à partir d'une
   seule observable scalaire, si m ≥ 2*d_box_dim + 1 (Whitney).

2. pcaEmbed(channels, m) — projection sur m premières composantes
   principales d'un set de K canaux observables.

3. directEmbed(channels, [keyX, keyY, keyZ]) — picking de 3 canaux
   pour interpretation directe (e.g. brightness/motion/entropy).

autoTau(series) — heuristique :
   premier minimum local de |autocorrelation| (plus stable que
   crossing 1/e quand autocorr plafonne).
"""
from __future__ import annotations
import numpy as np


def zscore(arr: np.ndarray) -> np.ndarray:
    """Centre + normalise par std. Évite division 0."""
    arr = np.asarray(arr, dtype=np.float64)
    mu = arr.mean()
    sigma = arr.std()
    if sigma < 1e-12:
        sigma = 1.0
    return (arr - mu) / sigma


def auto_tau(series: np.ndarray, max_lag: int | None = None) -> int:
    """Délai optimal = premier min local de |autocorrelation|.

    Args:
        series: 1D array.
        max_lag: borne supérieure du lag testé. Default = max(4, len/8), cap 120.
    Returns:
        τ ≥ 1.
    """
    x = zscore(series)
    n = len(x)
    if max_lag is None:
        max_lag = min(120, max(4, n // 8))
    prev = np.inf
    for lag in range(1, max_lag):
        # autocorr non-normalisée par lag
        c = abs(float((x[:n - lag] * x[lag:]).mean()))
        if c > prev:
            return max(1, lag - 1)
        prev = c
    return max(2, max_lag // 3)


def delay_embed(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Plongement par coordonnées retardées (Takens 1981).

    Args:
        series: 1D array de longueur N.
        m: dimension de plongement (>= 2).
        tau: délai entier (>= 1).
    Returns:
        (N - (m-1)*tau, m) array de vecteurs retardés.
    """
    if m < 2:
        raise ValueError(f"m >= 2 required, got {m}")
    if tau < 1:
        raise ValueError(f"tau >= 1 required, got {tau}")
    x = zscore(series)
    n = len(x)
    count = n - (m - 1) * tau
    if count <= 0:
        raise ValueError(
            f"series too short: need > (m-1)*tau = {(m - 1) * tau}, got {n}"
        )
    out = np.zeros((count, m), dtype=np.float64)
    for d in range(m):
        out[:, d] = x[d * tau : d * tau + count]
    return out


def direct_embed(
    channels: dict[str, np.ndarray],
    keys: list[str],
) -> np.ndarray:
    """Plongement direct : K canaux choisis explicitement comme axes.

    Args:
        channels: dict from compute_observables.
        keys: liste de K clés (typiquement 3).
    Returns:
        (N, K) array zscorée.
    """
    cols = [zscore(channels[k]) for k in keys]
    return np.stack(cols, axis=1)


def pca_embed(
    channels: dict[str, np.ndarray],
    keys: list[str],
    m: int = 3,
    n_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via power iteration + déflation. Compatible JS port.

    Pour m components principales :
    1. zscore chaque canal
    2. Cov matrix d×d
    3. Power iteration → 1er eigenvector, déflation, répéter

    Args:
        channels: dict.
        keys: list de K clés.
        m: nb components souhaités (capé à K).
    Returns:
        (points (N, m), eigenvalues (m,), explained_variance (m,))
    """
    cols = np.stack([zscore(channels[k]) for k in keys], axis=1)  # (N, K)
    n, d = cols.shape
    dim = min(m, d)

    cov = (cols.T @ cols) / n  # (d, d)
    M = cov.copy()

    vecs = np.zeros((d, dim))
    vals = np.zeros(dim)
    rng = np.random.default_rng(0)

    for c in range(dim):
        v = rng.standard_normal(d) - 0.5
        v = v / max(np.linalg.norm(v), 1e-12)
        lam = 0.0
        for _ in range(n_iter):
            w = M @ v
            norm = np.linalg.norm(w)
            if norm < 1e-12:
                break
            w = w / norm
            lam = float(norm)
            v = w
        vecs[:, c] = v
        vals[c] = lam
        # déflation : M ← M - λ v v^T
        M = M - lam * np.outer(v, v)

    points = cols @ vecs  # (N, dim)
    explained = vals / max(vals.sum(), 1e-12)
    return points, vals, explained
