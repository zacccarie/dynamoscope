"""Analyse multi-échelle : coarse-graining + entropie + structures invariantes.

Inspiré renormalization group (Wilson 1971). Idée : à différentes échelles
temporelles, quelles structures persistent ? Quelles disparaissent ?

Si entropie ≈ constante à toutes échelles → système scale-invariant (fractal).
Si entropie augmente puis baisse → échelle préférée (résonance).
"""
from __future__ import annotations
import numpy as np


def shannon_entropy(x: np.ndarray, bins: int = 32) -> float:
    """Entropie de Shannon H(X) = -Σ p(x) log₂ p(x).

    Quantifie incertitude/diversité de distribution.
    H=0 : signal constant. H=log₂(bins) : uniforme (max info).
    Pour distribution gaussienne : H ≈ ½ log(2πeσ²).
    Estimé via histogramme à `bins` cases.
    """
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def coarse_grain(latents: np.ndarray, factor: int) -> np.ndarray:
    """Average pooling temporel : remplace `factor` consecutifs frames par leur moyenne.

    Step de renormalization group : zoom out temporel.
    Si `factor=2`, vue divisée par 2 (T → T/2).
    Préserve magnitude, perd hautes fréquences.
    """
    n, d = latents.shape
    new_n = n // factor
    if new_n < 2:
        return latents.copy()
    trimmed = latents[: new_n * factor]
    reshaped = trimmed.reshape(new_n, factor, d)
    return reshaped.mean(axis=1)


def multiscale_entropy(latents: np.ndarray, scales: list[int] | None = None, bins: int = 24) -> dict:
    """Entropy ladder : H(z) à plusieurs échelles de coarse-graining (1, 2, 4, 8, 16, 32).

    Révèle scale-invariance (entropy plate = fractal/chaos) ou échelle préférée (peak).
    Lien direct avec :
    - Multiscale entropy (Costa et al. 2002) — biomarker complexité physiologique
    - Renormalization group analysis — physique statistique
    - Effective scale of organization — émergence
    """
    if scales is None:
        scales = [1, 2, 4, 8, 16, 32]
    out = []
    for s in scales:
        cg = coarse_grain(latents, s)
        if cg.shape[0] < 4:
            break
        # Moyenne entropie sur D dimensions
        ent = np.mean([shannon_entropy(cg[:, j], bins=bins) for j in range(cg.shape[1])])
        # Variance totale (energie restante)
        var = float(np.var(cg))
        out.append({
            "scale": int(s),
            "n_points": int(cg.shape[0]),
            "entropy": float(ent),
            "variance": var,
        })
    return {"ladder": out, "max_scale": int(scales[-1])}


def spectral_slope(latents: np.ndarray) -> float:
    """Exposant α de la loi puissance P(f) ∝ f^(-α) via régression log-log.

    Interprétation classique :
    - α = 0 : bruit blanc (no structure)
    - α = 1 : 1/f noise (Pareto, signal naturel complexe)
    - α = 2 : bruit brownien (random walk, mémoire)
    - α = 3 : chaos déterministe (Kolmogorov cascade turbulence)
    - α > 3 : trajectoire ultra-régulière
    """
    n = latents.shape[0]
    if n < 16:
        return 0.0
    fft = np.fft.rfft(latents - latents.mean(axis=0, keepdims=True), axis=0)
    power = (np.abs(fft) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Eviter DC + bord Nyquist
    mask = (freqs > 0) & (power > 0)
    if mask.sum() < 4:
        return 0.0
    log_f = np.log(freqs[mask])
    log_p = np.log(power[mask])
    slope = np.polyfit(log_f, log_p, 1)[0]
    return float(slope)
