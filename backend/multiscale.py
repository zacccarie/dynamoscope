"""Analyse multi-echelle : coarse-graining + entropie + structures invariantes."""
from __future__ import annotations
import numpy as np


def shannon_entropy(x: np.ndarray, bins: int = 32) -> float:
    """Entropie de Shannon d'un vecteur scalaire."""
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def coarse_grain(latents: np.ndarray, factor: int) -> np.ndarray:
    """Average pooling temporel par facteur."""
    n, d = latents.shape
    new_n = n // factor
    if new_n < 2:
        return latents.copy()
    trimmed = latents[: new_n * factor]
    reshaped = trimmed.reshape(new_n, factor, d)
    return reshaped.mean(axis=1)


def multiscale_entropy(latents: np.ndarray, scales: list[int] | None = None, bins: int = 24) -> dict:
    """Entropie moyenne sur dimensions latentes a chaque echelle de coarse-graining."""
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
    """Pente log-log du power spectrum -> exposant scaling (Kolmogorov-like)."""
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
