"""Wavelet multi-resolution analysis sur trajectoire latente.

DWT (Discrete Wavelet Transform) décompose signal en :
- approximation (basses fréquences = composantes lentes)
- détails à chaque scale (hautes fréquences localisées)

Plus translation-invariant que FFT pour signaux non-stationnaires.
Lien Mallat scattering : approximation pyramid + log-magnitudes des détails.

Use case : pour chaque dim latent, compute multi-scale energy distribution.
Reveals temporal structure à différentes échelles (vs entropy ladder = stat).
"""
from __future__ import annotations
import numpy as np
import pywt


def wavelet_decompose(
    series: np.ndarray,
    wavelet: str = "db4",
    max_level: int | None = None,
) -> dict:
    """DWT 1D multi-niveau sur signal scalaire.

    Args:
        series: signal 1D (N,)
        wavelet: type ondelette — 'db4' (Daubechies-4, default),
                 'haar' (simple), 'sym8' (Symlet 8), 'coif5' (Coiflet 5)
        max_level: niveaux max (auto si None = log2(N))

    Returns:
        - approximation (cA_n): basse fréquence niveau max
        - details: liste [cD_1, ..., cD_n] coefficients par niveau
        - energy_per_scale: énergie ‖cD‖² par niveau (signature spectrale)
        - dominant_scale: scale avec énergie max
    """
    n = len(series)
    if max_level is None:
        max_level = min(int(np.log2(n)) - 2, 8)
    coeffs = pywt.wavedec(series, wavelet, level=max_level)
    cA = coeffs[0]
    details = coeffs[1:]

    energy = np.array([float(np.sum(d ** 2)) for d in details])
    if energy.sum() > 1e-12:
        energy_norm = energy / energy.sum()
    else:
        energy_norm = energy

    dominant = int(np.argmax(energy))

    return {
        "wavelet": wavelet,
        "n_levels": int(max_level),
        "approximation": cA.tolist(),
        "approximation_size": int(cA.shape[0]),
        "detail_sizes": [int(d.shape[0]) for d in details],
        "energy_per_scale": energy.tolist(),
        "energy_normalized": energy_norm.tolist(),
        "dominant_scale": dominant + 1,  # 1-indexed
        "total_energy": float(energy.sum()),
    }


def wavelet_per_dim(
    latents: np.ndarray,
    wavelet: str = "db4",
    max_level: int | None = None,
) -> dict:
    """Applique DWT sur chaque dimension latent, agrège stats.

    Pour latents (N, D), retourne :
    - mean_energy_per_scale : énergie moyennée sur D dims
    - global_dominant_scale : scale avec max énergie agrégée
    """
    n, d = latents.shape
    all_energies = []
    for j in range(d):
        out = wavelet_decompose(latents[:, j], wavelet=wavelet, max_level=max_level)
        all_energies.append(out["energy_per_scale"])
    # Padding pour longueurs égales (rare car même n_levels)
    max_len = max(len(e) for e in all_energies)
    padded = np.array([e + [0.0] * (max_len - len(e)) for e in all_energies])
    mean_energy = padded.mean(axis=0)
    total = mean_energy.sum()
    energy_norm = mean_energy / max(total, 1e-12)
    return {
        "wavelet": wavelet,
        "n_levels": int(max_len),
        "mean_energy_per_scale": mean_energy.tolist(),
        "energy_normalized": energy_norm.tolist(),
        "global_dominant_scale": int(np.argmax(mean_energy)) + 1,
        "n_dims_processed": int(d),
        "total_energy": float(total),
    }
