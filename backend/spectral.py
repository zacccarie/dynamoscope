"""Analyse spectrale : DMD / Koopman operator sur trajectoire latente.

DMD (Dynamic Mode Decomposition, Schmid 2010) = SVD-based eigendecomposition de
l'opérateur linéaire qui mappe z[t] → z[t+1]. Trouve modes propres = orbites
quasi-périodiques + leur taux de décroissance.

Koopman operator (Koopman 1931) = opérateur linéaire infini-dim qui agit sur
observables d'un système dynamique non-linéaire. DMD approxime ce spectre
en dimension finie. Eigvals sur cercle unité = oscillations stables.

Power spectrum FFT = analyse fréquentielle classique = composantes périodiques.
"""
from __future__ import annotations
import numpy as np


def dmd(latents: np.ndarray, rank: int | None = None, dt: float = 1.0) -> dict:
    """Exact DMD (Tu et al. 2014) — approxime Koopman opérateur.

    Théorie : trouve matrice A telle que latents[t+1] ≈ A · latents[t].
    Decompose A en eigenvalues complexes λ (modes) + amplitudes.
    Eigvalue interpretation :
      - |λ| ≈ 1 : mode stable oscillant
      - |λ| < 1 : mode décroissant (transient)
      - |λ| > 1 : mode croissant (instable)
      - arg(λ) : fréquence angulaire du mode

    Algorithme :
    1. SVD tronquée de X = latents[:-1].T
    2. Projection A_tilde = U* Y V S^-1 (espace réduit)
    3. Eigendecomposition de A_tilde
    4. Reconstruction modes Phi dans espace original

    Args:
        latents: (N, D) trajectoire temporelle
        rank: troncature SVD (auto si None : 90% énergie cumulée)
        dt: pas temporel pour calcul fréquences continues

    Returns:
        Dict avec eigvals_re/im, amplitudes, freqs, decay, singular_values.
    """
    n, d = latents.shape
    if n < 4:
        raise ValueError("Need >= 4 timesteps for DMD")

    X = latents[:-1].T  # (D, N-1)
    Y = latents[1:].T   # (D, N-1)

    # SVD tronquee X = U S V*
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        # Garder modes dominants (90% energie cumulee)
        cum = np.cumsum(S) / S.sum()
        rank = int(np.searchsorted(cum, 0.90) + 1)
        rank = max(2, min(rank, len(S), n - 1))
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank].conj().T  # (N-1, rank)

    # A_tilde = U* Y V S^-1
    A_tilde = U_r.conj().T @ Y @ V_r @ np.diag(1.0 / np.maximum(S_r, 1e-12))

    # Eigendecomposition
    eigvals, W = np.linalg.eig(A_tilde)

    # Modes DMD : Phi = Y V S^-1 W
    Phi = Y @ V_r @ np.diag(1.0 / np.maximum(S_r, 1e-12)) @ W

    # Amplitudes b : initial coefficients
    x0 = latents[0]
    b, *_ = np.linalg.lstsq(Phi, x0, rcond=None)

    # Frequences continues : log(eigval)/dt -> omega complexe
    safe = np.where(np.abs(eigvals) > 1e-12, eigvals, 1e-12)
    omega = np.log(safe) / dt
    freqs = np.imag(omega) / (2 * np.pi)
    decay = np.real(omega)

    return {
        "rank": int(rank),
        "n_modes": int(len(eigvals)),
        "eigvals_re": np.real(eigvals).tolist(),
        "eigvals_im": np.imag(eigvals).tolist(),
        "amplitudes": np.abs(b).tolist(),
        "freqs": freqs.tolist(),
        "decay": decay.tolist(),
        "singular_values": S.tolist(),
    }


def power_spectrum(latents: np.ndarray, top_k: int = 64) -> dict:
    """Power spectrum via FFT — composantes fréquentielles signal.

    |FFT(z)|² = puissance par fréquence. Pour signal temporel :
    - Pic dominant = fréquence principale (e.g., BPM si vidéo musique)
    - Slope log-log = scaling : -2 (Brownian), -3 (chaos), -1 (1/f noise)
    - Largeur pics = stabilité oscillations
    """
    n, d = latents.shape
    # FFT par dimension, moyenne en magnitude
    fft = np.fft.rfft(latents - latents.mean(axis=0, keepdims=True), axis=0)
    power = (np.abs(fft) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Garde top_k frequences
    k = min(top_k, len(power))
    return {
        "freqs": freqs[:k].tolist(),
        "power": power[:k].tolist(),
        "total_energy": float(power.sum()),
    }
