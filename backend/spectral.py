"""Analyse spectrale : DMD / Koopman operator sur trajectoire latente."""
from __future__ import annotations
import numpy as np


def dmd(latents: np.ndarray, rank: int | None = None, dt: float = 1.0) -> dict:
    """Dynamic Mode Decomposition exact (Tu et al. 2014).
    latents: (N, D) trajectoire temporelle.
    Retourne eigenvalues complexes (Koopman spectrum) + amplitudes + frequences."""
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
    """Spectre de puissance moyen sur dimensions latentes."""
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
