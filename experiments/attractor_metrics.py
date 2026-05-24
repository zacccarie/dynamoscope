"""Attractor-preservation metrics pour rollout eval.

Plus que MSE : est-ce que la trajectoire prédite préserve la structure
dynamique de la GT ? Quatre familles de tests :

1. Spectral match : cosine similarity entre power spectra
   → capte phases + fréquences dominantes
2. Recurrence DET : determinisme via diagonal lines de la recurrence matrix
   → capte structure cyclique
3. Autocorrelation match : ACF similarity
   → capte phase + memory horizon
4. Norm boundedness : trajectoire finite et bornée

Tous opèrent sur le latent z_slow (predicted vs GT) ou sur la projection 1D
(norme par step). Plus rapide qu'une PH/correlation dim full.
"""
from __future__ import annotations
import numpy as np


def power_spectrum(z: np.ndarray) -> np.ndarray:
    """1D power spectrum of trajectory norm. z : (K, D) → (K//2+1,)."""
    sig = np.linalg.norm(z, axis=1)
    sig = sig - sig.mean()
    fft = np.fft.rfft(sig)
    return np.abs(fft) ** 2


def spectral_match(pred: np.ndarray, gt: np.ndarray) -> float:
    """Cosine similarity entre power spectra de pred et gt."""
    p = power_spectrum(pred); g = power_spectrum(gt)
    n = min(len(p), len(g))
    if n < 2: return 0.0
    p, g = p[:n], g[:n]
    if p.sum() < 1e-12 or g.sum() < 1e-12: return 0.0
    cos = (p * g).sum() / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-12)
    return float(cos)


def autocorr(z: np.ndarray, max_lag: int = 16) -> np.ndarray:
    """ACF of trajectory norm signal. Returns (max_lag,)."""
    sig = np.linalg.norm(z, axis=1)
    sig = sig - sig.mean()
    var = (sig * sig).sum() + 1e-12
    n = len(sig)
    max_lag = min(max_lag, n - 1)
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = (sig[:-lag] * sig[lag:]).sum() / var
    return acf


def autocorr_match(pred: np.ndarray, gt: np.ndarray, max_lag: int = 16
                    ) -> float:
    """L1 distance entre ACF de pred et gt, mapped to [0, 1] similarity."""
    a = autocorr(pred, max_lag); b = autocorr(gt, max_lag)
    n = min(len(a), len(b))
    if n < 2: return 0.0
    diff = np.abs(a[:n] - b[:n]).mean()
    return float(np.exp(-diff))  # 1 if match perfect, 0 if very different


def recurrence_DET(z: np.ndarray, eps_quantile: float = 0.2,
                    min_diag: int = 2) -> float:
    """Soft DET : fraction of recurrence matrix points falling on diagonals
    of length ≥ min_diag. Captures determinism / periodicity.
    """
    K = z.shape[0]
    if K < min_diag + 4: return 0.0
    # Pairwise distances
    d_ij = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    mask = np.eye(K, dtype=bool)
    for k in range(-1, 2):
        if abs(k) < K:
            mask |= np.eye(K, k=k, dtype=bool)
    if (~mask).sum() < 4: return 0.0
    eps = np.quantile(d_ij[~mask], eps_quantile)
    R = (d_ij < eps).astype(int)
    # Count points on diagonals of length ≥ min_diag (offsets ≠ 0)
    diag_pts = 0
    total_pts = R.sum()
    for off in range(-(K - min_diag), K - min_diag + 1):
        if abs(off) < min_diag: continue
        diag = np.diag(R, k=off)
        # Find runs of consecutive 1s
        runs = []
        cur = 0
        for v in diag:
            if v == 1: cur += 1
            else:
                if cur >= min_diag: runs.append(cur)
                cur = 0
        if cur >= min_diag: runs.append(cur)
        diag_pts += sum(runs)
    return float(diag_pts) / max(total_pts, 1)


def recurrence_DET_match(pred: np.ndarray, gt: np.ndarray) -> float:
    """1 − |DET(pred) − DET(gt)| ∈ [0, 1]."""
    return float(1.0 - abs(recurrence_DET(pred) - recurrence_DET(gt)))


def is_bounded(z: np.ndarray, threshold: float = 1e3) -> bool:
    """Bornée + finite."""
    if not np.isfinite(z).all(): return False
    return bool(np.linalg.norm(z, axis=1).max() < threshold)


def lyap_proxy(z: np.ndarray, tau: int = 2) -> float:
    """Mean log-growth of nearest-pair distances over horizon tau.
    Positive = divergent locally (chaotic). Zero = neutral. Negative = contract.
    """
    K = z.shape[0]
    if K < tau + 4: return 0.0
    d_ij = np.linalg.norm(z[:-tau, None, :] - z[:-tau, :], axis=-1)
    np.fill_diagonal(d_ij, np.inf)
    nn_idx = d_ij.argmin(axis=1)
    nn_d0 = d_ij[np.arange(K - tau), nn_idx]
    valid = np.isfinite(nn_d0) & (nn_d0 > 1e-9)
    if valid.sum() < 4: return 0.0
    anchors = np.arange(K - tau)[valid]
    neighbors = nn_idx[valid]
    d0 = nn_d0[valid]
    d_tau = np.linalg.norm(z[anchors + tau] - z[neighbors + tau], axis=-1)
    d_tau = np.maximum(d_tau, 1e-9)
    return float((np.log(d_tau) - np.log(d0)).mean())


def lyap_match(pred: np.ndarray, gt: np.ndarray) -> float:
    """1 − |λ_pred − λ_gt| mapped to similarity in [0, 1]."""
    diff = abs(lyap_proxy(pred) - lyap_proxy(gt))
    return float(np.exp(-diff))


def attractor_score(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Aggregate metrics. Returns dict — small + interpretable.
    pred, gt : (K, D) numpy arrays.
    """
    return {
        "bounded": is_bounded(pred),
        "spectral_match": spectral_match(pred, gt),
        "autocorr_match": autocorr_match(pred, gt),
        "recurrence_DET_match": recurrence_DET_match(pred, gt),
        "lyap_match": lyap_match(pred, gt),
        "lyap_pred": lyap_proxy(pred),
        "lyap_gt": lyap_proxy(gt),
        "DET_pred": recurrence_DET(pred),
        "DET_gt": recurrence_DET(gt),
    }


def aggregate_attractor(scores_list: list[dict]) -> dict:
    """Mean/std across multiple trajectories."""
    keys = scores_list[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in scores_list]
        if isinstance(vals[0], bool):
            out[f"{k}_frac"] = float(np.mean(vals))
        else:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out
