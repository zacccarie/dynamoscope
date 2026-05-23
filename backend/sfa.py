"""Slow Feature Analysis (Wiskott-Sejnowski 2002).

Inspiration biologique : cellules visuelles complexes extraient invariances
(identité d'objet préservée à travers rotation, translation, scale).

Principe SFA : trouve combinaisons linéaires W·X qui changent LE PLUS LENTEMENT
dans temps. Minimise <(dWX/dt)²> sous contrainte W·X unit variance + decorrelated.

Output = composantes ordonnées par slowness :
- λ₁ petit = signal le plus lent = identité globale
- λ_k grand = signal rapide = bruit/détails transitoires

Lien causal emergence (Hoel) : composantes lentes = macro-états qui peuvent
avoir effective info supérieure aux micro-états.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


def slow_feature_analysis(
    latents: np.ndarray,
    n_components: int = 8,
    polynomial_expand: int = 1,
) -> dict:
    """Pipeline SFA via generalized eigenvalue problem.

    Étapes :
    1. Center X (zero mean per dim)
    2. (Optionnel) Polynomial expansion order 2 → SFA non-linéaire (quadratique)
    3. Whitening : sphericize covariance via eigh
    4. Compute time derivatives dY/dt (central diff)
    5. Eigendecompose Cov(dY) → lambdas + R
    6. k smallest lambdas = k slowest features

    polynomial_expand=2 ajoute z_i × z_j → permet capturer invariances quadratiques
    (e.g., rotations, scaling-equivariance).
    """
    n, d = latents.shape
    if n < 8:
        raise ValueError("Trajectory too short")

    X = latents.astype(np.float64)
    # Center
    X = X - X.mean(axis=0, keepdims=True)

    # Polynomial expansion optionnel
    if polynomial_expand >= 2:
        n_lin = X.shape[1]
        # Limite croisements pour eviter explosion : top 16 dims
        keep = min(n_lin, 16)
        Xt = X[:, :keep]
        quads = []
        for i in range(keep):
            for j in range(i, keep):
                quads.append(Xt[:, i] * Xt[:, j])
        X = np.concatenate([X, np.stack(quads, axis=1)], axis=1)
        X = X - X.mean(axis=0, keepdims=True)

    # Whitening : sphericise covariance
    cov = X.T @ X / (n - 1)
    # eigh sur cov pour avoir transform whitening
    w, V = eigh(cov)
    # Filtre composantes degenerees
    eps = max(1e-9, 1e-6 * w.max())
    mask = w > eps
    w_pos = w[mask]
    V_pos = V[:, mask]
    W = V_pos / np.sqrt(w_pos)  # whitening matrix
    Y = X @ W  # signal whitenned

    # Derivative temporelle (central diff)
    dY = np.zeros_like(Y)
    dY[1:-1] = (Y[2:] - Y[:-2]) / 2.0
    dY[0] = Y[1] - Y[0]
    dY[-1] = Y[-1] - Y[-2]

    # Covariance derivative
    cov_dY = dY.T @ dY / (n - 1)

    # Eigvalues of cov_dY = slowness scores
    lambdas, R = eigh(cov_dY)
    # Tri ascendant : slowness petit = signal lent
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    R = R[:, order]

    k = min(n_components, Y.shape[1])
    slow = Y @ R[:, :k]  # (N, k) slow features
    mixing = W @ R[:, :k]  # latent_dim → slow_features

    return {
        "slow_features": slow.tolist(),
        "lambdas": lambdas[:k].tolist(),
        "slowness": [float(l) for l in lambdas[:k]],
        "n_components": int(k),
        "mixing_shape": list(mixing.shape),
        "polynomial_expand": int(polynomial_expand),
    }
