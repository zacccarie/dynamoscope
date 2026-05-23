"""SINDy : Sparse Identification of Nonlinear Dynamics (Brunton 2016).
Recupere equations differentielles explicites depuis trajectoire."""
from __future__ import annotations
import itertools
import numpy as np


def _poly_terms(d: int, order: int) -> list[tuple[int, ...]]:
    """Genere multi-indices polynomiaux jusqu'a order, dim d."""
    terms: list[tuple[int, ...]] = [(0,) * d]  # constante
    for o in range(1, order + 1):
        for combo in itertools.combinations_with_replacement(range(d), o):
            multi = [0] * d
            for k in combo:
                multi[k] += 1
            terms.append(tuple(multi))
    return terms


def _term_label(multi: tuple[int, ...], var_names: list[str]) -> str:
    if all(m == 0 for m in multi):
        return "1"
    parts: list[str] = []
    for i, m in enumerate(multi):
        if m == 0:
            continue
        if m == 1:
            parts.append(var_names[i])
        else:
            parts.append(f"{var_names[i]}^{m}")
    return "·".join(parts)


def feature_library(X: np.ndarray, order: int = 2, include_trig: bool = False) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Construit matrice Theta(X) : chaque colonne = un terme polynomial."""
    n, d = X.shape
    terms = _poly_terms(d, order)
    cols: list[np.ndarray] = []
    for multi in terms:
        col = np.ones(n, dtype=np.float64)
        for i, m in enumerate(multi):
            if m > 0:
                col = col * (X[:, i] ** m)
        cols.append(col)
    if include_trig:
        for i in range(d):
            cols.append(np.sin(X[:, i]))
            terms.append(("sin", i))
            cols.append(np.cos(X[:, i]))
            terms.append(("cos", i))
    Theta = np.stack(cols, axis=1)
    return Theta, terms


def central_diff(X: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Derivee par difference centrale, edges par diff avant/arriere."""
    n, d = X.shape
    dX = np.zeros_like(X, dtype=np.float64)
    dX[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX


def stlsq(Theta: np.ndarray, dX: np.ndarray, threshold: float = 0.05, n_iter: int = 12) -> np.ndarray:
    """Sequential Thresholded Least Squares (Brunton).
    Theta (n, p), dX (n, d). Retourne Xi (p, d) coefficients."""
    p = Theta.shape[1]
    d = dX.shape[1]
    Xi, *_ = np.linalg.lstsq(Theta, dX, rcond=None)
    for _ in range(n_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for k in range(d):
            big = ~small[:, k]
            if big.any():
                Xi[big, k], *_ = np.linalg.lstsq(Theta[:, big], dX[:, k], rcond=None)
    return Xi


def fit_sindy(
    X: np.ndarray,
    dt: float = 1.0,
    order: int = 2,
    threshold: float = 0.05,
    var_names: list[str] | None = None,
) -> dict:
    """Pipeline SINDy : trajectoire -> equations dx_i/dt = sum c_ij * phi_j(x)."""
    n, d = X.shape
    if n < 30:
        raise ValueError("Trajectory too short for SINDy")
    if var_names is None:
        var_names = [f"x{i}" for i in range(d)]

    # Lisse legerement pour stabilite derivees (moyenne 3-points)
    X_smooth = X.astype(np.float64).copy()
    X_smooth[1:-1] = (X[:-2] + 2 * X[1:-1] + X[2:]) / 4

    dX = central_diff(X_smooth, dt=dt)
    Theta, terms = feature_library(X_smooth, order=order)
    Xi = stlsq(Theta, dX, threshold=threshold)

    # Construit equations lisibles
    equations: list[dict] = []
    for k in range(d):
        coeffs = Xi[:, k]
        active = np.where(np.abs(coeffs) > 1e-12)[0]
        parts: list[dict] = []
        for j in active:
            parts.append({
                "coef": float(coeffs[j]),
                "term": _term_label(terms[j], var_names),
            })
        equations.append({
            "lhs": f"d{var_names[k]}/dt",
            "terms": parts,
        })

    # R^2 reconstruction
    dX_pred = Theta @ Xi
    ss_res = float(np.sum((dX - dX_pred) ** 2))
    ss_tot = float(np.sum((dX - dX.mean(axis=0)) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)

    return {
        "equations": equations,
        "r2": round(r2, 4),
        "n_active": int(np.sum(np.abs(Xi) > 1e-12)),
        "n_terms_lib": int(Theta.shape[1]),
        "order": int(order),
        "threshold": float(threshold),
    }
