"""SINDy : Sparse Identification of Nonlinear Dynamics (Brunton-Proctor-Kutz 2016).

But : depuis trajectoire (z[t]), retrouver équations différentielles symboliques
dz/dt = f(z). Approche : sparse regression dans bibliothèque polynomiale.

Hypothèse : f(z) = somme de peu de termes parmi {1, z_i, z_i z_j, z_i², ...}.
Sparsity (peu de termes actifs) = inductive bias correct pour physique.

Algorithme STLSQ (Sequential Thresholded Least Squares) :
1. Initial fit moindres carrés sur library complète
2. Seuil : coefficients |c| < threshold → 0
3. Refit sur termes actifs restants
4. Itérer jusqu'à convergence

Pour Lorenz σ=10, ρ=28, β=8/3, SINDy retrouve équations avec R²>0.99.
"""
from __future__ import annotations
import itertools
import numpy as np


def _poly_terms(d: int, order: int) -> list[tuple[int, ...]]:
    """Génère liste de tous monômes polynomiaux jusqu'à degré `order` en d variables.

    Ex : d=2, order=2 → [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
    représente {1, x, y, x², xy, y²} en notation multi-index.
    """
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
    """Construit matrice Θ(X) où chaque colonne = un terme polynomial évalué sur X.

    Theta(X) ∈ R^{N×P} où P = nombre de monômes jusqu'à degré `order`.
    Optionnellement ajoute sin(x), cos(x) pour systèmes oscillants.
    Sert de "bibliothèque" pour sparse regression : on cherche peu de colonnes actives.
    """
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
    """Approximation dérivée temporelle par différence centrée.

    dX/dt[i] ≈ (X[i+1] - X[i-1]) / (2 dt)
    Erreur O(dt²) (vs O(dt) pour forward/backward).
    Bords : forward/backward de 1er ordre.
    """
    n, d = X.shape
    dX = np.zeros_like(X, dtype=np.float64)
    dX[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX


def stlsq(Theta: np.ndarray, dX: np.ndarray, threshold: float = 0.05, n_iter: int = 12) -> np.ndarray:
    """STLSQ : Sequential Thresholded Least Squares (Brunton et al. 2016).

    Algorithme iteratif :
    1. Xi = lstsq(Θ, dX) — fit initial moindres carrés
    2. Zéro coefficients où |Xi| < threshold
    3. Refit lstsq sur colonnes restantes seulement
    4. Répète jusqu'à convergence (typiquement 5-15 itérations)

    Le seuil contrôle la sparsité : seuil élevé = peu de termes (modèle simple),
    seuil bas = beaucoup termes (overfit risque).

    Plus simple et plus stable que LASSO en pratique pour SINDy.
    """
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
    """Pipeline SINDy complet : trajectoire X(t) → équations dx_i/dt = Σ c_ij φ_j(x).

    Étapes :
    1. Lissage léger (moyenne 3-points) pour stabilité dérivées
    2. Dérivées centrales dX/dt
    3. Construit bibliothèque polynomiale Θ(X)
    4. STLSQ : régression sparse Θ → dX/dt
    5. Construit équations symboliques lisibles depuis coefficients actifs
    6. Calcule R² reconstruction

    Returns: équations formatées + R² + métadonnées sparse.
    """
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
