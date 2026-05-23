"""Tests SINDy : doit retrouver équations Lorenz canoniques."""
import numpy as np
from backend.sindy import fit_sindy, feature_library, _poly_terms, central_diff, stlsq


def test_poly_terms_count():
    """Nombre de monômes en d=3, order=2 = C(3+2, 2) = 10."""
    terms = _poly_terms(3, 2)
    assert len(terms) == 10
    assert (0, 0, 0) in terms  # constant


def test_feature_library_shape():
    """Theta(X) shape = (N, P)."""
    X = np.random.randn(50, 3)
    Theta, terms = feature_library(X, order=2)
    assert Theta.shape[0] == 50
    assert Theta.shape[1] == len(terms)


def test_central_diff_shape():
    """Derivative same shape as input."""
    X = np.random.randn(100, 3)
    dX = central_diff(X, dt=0.01)
    assert dX.shape == X.shape


def test_central_diff_constant_zero():
    """Signal constant → dérivée ≈ 0."""
    X = np.ones((50, 3))
    dX = central_diff(X)
    # Tous interior points = 0; bords aussi 0 car constant
    assert np.allclose(dX, 0)


def test_stlsq_sparsity():
    """STLSQ doit produire matrice sparse (coefficients sous seuil → 0)."""
    rng = np.random.RandomState(0)
    Theta = rng.randn(100, 10)
    # Cible : seulement 2 colonnes contribuent
    true_Xi = np.zeros((10, 1))
    true_Xi[2, 0] = 3.0
    true_Xi[5, 0] = -2.0
    dX = Theta @ true_Xi + 0.01 * rng.randn(100, 1)
    Xi = stlsq(Theta, dX, threshold=0.5)
    # Seulement les 2 vraies colonnes doivent être non-nulles
    n_active = int(np.sum(np.abs(Xi) > 1e-6))
    assert 1 <= n_active <= 3  # tolérance bruit


def test_sindy_recovers_lorenz(lorenz_raw):
    """SINDy sur Lorenz brut : retrouve équations canoniques avec R² > 0.95."""
    raw, dt = lorenz_raw
    out = fit_sindy(raw, dt=dt, order=2, threshold=0.5, var_names=["x", "y", "z"])
    assert out["r2"] > 0.95
    # Doit avoir 3 équations
    assert len(out["equations"]) == 3
    # Au moins quelques termes actifs (Lorenz a 7 termes)
    assert out["n_active"] >= 5
    # Vérifier équation dx/dt = -10x + 10y (coefficients ±10)
    eq_x = out["equations"][0]
    coefs = [t["coef"] for t in eq_x["terms"]]
    has_neg_10 = any(-11 < c < -8 for c in coefs)
    has_pos_10 = any(8 < c < 11 for c in coefs)
    assert has_neg_10 and has_pos_10
