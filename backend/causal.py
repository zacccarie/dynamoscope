"""Analyse causale : qui influence qui dans les séries temporelles latentes ?

3 méthodes complémentaires :
- **Granger causality** (1969) : X cause Y si prédiction de Y améliore avec X dans modèle linéaire
- **Transfer Entropy** (Schreiber 2000) : information transférée X→Y au-delà info Y intrinsèque
- **CCM** (Sugihara 2012, Convergent Cross Mapping) : Y cause X si l'historique de X contient info sur Y
  (subtil : direction CCM = X→Y signifie Y est cause, X est effet)

Granger linéaire, TE info-théorique, CCM topologique. Triangulation pour robustesse.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def _lag_matrix(x: np.ndarray, lag: int) -> np.ndarray:
    """Construit matrice (N-lag, lag) où chaque ligne = lag valeurs passées.

    Utilisé pour régression Granger : modèle Y_t = β·[Y_{t-lag},...,Y_{t-1}].
    """
    n = len(x)
    if lag <= 0 or lag >= n:
        return np.zeros((0, lag))
    out = np.zeros((n - lag, lag))
    for k in range(lag):
        out[:, k] = x[k : n - lag + k]
    return out


def granger_pairwise(series: np.ndarray, lag: int = 3) -> np.ndarray:
    """Test Granger pairwise : pour chaque paire (i, j), j cause-t-il i ?

    Méthode : compare 2 régressions :
      restreint : Y_i = f(Y_i passé)
      full      : Y_i = f(Y_i passé, Y_j passé)
    Si full bat restreint significativement → j cause Granger i.

    Score = (RSS_restreint - RSS_full) / RSS_full = amélioration relative.
    Score > 0 = j → i (j est cause Granger de i au lag donné).

    Args:
        series: (N, D) multivariate time series
        lag: nombre de pas passés inclus dans modèle

    Returns:
        (D, D) matrix où entry [j,i] = causalité de j vers i
    """
    n, d = series.shape
    if n <= lag + 2:
        return np.zeros((d, d))

    # Standardise pour stabilite numerique
    series = (series - series.mean(axis=0, keepdims=True)) / np.maximum(series.std(axis=0, keepdims=True), 1e-9)

    scores = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        y = series[lag:, i]
        Xi = _lag_matrix(series[:, i], lag)
        # Modele restreint : i seule
        beta_r, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        rss_r = float(np.sum((y - Xi @ beta_r) ** 2))
        for j in range(d):
            if j == i:
                continue
            Xj = _lag_matrix(series[:, j], lag)
            Xij = np.concatenate([Xi, Xj], axis=1)
            beta_f, *_ = np.linalg.lstsq(Xij, y, rcond=None)
            rss_f = float(np.sum((y - Xij @ beta_f) ** 2))
            # F-stat reduit a une amelioration relative (degrees ignored pour rapidite)
            if rss_f < 1e-6:
                scores[j, i] = 0.0
            else:
                # Clamp pour eviter explosions numeriques sur signaux quasi-deterministes
                ratio = (rss_r - rss_f) / rss_f
                scores[j, i] = float(np.clip(ratio, 0.0, 100.0))
    return scores


def transfer_entropy_binned(x: np.ndarray, y: np.ndarray, lag: int = 1, bins: int = 6) -> float:
    """Transfer Entropy y → x via histogramme conditionnel (Schreiber 2000).

    Formule : TE_{y→x} = I(x_t ; y_{t-lag} | x_{t-lag})
                       = "information additionnelle apportée par y_passé pour prédire x_actuel
                          au-delà de ce que x_passé prédit déjà"

    Différence avec Mutual Information : TE est asymétrique (conditionne sur x_passé).
    Plus robuste que Granger pour relations non-linéaires (mesure info-théorique).

    Unité : bits (log base 2).
    """
    n = len(x)
    if n <= lag + 4:
        return 0.0
    x_now = x[lag:]
    x_past = x[:-lag]
    y_past = y[:-lag]

    edges = lambda v: np.linspace(v.min(), v.max() + 1e-9, bins + 1)
    bx = np.clip(np.digitize(x_now, edges(x_now)) - 1, 0, bins - 1)
    bxp = np.clip(np.digitize(x_past, edges(x_past)) - 1, 0, bins - 1)
    byp = np.clip(np.digitize(y_past, edges(y_past)) - 1, 0, bins - 1)

    # Histogrammes joints
    p_xxy = np.zeros((bins, bins, bins))
    p_xx = np.zeros((bins, bins))
    p_xy = np.zeros((bins, bins))
    p_x = np.zeros(bins)
    for a, b, c in zip(bx, bxp, byp):
        p_xxy[a, b, c] += 1
        p_xx[a, b] += 1
        p_xy[b, c] += 1
        p_x[b] += 1
    total = max(len(bx), 1)
    p_xxy /= total
    p_xx /= total
    p_xy /= total
    p_x /= total

    te = 0.0
    for a in range(bins):
        for b in range(bins):
            for c in range(bins):
                pj = p_xxy[a, b, c]
                if pj <= 0:
                    continue
                num = pj * p_x[b]
                den = p_xx[a, b] * p_xy[b, c]
                if den > 0 and num > 0:
                    te += pj * np.log2(num / den)
    return float(max(0.0, te))


def transfer_entropy_matrix(series: np.ndarray, lag: int = 1, bins: int = 6) -> np.ndarray:
    """Matrice TE complète (D, D) : pour chaque paire (j, i), TE_{j→i}."""
    n, d = series.shape
    out = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            out[j, i] = transfer_entropy_binned(series[:, i], series[:, j], lag=lag, bins=bins)
    return out


def ccm_sugihara(x: np.ndarray, y: np.ndarray, e: int = 3, tau: int = 1, lib_size: int | None = None) -> float:
    """Convergent Cross Mapping (Sugihara et al. 2012).

    Théorème généralisé de Takens : si Y cause X dans système dynamique unifié,
    alors l'historique de X contient des traces de Y. CCM teste cette implication.

    Algorithme :
    1. Construit shadow manifold M_X via Takens embedding de x
    2. Pour chaque point de M_X, trouve e+1 nearest neighbors temporels
    3. Prédit y(t) comme moyenne pondérée des y aux temps voisins
    4. Skill = corrélation(y_vrai, y_prédit)

    Interpretation : CCM(x → y) élevé signifie Y est cause de X.
    (Direction sémantique inversée vs Granger ! Sugihara teste implication causale via topologie.)

    Utile pour systèmes déterministes non-linéaires (chaos, écosystèmes).
    """
    n = len(x)
    span = (e - 1) * tau
    if n <= span + 5:
        return 0.0
    # Shadow manifold de x
    Mx = np.zeros((n - span, e))
    for j in range(e):
        Mx[:, j] = x[j * tau : j * tau + (n - span)]
    y_aligned = y[span:]

    if lib_size is None:
        lib_size = len(Mx)
    lib_size = min(lib_size, len(Mx))

    tree = cKDTree(Mx[:lib_size])
    # Pour chaque point cible (tout l'echantillon), trouver e+1 voisins dans lib
    dists, idx = tree.query(Mx, k=e + 1)
    # exclure auto-match (index identique quand lib==Mx)
    if lib_size == len(Mx):
        valid = idx != np.arange(len(Mx))[:, None]
        # garder e voisins valides
        new_idx = np.zeros((len(Mx), e), dtype=int)
        new_dists = np.zeros((len(Mx), e))
        for i in range(len(Mx)):
            v = valid[i]
            new_idx[i] = idx[i, v][:e]
            new_dists[i] = dists[i, v][:e]
        idx = new_idx
        dists = new_dists
    else:
        idx = idx[:, :e]
        dists = dists[:, :e]

    # Poids exponentiels Sugihara
    d_min = dists[:, :1] + 1e-12
    w = np.exp(-dists / d_min)
    w /= w.sum(axis=1, keepdims=True)

    y_pred = (w * y_aligned[idx]).sum(axis=1)
    # Coefficient correlation
    a = y_aligned - y_aligned.mean()
    b = y_pred - y_pred.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0))


def causal_summary(series: np.ndarray, lag: int = 2, te_bins: int = 5) -> dict:
    """Pipeline causal complet : Granger + Transfer Entropy + CCM en un appel.

    Si D > 6 dimensions, garde 6 dims avec variance max (PCA-light implicite).
    Retourne 3 matrices D×D + labels pour visualisation graphe causal.

    Permet trianguler causalité :
    - Edge fort en Granger + TE + CCM = causalité robuste
    - Edge en TE seul = relation non-linéaire (Granger rate, CCM rate aussi car non-déterministe)
    - Edge en CCM seul = couplage chaotique déterministe
    """
    n, d = series.shape
    series = series.astype(np.float64)
    # Reduire si trop de dimensions (graphe lisible)
    if d > 6:
        var = series.var(axis=0)
        keep = np.argsort(var)[-6:]
        series = series[:, keep]
        labels = [f"d{int(k)}" for k in keep]
    else:
        labels = [f"d{j}" for j in range(d)]

    granger = granger_pairwise(series, lag=lag)
    te = transfer_entropy_matrix(series, lag=lag, bins=te_bins)

    # CCM matrice (top 4 dims pour cout)
    d_use = min(4, series.shape[1])
    ccm = np.zeros((d_use, d_use))
    for i in range(d_use):
        for j in range(d_use):
            if i == j:
                continue
            ccm[i, j] = ccm_sugihara(series[:, i], series[:, j], e=3, tau=1)

    return {
        "labels": labels,
        "granger": granger.tolist(),
        "transfer_entropy": te.tolist(),
        "ccm": ccm.tolist(),
        "ccm_labels": labels[:d_use],
        "lag": int(lag),
    }
