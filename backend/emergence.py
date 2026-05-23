"""Émergence causale quantifiable (Hoel-Albantakis-Tononi 2013).

Question : un système peut-il être plus déterministe à l'échelle macro qu'à la micro ?
Si oui = causal emergence (Hoel) = macro-pattern a effective info supérieure.

Méthode :
1. Effective Information EI = MI(state[t], state[t+1]) (mutual info temporal)
2. Coarse-grain states (group consecutive frames)
3. Compute EI à différentes échelles
4. φ_id ≈ gain EI(macro) - EI(micro)

φ_id > 0 → émergence causale présente.
"""
from __future__ import annotations
import numpy as np


def _mutual_info_bins(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """Mutual Information I(X;Y) = Σ p(x,y) log[p(x,y) / (p(x)p(y))].

    Mesure dépendance statistique X et Y (0 = indépendants, max = identiques).
    Estimé via histogramme 2D : binning continu → discret puis formule classique.
    """
    if len(x) != len(y) or len(x) < 4:
        return 0.0
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / max(hist.sum(), 1)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    pxpy = px * py
    nz = (pxy > 0) & (pxpy > 0)
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / pxpy[nz])))


def transition_mi(X: np.ndarray, bins: int = 16) -> float:
    """Effective Information EI(t→t+1) moyennée sur dimensions du state.

    EI mesure prédictibilité de l'état suivant depuis état actuel.
    Élevé = système déterministe / structuré.
    Bas = aléatoire / chaotique.
    """
    n, d = X.shape
    if n < 8:
        return 0.0
    total = 0.0
    for k in range(d):
        total += _mutual_info_bins(X[:-1, k], X[1:, k], bins=bins)
    return total / d


def coarse_grain_states(X: np.ndarray, factor: int) -> np.ndarray:
    """Construit macro-états en groupant `factor` micro-états consécutifs.

    Macro_state[i] = moyenne(micro_states[i·factor : (i+1)·factor]).
    Diminue résolution temporelle, augmente potentiellement déterminisme.
    """
    n, d = X.shape
    new_n = n // factor
    if new_n < 4:
        return X[:0]
    trimmed = X[: new_n * factor]
    return trimmed.reshape(new_n, factor, d).mean(axis=1)


def effective_information_ladder(
    X: np.ndarray, scales: list[int] | None = None, bins: int = 14
) -> dict:
    """Compute EI(s) pour échelles s = [1, 2, 4, 8, 16, 32] via coarse-graining.

    φ_id ≈ max(EI(s>1) - EI(1)) > 0 → émergence (macro plus déterministe).
    Optimal_scale = échelle où EI maximum.

    Pour Lorenz : φ_id ≈ 0 (système déjà déterministe au micro).
    Pour systèmes stochastiques avec macro-patterns : φ_id > 0.
    """
    if scales is None:
        scales = [1, 2, 4, 8, 16]
    ladder = []
    ei_micro = transition_mi(X, bins=bins)
    for s in scales:
        if s == 1:
            ei = ei_micro
            n_states = X.shape[0]
        else:
            Xc = coarse_grain_states(X, s)
            if Xc.shape[0] < 8:
                break
            ei = transition_mi(Xc, bins=bins)
            n_states = Xc.shape[0]
        gain = ei - ei_micro
        ladder.append({
            "scale": int(s),
            "n_states": int(n_states),
            "ei": float(ei),
            "gain": float(gain),
        })

    # Phi_id approx : meilleur gain positif
    gains = [r["gain"] for r in ladder]
    best_idx = int(np.argmax(gains)) if gains else 0
    best = ladder[best_idx] if ladder else {"scale": 1, "gain": 0.0, "ei": 0.0}
    causal_emergence = float(max(0.0, best["gain"]))

    return {
        "ladder": ladder,
        "ei_micro": float(ei_micro),
        "phi_id_approx": causal_emergence,
        "optimal_scale": int(best["scale"]),
        "interpretation": (
            "macro states are more predictable than micro"
            if causal_emergence > 0.02
            else "no significant emergence detected"
        ),
    }
