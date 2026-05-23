"""Video DNA : score composite synthétisant tous les analytics en une métrique.

Construit "scorecard" 7-axes normalisée [0,1] :
- chaos : Lyapunov exponent
- topology : H1 cycles + persistence entropy
- complexity : correlation dimension
- spectral : exposant scaling
- structure : cluster density
- causality : transition mutual info
- predictability : proxy signal-to-noise

Composite = somme pondérée. Classification automatique selon thresholds combinés.
Permet comparer 2 vidéos par un seul nombre.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

from .dynamics import analyse_trajectory
from .spectral import dmd, power_spectrum
from .multiscale import spectral_slope
from .topology import persistent_homology
from .regime_classifier import classify_from_analysis


def _clip01(x: float) -> float:
    """Clip value dans [0, 1] pour normalisation scoring."""
    return max(0.0, min(1.0, float(x)))


def compute_dna(latents: np.ndarray, coords_3d: np.ndarray) -> dict:
    """Pipeline DNA complet : 7 axes + composite + label classification.

    Étapes :
    1. Lyapunov + corr_dim depuis trajectoire 3D
    2. Spectral slope depuis latents
    3. PH H1 count + entropy
    4. Velocity stats (predictability proxy)
    5. HDBSCAN clusters (cluster_density)
    6. Mutual info temporal (causality proxy)
    7. Normalisation chaque axe → [0, 1] via clip
    8. Composite = Σ weight × axis
    9. Classification via règles thresholds combinés
    """
    n = latents.shape[0]

    # 1. Dynamiques (Lyapunov + corr.dim sur coords_3d)
    dyn = analyse_trajectory(coords_3d)
    lyap = abs(dyn["lyapunov"])
    corr_dim = abs(dyn["correlation_dim"])

    # 2. Spectral slope (sur latents pour signal raw)
    slope = abs(spectral_slope(latents))

    # 3. Persistence (H1 count + entropy)
    ph = persistent_homology(coords_3d, max_dim=1, max_n=300)
    h1 = ph["diagrams"][1]["count"] if len(ph["diagrams"]) > 1 else 0
    h1_entropy = ph["persistence_entropy"].get("H1", 0.0)

    # 4. Predictability — utilise difference frame-a-frame comme proxy (sans train MLP ici, eviter cout)
    diffs = np.linalg.norm(latents[1:] - latents[:-1], axis=1)
    velocity_norm = float(np.mean(diffs) / (np.std(diffs) + 1e-9))  # signal/noise ratio
    predictability = _clip01(1.0 / (1.0 + np.std(diffs) / max(np.mean(diffs), 1e-9)))

    # 5. Clusters diversity (HDBSCAN rapide)
    try:
        X = normalize(latents)
        clusterer = HDBSCAN(min_cluster_size=max(2, n // 20), cluster_selection_method="eom")
        labels = clusterer.fit_predict(X)
        n_clusters = len([u for u in set(labels) if u != -1])
        cluster_density = n_clusters / max(math.log(max(n, 2)), 1.0)
    except Exception:
        n_clusters = 0
        cluster_density = 0.0

    # 6. Causality : KSG TE estimator entre dim 0 et dim 1 du coord 3D.
    # Mesure couplage causal SHUFFLED-corrected. Évite saturation MI.
    if coords_3d.shape[1] >= 2 and n >= 30:
        try:
            from .causal_advanced import transfer_entropy_ksg
            # TE empirique
            te_01 = transfer_entropy_ksg(coords_3d[:, 0], coords_3d[:, 1], lag=1, k=4)
            # TE baseline avec shuffled control
            rng = np.random.RandomState(0)
            y_shuf = rng.permutation(coords_3d[:, 1])
            te_shuf = transfer_entropy_ksg(coords_3d[:, 0], y_shuf, lag=1, k=4)
            # Causality = excess TE au-dessus chance (delta normalisé)
            causality = max(0.0, te_01 - te_shuf)
        except Exception:
            causality = 0.0
    else:
        causality = 0.0

    # 7. Régime confidence : verdict heuristique depuis dyn metrics
    # (port phase-space-video). Donne 8e axe DNA + label régime.
    regime_verdict = classify_from_analysis(dyn, embedding_dim=coords_3d.shape[1])

    # Normalisation : ré-calibrée empiriquement via ablation study sur 6 systèmes canoniques.
    # Voir backend/dna_validation.py. axis_variance ~ discrimination power.
    axes = {
        "chaos": _clip01(lyap * 4.0),                    # Lyapunov amplifié
        "topology": _clip01(math.log(h1 + 1) / 5.0),     # log scale h1
        "complexity": _clip01(corr_dim / 3.0),           # dim 3D max ≈ 3
        "spectral": _clip01(slope / 4.0),                # slope -3..-4 typique chaos
        "structure": _clip01(cluster_density / 2.0),
        "causality": _clip01(causality * 1.2),           # KSG TE excess vs shuffle, échelle empirique
        "predictability": _clip01(predictability),
        "regime_confidence": _clip01(regime_verdict.confidence),  # 8e axe : verdict classifier
    }

    # Composite weights : empirical discrimination + thematic coverage.
    # regime_confidence ajouté avec poids 0.10 — interpretation directe régime.
    # Réajustement weights existants pour somme = 1.0.
    weights = {
        "topology": 0.27,
        "causality": 0.18,
        "spectral": 0.16,
        "regime_confidence": 0.10,
        "predictability": 0.09,
        "chaos": 0.09,
        "complexity": 0.06,
        "structure": 0.05,
    }
    composite = sum(axes[k] * weights[k] for k in axes) * 100

    # Classification
    if axes["chaos"] > 0.6 and axes["predictability"] < 0.4:
        label = "highly chaotic · low predictability"
    elif axes["chaos"] > 0.4 and axes["topology"] > 0.5:
        label = "chaotic with periodic structure"
    elif axes["structure"] > 0.5 and axes["predictability"] > 0.6:
        label = "structured · clustered · predictable"
    elif axes["complexity"] < 0.3 and axes["chaos"] < 0.3:
        label = "low complexity · quasi-static"
    elif axes["spectral"] > 0.5:
        label = "scale-invariant · power-law dynamics"
    else:
        label = "intermediate complexity · mixed regime"

    return {
        "composite_score": round(composite, 2),
        "axes": {k: round(v, 4) for k, v in axes.items()},
        "weights": weights,
        "label": label,
        "regime_verdict": regime_verdict.to_dict(),
        "n_clusters": int(n_clusters),
        "raw_metrics": {
            "lyapunov": round(lyap, 4),
            "correlation_dim": round(corr_dim, 4),
            "spectral_slope": round(slope, 4),
            "h1_cycles": int(h1),
            "h1_entropy": round(h1_entropy, 4),
            "n_clusters": int(n_clusters),
            "predictability_proxy": round(predictability, 4),
            "causality_proxy": round(causality, 4),
            "max_diag_ratio": dyn.get("max_diag_ratio", 0.0),
            "convergence_rate": dyn.get("convergence_rate", 0.0),
        },
    }
