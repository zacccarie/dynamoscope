"""Verdict classifier : régime dynamique depuis métriques.

Port heuristique de js/metrics.js classify(). Entrée = métriques calculées
par analyse_trajectory (Lyapunov, corrDim, RQA DET, Lmax/N, convergence,
dimension plongement). Sortie = label régime + confiance.

6 régimes :
  fixed     point fixe (système converge)
  noise     stochastique sans structure
  strange   attracteur étrange (chaos déterministe)
  cycle     cycle limite périodique
  torus     quasi-périodique (incommensurable)
  unknown   indéterminé (typiquement régulier mais haute dim)

Non un classifieur ML : règles cascadées sur seuils empiriques.
"""
from __future__ import annotations
from dataclasses import dataclass


# Seuils empiriques — issus de l'outil JS, calibrés sur systèmes connus
TH_NOISE_DIAG = 0.045   # Lmax/N en dessous = aucune diagonale = bruit
TH_REGULAR_DIAG = 0.65  # Lmax/N au-dessus = dynamique régulière (cycle/tore)
TH_CONVERGE_REL = 0.55  # décroissance relative rayon → point fixe


@dataclass
class RegimeVerdict:
    kind: str           # fixed | noise | strange | cycle | torus | unknown
    label: str          # libellé FR
    description: str    # phrase explicative
    confidence: float   # ∈ [0, 1]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "confidence": round(self.confidence, 4),
        }


def classify_regime(
    *,
    embedding_dim: int,
    correlation_dim: float,
    max_diag_ratio: float,
    lyapunov: float,
    convergence: float,
    rqa_det: float | None = None,
) -> RegimeVerdict:
    """Cascading rules sur métriques pour assigner régime.

    Args:
        embedding_dim: dim du plongement utilisé (m).
        correlation_dim: dim de corrélation (Grassberger-Procaccia).
        max_diag_ratio: Lmax / N de matrice récurrence.
        lyapunov: exposant Lyapunov estimé.
        convergence: convergence_rate (cf dynamics.py).
        rqa_det: optionnel, déterminisme RQA.
    Returns:
        RegimeVerdict.
    """
    m = embedding_dim
    d2 = correlation_dim

    # Détection bruit : pas de diagonales OU dim corrélation saturée à m
    is_noise = max_diag_ratio < TH_NOISE_DIAG or d2 >= m - 0.45
    is_regular = max_diag_ratio > TH_REGULAR_DIAG

    if convergence > TH_CONVERGE_REL:
        return RegimeVerdict(
            kind="fixed",
            label="Point fixe",
            description=(
                "Système converge vers état stable — attracteur ponctuel, "
                "rayon trajectoire décroît au fil du temps."
            ),
            confidence=min(1.0, 0.55 + convergence * 0.16),
        )

    if is_noise:
        return RegimeVerdict(
            kind="noise",
            label="Stochastique",
            description=(
                "Aucune structure déterministe — diagonales récurrence absentes, "
                "trajectoire ne revisite pas ses états."
            ),
            confidence=min(1.0, 0.6 + (TH_NOISE_DIAG - max_diag_ratio) * 4),
        )

    if not is_regular:
        return RegimeVerdict(
            kind="strange",
            label="Attracteur étrange",
            description=(
                f"Chaos déterministe borné — trajectoire repliée, sensible aux "
                f"conditions initiales. Dimension fractale D≈{d2:.2f}."
            ),
            confidence=min(1.0, 0.5 + (TH_REGULAR_DIAG - max_diag_ratio) * 1.1),
        )

    if d2 < 1.5:
        return RegimeVerdict(
            kind="cycle",
            label="Cycle limite",
            description=(
                "Oscillation périodique stable — trajectoire suit boucle fermée, "
                "états voisins restent parallèles."
            ),
            confidence=min(1.0, 0.65 + max_diag_ratio * 0.3),
        )

    if d2 < 2.6:
        return RegimeVerdict(
            kind="torus",
            label="Quasi-périodique",
            description=(
                f"Trajectoire régulière non chaotique D≈{d2:.2f} — plusieurs "
                f"fréquences incommensurables, mouvement dense sur tore."
            ),
            confidence=min(1.0, 0.6 + max_diag_ratio * 0.3),
        )

    return RegimeVerdict(
        kind="unknown",
        label="Indéterminé",
        description=(
            "Dynamique régulière mais haute dimension — augmenter m ou ajuster τ "
            "pour mieux déplier structure."
        ),
        confidence=0.4,
    )


def classify_from_analysis(
    analysis: dict,
    embedding_dim: int | None = None,
) -> RegimeVerdict:
    """Helper : wrap autour de analyse_trajectory dict.

    Args:
        analysis: dict from dynamics.analyse_trajectory.
        embedding_dim: m. Si None, utilise len(coords.shape[1]) implicite via 3 default.
    """
    return classify_regime(
        embedding_dim=embedding_dim if embedding_dim is not None else 3,
        correlation_dim=analysis.get("correlation_dim", 0.0),
        max_diag_ratio=analysis.get("max_diag_ratio", 0.0),
        lyapunov=analysis.get("lyapunov", 0.0),
        convergence=analysis.get("convergence_rate", 0.0),
        rqa_det=analysis.get("rqa", {}).get("DET", 0.0),
    )
