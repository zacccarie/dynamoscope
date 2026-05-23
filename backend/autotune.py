"""Auto-tune params UMAP via optimisation (random search ou Bayesian GP-EI).

Problème : params UMAP (n_neighbors, min_dist, metric, smoothing) influencent fortement
visualisation. Manuellement fastidieux à explorer. Auto-tune trouve config qui
maximise un objectif composite mesurant "qualité" embedding.

Objectifs disponibles :
- faithfulness : k-NN preservation (original 2048d vs projection 3D)
- continuity : trajectoire temporelle lisse
- spread : utilise tout l'espace de visualisation
- composite : combinaison pondérée

Algos :
- random_search : exploration uniforme (surprisingly competitive, Bergstra 2012)
- bayesian_search : Gaussian Process + Expected Improvement (Snoek 2012)
"""
from __future__ import annotations
import random
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .reducer import reduce_3d


METRICS = ["cosine", "euclidean", "manhattan", "correlation"]


def temporal_continuity_score(coords: np.ndarray) -> float:
    """Mesure rugosité trajectoire = moyenne ‖coord[t+1] - coord[t]‖.

    Petit = trajectoire lisse, frames adjacentes proches en 3D.
    Grand = jumps artefactuels UMAP.
    Pour scoring, on inverse plus tard (1 - normalised).
    """
    diffs = np.diff(coords, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def knn_faithfulness(latents: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """Trustworthiness score : overlap k-nearest-neighbors original vs projection.

    Pour chaque frame i, compare ses k voisins dans latent 2048d (cosine)
    vs ses k voisins dans projection 3D (euclidean). Overlap = qualité préservation
    locale.

    1.0 = topologie parfaitement préservée.
    0.0 = projection complètement déconnectée du structure original.
    """
    n = latents.shape[0]
    if n <= k + 1:
        return 0.0
    nn_orig = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(latents)
    nn_proj = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords)
    _, idx_o = nn_orig.kneighbors(latents)
    _, idx_p = nn_proj.kneighbors(coords)
    # exclude self (index 0)
    overlaps = []
    for i in range(n):
        a = set(idx_o[i, 1:])
        b = set(idx_p[i, 1:])
        overlaps.append(len(a & b) / k)
    return float(np.mean(overlaps))


def spread_score(coords: np.ndarray) -> float:
    """Mesure étalement points dans cube [-1,1]³. Variance par axe moyennée.

    Évite UMAP qui colle tout dans un coin (utilise mal espace disponible).
    Normalisé : 1.0 = distribution uniforme dans cube, 0.0 = points concentrés.
    """
    var = float(np.mean(np.var(coords, axis=0)))
    return min(1.0, var / 0.34)  # var max ≈ 1/3 pour distribution uniform [-1,1]


def composite_score(latents: np.ndarray, coords: np.ndarray, weights: dict | None = None) -> dict:
    """Score composite pondéré : 0.5·faithfulness + 0.3·continuity + 0.2·spread.

    Compromis qualitatif : préserve voisinages (faith) > lisse (cont) > use space (spread).
    Returns dict avec decomposition pour debug + total in [0, 1].
    """
    if weights is None:
        weights = {"faithfulness": 0.5, "continuity": 0.3, "spread": 0.2}
    cont = temporal_continuity_score(coords)
    cont_norm = max(0.0, 1.0 - cont / 2.0)  # continuity normalise (2.0 = pire cas, 0 = parfait)
    faith = knn_faithfulness(latents, coords)
    spread = spread_score(coords)
    total = (
        weights["faithfulness"] * faith
        + weights["continuity"] * cont_norm
        + weights["spread"] * spread
    )
    return {
        "total": float(total),
        "faithfulness": float(faith),
        "continuity_raw": float(cont),
        "continuity_norm": float(cont_norm),
        "spread": float(spread),
    }


def bayesian_search(
    latents: np.ndarray,
    n_trials: int = 20,
    objective: str = "composite",
    n_initial: int = 5,
) -> dict:
    """Bayesian Optimization via skopt gp_minimize avec Expected Improvement.

    Algorithme :
    1. Phase initiale : n_initial trials aléatoires pour bootstrap GP
    2. Phase BO : fit Gaussian Process sur scores observés
    3. Choisit prochain trial via EI = P(amélioration) × E(amélioration | améliore)
    4. Itère

    Plus efficient que random quand surface objective est lisse + n_trials suffisant
    (~25+). Sur petits budgets, random parfois gagne par chance.

    GP modeling overhead ~50-100ms par trial vs 0 pour random.
    """
    from skopt import gp_minimize
    from skopt.space import Integer, Real, Categorical

    n = latents.shape[0]
    nn_max = min(50, max(5, n // 4))
    space = [
        Integer(5, nn_max, name="n_neighbors"),
        Real(0.0, 0.5, name="min_dist"),
        Categorical(METRICS, name="metric"),
        Integer(0, 8, name="smoothing"),
    ]

    history: list[dict] = []

    def neg_objective(params):
        n_neighbors, min_dist, metric, smoothing = params
        try:
            coords = reduce_3d(
                latents, method="umap",
                n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                metric=metric, smoothing=int(smoothing),
            )
        except Exception:
            return 0.0
        scores = composite_score(latents, coords)
        if objective == "continuity":
            obj_val = scores["continuity_norm"]
        elif objective == "faithfulness":
            obj_val = scores["faithfulness"]
        elif objective == "spread":
            obj_val = scores["spread"]
        else:
            obj_val = scores["total"]
        history.append({
            "trial": len(history),
            "params": {
                "n_neighbors": int(n_neighbors), "min_dist": float(min_dist),
                "metric": metric, "smoothing": int(smoothing),
            },
            "scores": scores,
            "objective": float(obj_val),
        })
        return -obj_val  # gp_minimize minimise → on inverse

    t_start = time.time()
    result = gp_minimize(
        neg_objective, space, n_calls=n_trials,
        n_initial_points=max(3, min(n_initial, n_trials // 2)),
        random_state=42, acq_func="EI",
    )
    elapsed = time.time() - t_start

    # Best
    best_idx = int(np.argmax([h["objective"] for h in history]))
    best = history[best_idx]
    return {
        "best": best,
        "history": history,
        "n_trials": len(history),
        "objective_name": objective,
        "compute_s": round(elapsed, 2),
        "method": "bayesian_gp",
    }


def random_search(
    latents: np.ndarray,
    n_trials: int = 20,
    objective: str = "composite",
    seed: int = 42,
) -> dict:
    """Random uniform search sur params UMAP — baseline simple.

    Surprenamment compétitif (Bergstra-Bengio 2012) sur petits budgets
    car distributions params souvent multimodales.

    Échantillonne uniformément :
    - n_neighbors ∈ [5, n//4]
    - min_dist ∈ [0, 0.5]
    - metric ∈ {cosine, euclidean, manhattan, correlation}
    - smoothing ∈ [0, 8]

    Returns best params + history complète pour analyse.
    """
    rng = random.Random(seed)
    n = latents.shape[0]
    nn_max = min(50, max(5, n // 4))

    history: list[dict] = []
    best = None

    t_start = time.time()
    for trial in range(n_trials):
        params = {
            "method": "umap",
            "n_neighbors": rng.randint(5, nn_max),
            "min_dist": round(rng.uniform(0.0, 0.5), 3),
            "metric": rng.choice(METRICS),
            "smoothing": rng.randint(0, 8),
        }
        try:
            coords = reduce_3d(
                latents,
                method=params["method"],
                n_neighbors=params["n_neighbors"],
                min_dist=params["min_dist"],
                metric=params["metric"],
                smoothing=params["smoothing"],
            )
        except Exception as e:
            continue
        scores = composite_score(latents, coords)
        # Selectionne score par objectif
        if objective == "continuity":
            obj_value = scores["continuity_norm"]
        elif objective == "faithfulness":
            obj_value = scores["faithfulness"]
        elif objective == "spread":
            obj_value = scores["spread"]
        else:
            obj_value = scores["total"]

        entry = {"trial": trial, "params": params, "scores": scores, "objective": float(obj_value)}
        history.append(entry)
        if best is None or obj_value > best["objective"]:
            best = entry

    elapsed = time.time() - t_start
    return {
        "best": best,
        "history": history,
        "n_trials": len(history),
        "objective_name": objective,
        "compute_s": round(elapsed, 2),
    }
