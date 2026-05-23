"""Auto-tune des params d'embedding via random search.
Objectifs : continuity (temporel), faithfulness (k-NN preservation), spread (variance), composite."""
from __future__ import annotations
import random
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .reducer import reduce_3d


METRICS = ["cosine", "euclidean", "manhattan", "correlation"]


def temporal_continuity_score(coords: np.ndarray) -> float:
    """Petit -> bon. Mean ‖z[t+1] − z[t]‖."""
    diffs = np.diff(coords, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def knn_faithfulness(latents: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """[0..1]. Mesure overlap moyen des k-NN entre espace original et projection.
    1 = topologie locale parfaitement preservee."""
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
    """Variance moyenne par axe — recompense distribution etalee. Normalise [0..1]."""
    var = float(np.mean(np.var(coords, axis=0)))
    return min(1.0, var / 0.34)  # var max ≈ 1/3 pour distribution uniform [-1,1]


def composite_score(latents: np.ndarray, coords: np.ndarray, weights: dict | None = None) -> dict:
    """Score composite + decomposition."""
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
    """Bayesian optimization via skopt gp_minimize. Convergence + rapide que random."""
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
    """Exploration random sur params UMAP.
    objective : 'composite', 'faithfulness', 'continuity', 'spread'."""
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
