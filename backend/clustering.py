"""Découverte clusters non-supervisée via HDBSCAN sur latents originaux.
Plus transition matrix entre clusters = narrative structure."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize


def cluster_latents(
    latents: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    normalize_vectors: bool = True,
) -> dict:
    """HDBSCAN sur latents. Retourne labels + stats par cluster.
    label = -1 -> bruit / non-classe."""
    n = latents.shape[0]
    X = normalize(latents) if normalize_vectors else latents.astype(np.float64)

    clusterer = HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    unique = sorted(set(labels.tolist()))
    n_noise = int(np.sum(labels == -1))
    n_clusters = len([u for u in unique if u != -1])

    clusters: list[dict] = []
    for c in unique:
        if c == -1:
            continue
        mask = labels == c
        idx = np.where(mask)[0].tolist()
        # Stats intra-cluster
        cluster_vecs = X[mask]
        centroid = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        # Span temporel : min/max indices
        time_span = (int(min(idx)), int(max(idx)))
        clusters.append({
            "id": int(c),
            "size": int(mask.sum()),
            "indices": idx,
            "time_span": time_span,
            "intra_mean_dist": float(dists.mean()),
            "intra_std_dist": float(dists.std()),
            "compactness": float(1.0 / (1.0 + dists.mean())),  # 1 = parfait, 0 = etale
        })

    # Tri par taille
    clusters.sort(key=lambda c: -c["size"])

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "labels": labels.tolist(),
        "clusters": clusters,
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
        },
    }


def transition_matrix(labels: list[int], include_noise: bool = False) -> dict:
    """Compte transitions cluster[t] -> cluster[t+1].
    Retourne matrice + temps total par cluster + sequence."""
    arr = np.asarray(labels)
    if not include_noise:
        # Pour transitions, remplace -1 par cluster precedent (carry-forward)
        mask = arr == -1
        if mask.any():
            valid_indices = np.where(~mask)[0]
            arr = arr.copy()
            for i in range(len(arr)):
                if arr[i] == -1:
                    # cherche prev valid
                    prev = valid_indices[valid_indices < i]
                    if len(prev) > 0:
                        arr[i] = arr[prev[-1]]
    unique = sorted(set(int(x) for x in arr if x != -1))
    n_c = len(unique)
    if n_c == 0:
        return {"matrix": [], "labels": [], "time_per_cluster": [], "sequence": []}

    idx_map = {c: i for i, c in enumerate(unique)}
    M = np.zeros((n_c, n_c), dtype=np.int32)
    sequence: list[int] = []
    for t in range(len(arr) - 1):
        a, b = int(arr[t]), int(arr[t+1])
        if a not in idx_map or b not in idx_map:
            continue
        M[idx_map[a], idx_map[b]] += 1
        if not sequence or sequence[-1] != idx_map[a]:
            sequence.append(idx_map[a])
    # Time per cluster
    time_per = np.bincount(
        [idx_map[int(x)] for x in arr if int(x) in idx_map], minlength=n_c
    ).tolist()
    return {
        "matrix": M.tolist(),
        "cluster_ids": [int(c) for c in unique],
        "time_per_cluster": [int(t) for t in time_per],
        "sequence_unique": sequence,
        "total_transitions": int(M.sum()),
    }
