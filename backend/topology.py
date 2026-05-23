"""Topologie : persistent homology via Vietoris-Rips (ripser) + persistence entropy."""
from __future__ import annotations
import numpy as np
from ripser import ripser


def persistent_homology(points: np.ndarray, max_dim: int = 1, max_n: int = 400) -> dict:
    """Calcule H0 + H1 (et H2 si max_dim=2) via Vietoris-Rips."""
    n = points.shape[0]
    if n > max_n:
        idx = np.linspace(0, n - 1, max_n).astype(int)
        points = points[idx]

    result = ripser(points, maxdim=max_dim)
    diagrams = result["dgms"]

    out = []
    for dim, dgm in enumerate(diagrams):
        # Filtre points infinis (composantes connexes non-mortes)
        finite = np.isfinite(dgm[:, 1])
        bd = dgm[finite].tolist()
        # Inclus aussi non-finite avec death = max birth (pour affichage)
        if (~finite).any() and len(bd):
            max_b = float(np.max(dgm[finite, 1])) if finite.any() else 1.0
            for bp in dgm[~finite]:
                bd.append([float(bp[0]), float(max_b * 1.1)])
        out.append({
            "dim": int(dim),
            "pairs": bd,  # liste [birth, death]
            "count": int(len(bd)),
        })

    # Persistence entropy par dimension (Rucco et al.)
    entropies = {}
    total_persist = {}
    for d in out:
        if not d["pairs"]:
            entropies[f"H{d['dim']}"] = 0.0
            total_persist[f"H{d['dim']}"] = 0.0
            continue
        L = np.array([p[1] - p[0] for p in d["pairs"]])
        L = L[L > 0]
        if len(L) == 0:
            entropies[f"H{d['dim']}"] = 0.0
            total_persist[f"H{d['dim']}"] = 0.0
            continue
        Lt = L.sum()
        p = L / Lt
        ent = float(-np.sum(p * np.log2(p)))
        entropies[f"H{d['dim']}"] = ent
        total_persist[f"H{d['dim']}"] = float(Lt)

    return {
        "diagrams": out,
        "persistence_entropy": entropies,
        "total_persistence": total_persist,
        "n_points": int(points.shape[0]),
    }


def sliding_window_ph(points: np.ndarray, window: int = 80, stride: int = 40, max_dim: int = 1) -> dict:
    """PH glissante : retourne persistence H1 par fenetre = signature topologique temporelle."""
    n = points.shape[0]
    if n < window + stride:
        # Single window
        full = persistent_homology(points, max_dim=max_dim)
        return {"windows": [{"start": 0, "end": n, **full}], "n_windows": 1}

    windows = []
    for start in range(0, n - window + 1, stride):
        chunk = points[start : start + window]
        ph = persistent_homology(chunk, max_dim=max_dim)
        windows.append({
            "start": int(start),
            "end": int(start + window),
            "h1_count": ph["diagrams"][1]["count"] if max_dim >= 1 and len(ph["diagrams"]) > 1 else 0,
            "h0_count": ph["diagrams"][0]["count"],
            "entropy": ph["persistence_entropy"],
            "total": ph["total_persistence"],
        })
    return {"windows": windows, "n_windows": len(windows), "window_size": window, "stride": stride}
