"""Vineyards : continuous birth-death tracking through time-varying PH.

Cohen-Steiner et al. 2006 : "Vines and Vineyards by Updating Persistence
in Linear Time". Tracks persistence pairs across continuously deforming
filtrations, producing 'threads' (vines) that birth, persist, and die.

Notre simplified version :
1. Compute PH at each sliding window
2. Match persistence pairs between consecutive windows via bipartite
   matching (closest birth-death pair in (birth, death) plane)
3. Threads = sequences of matched pairs across windows

Output : per-thread time series of (birth, death) coordinates.
Visualizes which topological features persist, when new loops appear,
when existing loops die.

References :
- Cohen-Steiner-Edelsbrunner-Morozov 2006 (original vineyards)
- Munch 2017 (overview vineyards in TDA)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from ripser import ripser


def _ph_window(points: np.ndarray, max_dim: int = 1) -> list[list[tuple[float, float]]]:
    """PH on window, return list of (birth, death) pairs per dim."""
    if points.shape[0] < 3:
        return [[] for _ in range(max_dim + 1)]
    result = ripser(points, maxdim=max_dim)
    diagrams = result["dgms"]
    out: list[list[tuple[float, float]]] = []
    for dgm in diagrams:
        finite = np.isfinite(dgm[:, 1])
        # Include essential class avec death = max+padding
        finite_pairs = [(float(b), float(d)) for b, d in dgm[finite]]
        if (~finite).any() and finite.any():
            max_b = float(np.max(dgm[finite, 1]))
            for bp in dgm[~finite]:
                finite_pairs.append((float(bp[0]), max_b * 1.1))
        out.append(finite_pairs)
    return out


def _match_pairs(prev: list[tuple[float, float]],
                 curr: list[tuple[float, float]],
                 max_dist: float = 0.5) -> list[tuple[int, int]]:
    """Bipartite matching closest pairs via Hungarian algorithm.

    Returns list (prev_idx, curr_idx) for matched pairs.
    Unmatched = vines that die or birth at this transition.
    """
    if not prev or not curr:
        return []
    P = np.array(prev)  # (n_prev, 2)
    C = np.array(curr)  # (n_curr, 2)
    # Cost matrix : L2 distance dans (birth, death) space
    diff = P[:, None, :] - C[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=-1))
    # Hungarian on rectangular matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matches.append((int(r), int(c)))
    return matches


def compute_vineyards(
    points: np.ndarray, window: int = 60, stride: int = 15, max_dim: int = 1,
) -> dict:
    """Compute vineyards : matched persistence pairs through sliding windows.

    Args:
        points: trajectoire (N, d)
        window: taille window pour PH local
        stride: pas entre windows successives
        max_dim: dimension homologique max (0 ou 1)

    Returns:
        - windows : metadata par window (start, end, center)
        - dim_threads : dict {dim: list of threads}, chaque thread = list of
          {window_idx, birth, death, persistence}
        - n_threads_per_dim : total threads par dim
        - births_count : new vines birthed per window
        - deaths_count : vines that died per window
    """
    n = points.shape[0]
    if n < window + stride:
        # Single window fallback
        diags = _ph_window(points, max_dim=max_dim)
        return {
            "windows": [{"start": 0, "end": n, "center": n // 2}],
            "dim_threads": {
                str(dim): [[{
                    "window_idx": 0, "birth": float(b), "death": float(d),
                    "persistence": float(d - b),
                }] for b, d in pairs]
                for dim, pairs in enumerate(diags)
            },
            "n_threads_per_dim": {str(dim): len(pairs) for dim, pairs in enumerate(diags)},
            "births_per_window": [],
            "deaths_per_window": [],
            "n_windows": 1,
        }

    # Collect PH per window
    windows_meta: list[dict] = []
    all_diagrams: list[list[list[tuple[float, float]]]] = []  # [window][dim][pair]
    for start in range(0, n - window + 1, stride):
        chunk = points[start : start + window]
        diags = _ph_window(chunk, max_dim=max_dim)
        all_diagrams.append(diags)
        windows_meta.append({
            "start": int(start),
            "end": int(start + window),
            "center": int(start + window // 2),
            "n_pairs": [len(d) for d in diags],
        })
    n_windows = len(all_diagrams)

    # Match across consecutive windows per dim
    dim_threads: dict[str, list[list[dict]]] = {}
    births_count: list[int] = [0] * n_windows
    deaths_count: list[int] = [0] * n_windows

    for dim in range(max_dim + 1):
        # active_threads[idx_in_prev] = thread_id
        threads: list[list[dict]] = []
        # Initialize : each pair in window 0 starts a thread
        active_map: dict[int, int] = {}
        for pair_idx, (b, d) in enumerate(all_diagrams[0][dim]):
            tid = len(threads)
            threads.append([{
                "window_idx": 0, "birth": b, "death": d, "persistence": d - b,
            }])
            active_map[pair_idx] = tid
        births_count[0] += len(active_map)

        # Process subsequent windows
        for w in range(1, n_windows):
            prev = all_diagrams[w - 1][dim]
            curr = all_diagrams[w][dim]
            matches = _match_pairs(prev, curr)

            new_active: dict[int, int] = {}
            matched_curr: set[int] = set()
            matched_prev: set[int] = set()
            for prev_i, curr_i in matches:
                matched_prev.add(prev_i)
                matched_curr.add(curr_i)
                tid = active_map.get(prev_i)
                if tid is None:
                    continue
                b, d = curr[curr_i]
                threads[tid].append({
                    "window_idx": w, "birth": b, "death": d, "persistence": d - b,
                })
                new_active[curr_i] = tid

            # Unmatched in prev = died
            deaths_count[w] += len(prev) - len(matched_prev)
            # Unmatched in curr = new birth
            for ci, (b, d) in enumerate(curr):
                if ci not in matched_curr:
                    tid = len(threads)
                    threads.append([{
                        "window_idx": w, "birth": b, "death": d, "persistence": d - b,
                    }])
                    new_active[ci] = tid
                    births_count[w] += 1

            active_map = new_active

        dim_threads[str(dim)] = threads

    return {
        "windows": windows_meta,
        "dim_threads": dim_threads,
        "n_threads_per_dim": {k: len(v) for k, v in dim_threads.items()},
        "births_per_window": births_count,
        "deaths_per_window": deaths_count,
        "n_windows": n_windows,
        "window_size": window,
        "stride": stride,
    }
