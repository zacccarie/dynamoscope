"""DNA composite score validation : ablation study + discrimination benchmark.

Mesure :
1. Leave-one-axis-out : impact suppression de chaque axe sur discrimination
2. Single-axis baselines : chaque axe seul vs composite
3. Cross-system discrimination : DNA différencie-t-il les 6 systèmes canoniques ?
4. Permutation test : composite mieux que random weights ?

Output : validation rigoureuse des poids actuels OU recommandation
de poids alternatifs basés sur discrimination empirique.
"""
from __future__ import annotations
import itertools
import numpy as np
from .dna import compute_dna
from .systems import SYSTEMS


def dna_ablation_study(seed: int = 42) -> dict:
    """Computes DNA pour tous les 6 systèmes canoniques.

    Pour chaque système, compute axes 7D. Mesure discrimination via :
    - cross-system axes distances
    - leave-one-axis-out variance loss
    - single-axis discrimination power

    Returns dict avec scoring per-axis + recommandation poids optimaux.
    """
    rng = np.random.RandomState(seed)
    results = {}

    # Compute DNA pour chaque système
    for sid, fn in SYSTEMS.items():
        out = fn()
        coords = np.array(out["coords"])
        raw = np.array(out["raw_coords"])
        dna = compute_dna(raw, coords)
        results[sid] = dna

    # Build axes matrix : (n_systems, n_axes)
    axes_keys = list(results[next(iter(results))]["axes"].keys())
    n_axes = len(axes_keys)
    n_sys = len(results)
    A = np.zeros((n_sys, n_axes))
    sys_ids = list(results.keys())
    for i, sid in enumerate(sys_ids):
        for j, k in enumerate(axes_keys):
            A[i, j] = results[sid]["axes"][k]

    # Discrimination power per axis : variance across systems
    axis_variance = A.var(axis=0)
    discrim_score_per_axis = axis_variance / axis_variance.sum()

    # Pairwise system distances using all axes (full composite)
    weights_default = np.array([0.18, 0.15, 0.18, 0.10, 0.15, 0.12, 0.12])
    composite_full = (A * weights_default).sum(axis=1)

    # Leave-one-axis-out : recompute composite without axis j, measure
    # discrimination = inter-system std
    looo_discrim = {}
    full_std = float(composite_full.std())
    for j, k in enumerate(axes_keys):
        mask = np.ones(n_axes, dtype=bool)
        mask[j] = False
        # Renormalize weights
        w_red = weights_default[mask] / weights_default[mask].sum()
        composite_red = (A[:, mask] * w_red).sum(axis=1)
        looo_discrim[k] = {
            "std_without_axis": float(composite_red.std()),
            "discrim_loss_pct": float((full_std - composite_red.std()) / full_std * 100),
        }

    # Sort axes by loss-when-removed : higher loss = more critical
    critical_order = sorted(looo_discrim.items(),
                             key=lambda x: -x[1]["discrim_loss_pct"])

    # Permutation test : random weights distribution
    n_trials = 200
    random_stds = []
    for _ in range(n_trials):
        w_random = rng.dirichlet(np.ones(n_axes))
        composite_r = (A * w_random).sum(axis=1)
        random_stds.append(composite_r.std())
    random_stds = np.array(random_stds)
    percentile = float((random_stds < full_std).mean() * 100)

    return {
        "systems_tested": sys_ids,
        "axes_keys": axes_keys,
        "axes_matrix": A.tolist(),
        "axis_variance_across_systems": axis_variance.tolist(),
        "discrim_score_per_axis": {
            k: float(v) for k, v in zip(axes_keys, discrim_score_per_axis)
        },
        "leave_one_out": looo_discrim,
        "critical_order": [{"axis": k, **v} for k, v in critical_order],
        "default_weights": weights_default.tolist(),
        "full_composite_std": full_std,
        "random_weights_baseline": {
            "n_trials": n_trials,
            "mean_std": float(random_stds.mean()),
            "max_std": float(random_stds.max()),
            "default_weights_percentile": percentile,
            "interpretation": (
                f"Default weights better than {percentile:.1f}% of random weight assignments. "
                f"{'Strong evidence weights are optimized.' if percentile > 70 else 'Weights not particularly optimal vs random — review weighting.'}"
            ),
        },
    }


def optimize_dna_weights(n_iter: int = 1000, seed: int = 42) -> dict:
    """Find DNA weights maximizing inter-system discrimination via random search.

    Returns weights that maximize std of composite across 6 canonical systems.
    """
    rng = np.random.RandomState(seed)
    # Build axes matrix once
    A = []
    sys_ids = []
    for sid, fn in SYSTEMS.items():
        out = fn()
        dna = compute_dna(np.array(out["raw_coords"]), np.array(out["coords"]))
        A.append(list(dna["axes"].values()))
        sys_ids.append(sid)
    A = np.array(A)
    n_axes = A.shape[1]
    axes_keys = list(SYSTEMS["lorenz"]() and compute_dna(
        np.array(SYSTEMS["lorenz"]()["raw_coords"]),
        np.array(SYSTEMS["lorenz"]()["coords"])
    )["axes"].keys())

    best_std = 0.0
    best_w = np.ones(n_axes) / n_axes
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n_axes))
        std = (A * w).sum(axis=1).std()
        if std > best_std:
            best_std = float(std)
            best_w = w

    return {
        "optimal_weights": best_w.tolist(),
        "optimal_weights_named": {k: float(v) for k, v in zip(axes_keys, best_w)},
        "best_std_achieved": best_std,
        "n_iterations": n_iter,
    }
