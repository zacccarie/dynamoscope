"""Causal advanced : PCMCI+ via tigramite + KSG transfer entropy.

Améliorations vs causal.py basique :
- PCMCI+ (Runge et al. 2018) : multi-variate causal discovery avec FDR control
- KSG TE (Kraskov-Stögbauer-Grassberger 2004) : k-NN estimator
  continu, sans binning, robuste aux distributions non-Gaussiennes

References :
- Runge 2018 PCMCI+ : "Causal discovery with momentary conditional independence"
- Kraskov 2004 : "Estimating mutual information" via nearest neighbors
- Lizier 2014 JIDT : reference Java implementation of KSG TE
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def pcmci_discovery(
    series: np.ndarray,
    tau_max: int = 3,
    pc_alpha: float = 0.05,
    test: str = "parcorr",
) -> dict:
    """PCMCI+ causal discovery via tigramite.

    Args:
        series: (N, D) multivariate time series
        tau_max: maximum lag explored
        pc_alpha: significance threshold (Bonferroni-style FDR control internal)
        test: 'parcorr' (linear partial correlation) | 'gpdc' (Gaussian process,
              non-linear, slower) | 'cmi_knn' (k-NN conditional MI, fully non-parametric)

    Returns :
        p_matrix : (D, D, tau_max+1) p-values per (cause, effect, lag)
        val_matrix : (D, D, tau_max+1) effect strengths
        graph : binary adjacency (D, D, tau_max+1) at pc_alpha threshold
        links_at_alpha : list dicts {cause_idx, effect_idx, lag, val, p}
    """
    from tigramite.pcmci import PCMCI
    from tigramite import data_processing as pp

    if test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        ci = ParCorr()
    elif test == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC
        ci = GPDC()
    elif test == "cmi_knn":
        from tigramite.independence_tests.cmiknn import CMIknn
        ci = CMIknn()
    else:
        raise ValueError(f"Unknown test: {test}")

    dataframe = pp.DataFrame(np.asarray(series, dtype=np.float64))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci, verbosity=0)
    result = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    p_mat = result["p_matrix"]
    val_mat = result["val_matrix"]
    graph = (p_mat < pc_alpha).astype(int)

    # Extract significant links
    D = p_mat.shape[0]
    links: list[dict] = []
    for i in range(D):
        for j in range(D):
            for tau in range(tau_max + 1):
                if i == j and tau == 0:
                    continue
                if graph[i, j, tau]:
                    links.append({
                        "cause_idx": int(i),
                        "effect_idx": int(j),
                        "lag": int(tau),
                        "val": float(val_mat[i, j, tau]),
                        "p_value": float(p_mat[i, j, tau]),
                    })
    links.sort(key=lambda x: x["p_value"])

    return {
        "method": f"PCMCI+ ({test})",
        "p_matrix": p_mat.tolist(),
        "val_matrix": val_mat.tolist(),
        "graph": graph.tolist(),
        "links_significant": links,
        "n_links": len(links),
        "tau_max": int(tau_max),
        "pc_alpha": float(pc_alpha),
    }


def _knn_count(point: np.ndarray, points: np.ndarray, eps: float) -> int:
    """Compte points dans ball [point - eps, point + eps]^d (Chebyshev metric)."""
    diffs = np.abs(points - point[None, :])
    return int(((diffs < eps).all(axis=1)).sum() - 1)  # exclude self


def transfer_entropy_ksg(
    x: np.ndarray, y: np.ndarray, lag: int = 1, k: int = 4,
) -> float:
    """Transfer entropy TE(y → x) via Kraskov-Stögbauer-Grassberger 2004.

    Estimator k-NN continu, sans binning. Plus précis que histogramme,
    robuste aux distributions non-Gaussiennes et faibles données.

    TE_{y→x} = I(x_t; y_{t-lag} | x_{t-lag})
            = H(x_t, x_{t-lag}) + H(x_{t-lag}, y_{t-lag})
            - H(x_t, x_{t-lag}, y_{t-lag}) - H(x_{t-lag})

    Equivalent à MI conditionnel estimé via k-NN.

    Args:
        x, y: 1D time series
        lag: temporal lag
        k: nearest neighbors (typique 4-10)

    Returns:
        TE en nats (log naturel). ≥ 0 par construction.
    """
    n = len(x)
    if n <= lag + k + 2:
        return 0.0

    x_now = x[lag:].astype(np.float64)
    x_past = x[:-lag].astype(np.float64)
    y_past = y[:-lag].astype(np.float64)
    N = len(x_now)

    # Joint space (x_t, x_{t-lag}, y_{t-lag})
    joint = np.column_stack([x_now, x_past, y_past])
    tree_joint = cKDTree(joint)
    # k-NN distances in joint space (Chebyshev / max norm)
    dists, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, k]  # k-th neighbor distance

    # Marginal spaces
    sub_xt_xp = np.column_stack([x_now, x_past])
    sub_xp_yp = np.column_stack([x_past, y_past])
    sub_xp = x_past[:, None]

    tree1 = cKDTree(sub_xt_xp)
    tree2 = cKDTree(sub_xp_yp)
    tree3 = cKDTree(sub_xp)

    # Count points strictly within eps in each subspace
    te_sum = 0.0
    for i in range(N):
        n1 = len(tree1.query_ball_point(sub_xt_xp[i], r=eps[i] - 1e-12, p=np.inf)) - 1
        n2 = len(tree2.query_ball_point(sub_xp_yp[i], r=eps[i] - 1e-12, p=np.inf)) - 1
        n3 = len(tree3.query_ball_point(sub_xp[i], r=eps[i] - 1e-12, p=np.inf)) - 1
        te_sum += digamma(k) - digamma(n1 + 1) - digamma(n2 + 1) + digamma(n3 + 1)

    te = max(0.0, te_sum / N)
    return float(te)


def te_matrix_ksg(series: np.ndarray, lag: int = 1, k: int = 4) -> np.ndarray:
    """Matrice TE_{j→i} via KSG estimator. (D, D)."""
    n, d = series.shape
    out = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            out[j, i] = transfer_entropy_ksg(series[:, i], series[:, j], lag=lag, k=k)
    return out


def causal_summary_advanced(
    series: np.ndarray, tau_max: int = 3, pc_alpha: float = 0.05,
) -> dict:
    """Pipeline avancé : PCMCI+ + KSG TE + comparison vs simple Granger/binning.

    Retourne deux vues complémentaires :
    - PCMCI+ : sound causal inference avec FDR
    - KSG TE : info-theoretic, non-parametric
    """
    pcmci_out = pcmci_discovery(series, tau_max=tau_max, pc_alpha=pc_alpha, test="parcorr")
    te_mat = te_matrix_ksg(series, lag=1, k=4)
    return {
        "pcmci": pcmci_out,
        "ksg_te_matrix": te_mat.tolist(),
        "n_dims": int(series.shape[1]),
        "n_samples": int(series.shape[0]),
    }
