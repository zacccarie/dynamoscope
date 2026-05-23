"""Analyse dynamique : Lyapunov, recurrence plot, RQA, Takens embedding."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform


def mutual_info_lag(series: np.ndarray, max_lag: int = 50, bins: int = 16) -> int:
    """Premier minimum de mutual info -> lag optimal Takens. Heuristique simple via histogramme."""
    n = len(series)
    mi = np.zeros(max_lag + 1)
    series = (series - series.min()) / max(series.ptp(), 1e-9)
    for lag in range(1, max_lag + 1):
        x = series[:-lag]
        y = series[lag:]
        hx, _ = np.histogram(x, bins=bins, density=False)
        hy, _ = np.histogram(y, bins=bins, density=False)
        hxy, _, _ = np.histogram2d(x, y, bins=bins, density=False)
        px = hx / hx.sum()
        py = hy / hy.sum()
        pxy = hxy / max(hxy.sum(), 1)
        nonzero = pxy > 0
        px_grid = px[:, None] * py[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mi[lag] = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / np.maximum(px_grid[nonzero], 1e-12)))
    # premier minimum local
    for lag in range(2, max_lag):
        if mi[lag] < mi[lag - 1] and mi[lag] < mi[lag + 1]:
            return lag
    return max(1, int(np.argmin(mi[1:]) + 1))


def takens_embed(series: np.ndarray, m: int = 3, tau: int = 1) -> np.ndarray:
    """Reconstruction phase space par delay coordinates. (N, m)."""
    n = len(series)
    span = (m - 1) * tau
    if span >= n:
        raise ValueError(f"Series too short: {n} <= span {span}")
    out = np.zeros((n - span, m), dtype=np.float32)
    for j in range(m):
        out[:, j] = series[j * tau : j * tau + (n - span)]
    return out


def recurrence_matrix(traj: np.ndarray, eps: float | None = None, max_n: int = 400) -> tuple[np.ndarray, float]:
    """Recurrence plot binaire. Down-sample si trop long. Retourne (R, eps)."""
    n = traj.shape[0]
    if n > max_n:
        idx = np.linspace(0, n - 1, max_n).astype(int)
        traj = traj[idx]
        n = max_n
    dists = squareform(pdist(traj, metric="euclidean"))
    if eps is None:
        # 10e percentile des distances non-diagonales = seuil
        mask = ~np.eye(n, dtype=bool)
        eps = float(np.percentile(dists[mask], 10))
    R = (dists <= eps).astype(np.uint8)
    return R, eps


def rqa_stats(R: np.ndarray, lmin: int = 2) -> dict:
    """RQA basique : RR, DET, LAM."""
    n = R.shape[0]
    rr = float(R.sum() / (n * n))

    # Determinism : fraction des points sur diagonales >= lmin
    diag_points = 0
    det_points = 0
    for k in range(-(n - 1), n):
        if k == 0:
            continue
        d = np.diag(R, k=k)
        runs = _run_lengths(d)
        diag_points += int(d.sum())
        det_points += int(sum(r for r in runs if r >= lmin))
    det = det_points / max(diag_points, 1)

    # Laminarity : fraction des points sur verticales >= lmin
    lam_points = 0
    vert_points = 0
    for col in range(n):
        v = R[:, col]
        runs = _run_lengths(v)
        vert_points += int(v.sum())
        lam_points += int(sum(r for r in runs if r >= lmin))
    lam = lam_points / max(vert_points, 1)

    return {"RR": rr, "DET": det, "LAM": lam}


def _run_lengths(binary: np.ndarray) -> list[int]:
    runs: list[int] = []
    count = 0
    for v in binary:
        if v:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def lyapunov_rosenstein(
    traj: np.ndarray,
    mean_period: int = 5,
    k_neighbors: int = 1,
    max_steps: int | None = None,
) -> tuple[float, np.ndarray]:
    """Rosenstein 1993 : largest Lyapunov exponent depuis trajectoire embeded.
    Retourne (lambda_estim, divergence_curve)."""
    n = traj.shape[0]
    if max_steps is None:
        max_steps = min(40, n // 4)

    # Pour chaque point i, trouve nearest neighbor j avec |i-j| > mean_period
    dists = squareform(pdist(traj, metric="euclidean"))
    np.fill_diagonal(dists, np.inf)
    # exclure voisins temporels proches
    for i in range(n):
        lo = max(0, i - mean_period)
        hi = min(n, i + mean_period + 1)
        dists[i, lo:hi] = np.inf

    neighbors = np.argmin(dists, axis=1)
    # validité : i et neighbor doivent avoir au moins max_steps de marge
    valid = (np.arange(n) + max_steps < n) & (neighbors + max_steps < n)
    idx = np.where(valid)[0]
    if len(idx) < 10:
        return 0.0, np.zeros(max_steps)

    log_div = np.zeros(max_steps, dtype=np.float64)
    counts = np.zeros(max_steps, dtype=np.int32)
    for step in range(max_steps):
        i_s = idx + step
        j_s = neighbors[idx] + step
        d = np.linalg.norm(traj[i_s] - traj[j_s], axis=1)
        d = d[d > 1e-12]
        if len(d) > 0:
            log_div[step] = np.mean(np.log(d))
            counts[step] = len(d)

    # fit lineaire sur partie initiale (typ. 0..max_steps//2)
    fit_end = max(5, max_steps // 2)
    t = np.arange(fit_end)
    slope = np.polyfit(t, log_div[:fit_end], 1)[0]
    return float(slope), log_div


def correlation_dimension(traj: np.ndarray, n_eps: int = 14) -> float:
    """Estimation rapide dim de correlation (Grassberger-Procaccia)."""
    n = traj.shape[0]
    if n > 800:
        idx = np.linspace(0, n - 1, 800).astype(int)
        traj = traj[idx]
        n = 800
    dists = pdist(traj, metric="euclidean")
    dists = dists[dists > 1e-12]
    if len(dists) < 10:
        return 0.0
    eps_min, eps_max = np.percentile(dists, 5), np.percentile(dists, 95)
    if eps_min <= 0 or eps_max <= eps_min:
        return 0.0
    epses = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
    C = np.array([(dists <= e).sum() for e in epses], dtype=np.float64)
    C = C / (n * (n - 1) / 2)
    valid = C > 0
    if valid.sum() < 4:
        return 0.0
    slope = np.polyfit(np.log(epses[valid]), np.log(C[valid]), 1)[0]
    return float(slope)


def analyse_trajectory(coords: np.ndarray, max_n: int = 600) -> dict:
    """Pipeline complet : embed-ready coords (N, d) -> stats dynamiques."""
    n = coords.shape[0]
    if n > max_n:
        idx = np.linspace(0, n - 1, max_n).astype(int)
        traj = coords[idx]
    else:
        traj = coords

    lam, div_curve = lyapunov_rosenstein(traj.astype(np.float32))
    R, eps = recurrence_matrix(traj, max_n=300)
    rqa = rqa_stats(R)
    d2 = correlation_dimension(traj)

    return {
        "lyapunov": round(lam, 4),
        "correlation_dim": round(d2, 3),
        "epsilon": round(eps, 4),
        "rqa": {k: round(v, 4) for k, v in rqa.items()},
        "divergence_curve": div_curve.tolist(),
        "recurrence": R.tolist(),
        "recurrence_size": R.shape[0],
    }
