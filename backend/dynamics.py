"""Analyse dynamique : vidéo = trajectoire dans espace d'états.

Métriques classiques systèmes dynamiques :
- Lyapunov exponent : taux divergence trajectoires proches (chaos quantifié)
- Recurrence plot : visualise retours dans même région phase space
- RQA (Recurrence Quantification Analysis) : stats structure récurrente
- Correlation dimension : effective dim via Grassberger-Procaccia
- Takens embedding : reconstruction phase space depuis observable scalaire
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform


def mutual_info_lag(series: np.ndarray, max_lag: int = 50, bins: int = 16) -> int:
    """Trouve lag τ optimal pour Takens embedding via premier minimum mutual info.

    Théorie : MI(x(t), x(t+τ)) mesure dépendance temporelle.
    Premier minimum local de MI(τ) = lag où x(t+τ) "moins corrélé" à x(t) → bon pour embedding.

    Args:
        series: signal scalaire 1D
        max_lag: lag maximum exploré
        bins: histogram bins pour estimation MI

    Returns:
        Lag τ optimal (int).
    """
    n = len(series)
    mi = np.zeros(max_lag + 1)
    series = (series - series.min()) / max(np.ptp(series), 1e-9)
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
    # Premier minimum local de MI = lag optimal Takens
    for lag in range(2, max_lag):
        if mi[lag] < mi[lag - 1] and mi[lag] < mi[lag + 1]:
            return lag
    return max(1, int(np.argmin(mi[1:]) + 1))


def takens_embed(series: np.ndarray, m: int = 3, tau: int = 1) -> np.ndarray:
    """Reconstruction phase space par delay coordinates (Takens 1981).

    Théorème Takens : pour système dynamique générique, l'embedding
    (x(t), x(t+τ), ..., x(t+(m-1)τ)) avec m ≥ 2d+1 (d = dim attracteur)
    est diffeomorphe à la dynamique originale. Permet reconstruire l'attracteur
    depuis UNE seule observable scalaire.

    Args:
        series: signal 1D (e.g., 1 dimension du latent)
        m: dimension embedding
        tau: delay (typiquement 1er minimum MI)

    Returns:
        Trajectoire embedded (N - (m-1)τ, m).
    """
    n = len(series)
    span = (m - 1) * tau
    if span >= n:
        raise ValueError(f"Series too short: {n} <= span {span}")
    out = np.zeros((n - span, m), dtype=np.float32)
    for j in range(m):
        out[:, j] = series[j * tau : j * tau + (n - span)]
    return out


def recurrence_matrix(traj: np.ndarray, eps: float | None = None, max_n: int = 400) -> tuple[np.ndarray, float]:
    """Matrice de récurrence binaire : R[i,j] = 1 si ‖traj[i] - traj[j]‖ < eps.

    Inventée Eckmann-Kamphorst-Ruelle 1987. Visualise structure temporelle :
    - Diagonale = trivialement R[i,i]=1
    - Lignes diagonales parallèles = orbites quasi-périodiques (système revisite même état)
    - Blocs = phases longues quasi-stationnaires
    - Patterns aléatoires = stochastique

    Args:
        traj: (N, d) trajectoire
        eps: seuil distance (None = 10ème percentile auto)
        max_n: subsample si N > max_n (matrices N² explosent vite)

    Returns:
        (R binaire (n,n), eps utilisé)
    """
    n = traj.shape[0]
    if n > max_n:
        idx = np.linspace(0, n - 1, max_n).astype(int)
        traj = traj[idx]
        n = max_n
    dists = squareform(pdist(traj, metric="euclidean"))
    if eps is None:
        # 10e percentile = ~10% paires sont "récurrentes" en moyenne
        mask = ~np.eye(n, dtype=bool)
        eps = float(np.percentile(dists[mask], 10))
    R = (dists <= eps).astype(np.uint8)
    return R, eps


def rqa_stats(R: np.ndarray, lmin: int = 2) -> dict:
    """RQA = Recurrence Quantification Analysis (Marwan et al.).

    3 métriques principales :
    - **RR** (Recurrence Rate) : densité globale points récurrents
    - **DET** (Determinism) : fraction points sur diagonales lmin+ → prédictibilité système
    - **LAM** (Laminarity) : fraction points sur verticales lmin+ → phases quasi-stationnaires

    DET élevé = trajectoire prédictible (revient sur même chemin).
    LAM élevé = système "freeze" temporairement dans régimes stables.
    """
    n = R.shape[0]
    rr = float(R.sum() / (n * n))

    # Determinism : compte points sur diagonales de longueur ≥ lmin
    diag_points = 0
    det_points = 0
    for k in range(-(n - 1), n):
        if k == 0:
            continue  # skip diagonale principale (trivialement 1)
        d = np.diag(R, k=k)
        runs = _run_lengths(d)
        diag_points += int(d.sum())
        det_points += int(sum(r for r in runs if r >= lmin))
    det = det_points / max(diag_points, 1)

    # Laminarity : compte points sur verticales
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
    """Helper : longueurs des runs consécutifs de 1 dans vecteur binaire.

    Ex : [0,1,1,1,0,1,1,0] → [3, 2]. Utilisé par DET/LAM.
    """
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
    """Plus grand exposant de Lyapunov via Rosenstein et al. 1993.

    Théorie : si λ > 0, deux trajectoires initialement proches divergent
    exponentiellement comme `d(t) ≈ d(0) · exp(λt)`. λ quantifie chaos.

    Algorithme :
    1. Pour chaque point i, trouve nearest neighbor j (exclus voisins temporels proches)
    2. Trace divergence ‖traj[i+k] - traj[j+k]‖ pour k=0..max_steps
    3. Fit linéaire sur log(divergence moyenne) vs k → pente = λ

    Args:
        traj: trajectoire (N, d)
        mean_period: exclut voisins temporels |i-j| < mean_period
        k_neighbors: nombre voisins par point (1 = rapide, plus = stable)
        max_steps: horizon temporel pour fit

    Returns:
        (λ_estim, courbe_divergence)
    """
    n = traj.shape[0]
    if max_steps is None:
        max_steps = min(40, n // 4)

    dists = squareform(pdist(traj, metric="euclidean"))
    np.fill_diagonal(dists, np.inf)
    # Exclut voisins temporels proches (Theiler window) — évite fausse récurrence
    for i in range(n):
        lo = max(0, i - mean_period)
        hi = min(n, i + mean_period + 1)
        dists[i, lo:hi] = np.inf

    neighbors = np.argmin(dists, axis=1)
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

    # Fit linéaire log-divergence sur partie initiale (avant saturation)
    fit_end = max(5, max_steps // 2)
    t = np.arange(fit_end)
    slope = np.polyfit(t, log_div[:fit_end], 1)[0]
    return float(slope), log_div


def correlation_dimension(traj: np.ndarray, n_eps: int = 14) -> float:
    """Dimension de corrélation (Grassberger-Procaccia 1983).

    Théorie : compte paires de points à distance < eps. Pour fractale,
    C(eps) ∝ eps^d où d = dim fractale (cap. = capacity dim ≤ box-counting).

    Pour Lorenz, théorie prédit d ≈ 2.06 (attracteur quasi-2D dans 3D).
    Calculé via régression log-log de C(eps) vs eps.
    """
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
    """Pipeline complet analyse dynamique : Lyapunov + RQA + corr.dim + recurrence.

    Args:
        coords: trajectoire (N, d) — typiquement coords 3D UMAP
        max_n: subsample si N > max_n pour limiter coût matriciel

    Returns:
        Dict avec lyapunov, correlation_dim, epsilon, rqa {RR/DET/LAM},
        divergence_curve, recurrence matrix binary.
    """
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
