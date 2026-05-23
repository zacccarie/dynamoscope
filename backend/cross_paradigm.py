"""Cross-paradigm analysis : connecte classical observables et neural latents.

Question : que captent réellement neural encoders ? Sont-ils redécouverte
d'observables physiques classiques, ou extraient-ils signal supra-physical ?

Approche : linear probe. Pour chaque neural dim d, fit ridge regression
z_neural[:, d] = β · observables + b. Report R² + top-K observables
explanantes par dim.

R² interprétation :
- R² > 0.8 : neural dim ≈ combinaison linéaire d'observables (interpretable)
- R² < 0.2 : neural dim capture signal qui ne se réduit pas linéairement aux 12 raw obs
- 0.2 < R² < 0.8 : partiellement interpretable

Si R² élevé majorité dims → neural redécouvre observables classiques.
Si R² faible → neural extrait *quelque chose de nouveau*. Both findings
sont scientifiquement intéressants.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .observables import OBSERVABLE_KEYS


def observable_probe(
    z_neural: np.ndarray,
    observables: dict[str, np.ndarray],
    alpha: float = 1.0,
    keys: list[str] | None = None,
) -> dict:
    """Linear probe : explain z_neural dims as combinations of observables.

    Args:
        z_neural: (T, D) trajectoire latente.
        observables: dict {key: (T,)} from compute_observables.
        alpha: ridge regularization.
        keys: subset d'observables à utiliser (default = tous).
    Returns:
        {
          "mean_r2", "max_r2", "median_r2",
          "n_dims_r2_above_0.5",
          "n_dims_r2_above_0.8",
          "per_dim_r2": list (D,) R² par dim,
          "per_dim_top_obs": list (D,) [(obs_key, coef), ...] top-3 observables
              avec coefficients standardisés signed.
          "obs_keys"
        }
    """
    obs_keys = keys if keys is not None else OBSERVABLE_KEYS
    X = np.stack([observables[k] for k in obs_keys], axis=1)
    T = min(X.shape[0], z_neural.shape[0])
    X = X[:T]
    z = z_neural[:T]

    if T < 4 or z.shape[1] == 0:
        return {
            "mean_r2": 0.0, "max_r2": 0.0, "median_r2": 0.0,
            "n_dims_r2_above_0.5": 0, "n_dims_r2_above_0.8": 0,
            "per_dim_r2": [], "per_dim_top_obs": [],
            "obs_keys": obs_keys,
        }

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    per_dim_r2 = []
    per_dim_top_obs = []
    for d in range(z.shape[1]):
        y = z[:, d]
        if y.std() < 1e-9:
            per_dim_r2.append(0.0)
            per_dim_top_obs.append([])
            continue
        model = Ridge(alpha=alpha).fit(Xs, y)
        r2 = float(model.score(Xs, y))
        per_dim_r2.append(r2)
        coefs = model.coef_
        ranked = sorted(zip(obs_keys, coefs), key=lambda x: -abs(x[1]))
        per_dim_top_obs.append([(k, float(c)) for k, c in ranked[:3]])

    arr = np.array(per_dim_r2)
    return {
        "mean_r2": float(arr.mean()),
        "max_r2": float(arr.max()),
        "median_r2": float(np.median(arr)),
        "n_dims_r2_above_0.5": int((arr > 0.5).sum()),
        "n_dims_r2_above_0.8": int((arr > 0.8).sum()),
        "n_dims_total": int(len(arr)),
        "per_dim_r2": per_dim_r2,
        "per_dim_top_obs": per_dim_top_obs,
        "obs_keys": obs_keys,
    }


def observable_probe_transfer(
    z_train: np.ndarray, obs_train: dict[str, np.ndarray],
    z_test: np.ndarray, obs_test: dict[str, np.ndarray],
    alpha: float = 1.0,
) -> dict:
    """Train probe on (z_train, obs_train), evaluate on (z_test, obs_test).

    Measures generalization : si probe fit on procedural transfère à real video,
    neural-classical mapping est universel.

    Returns:
        {"train_r2": ..., "test_r2": ..., "transfer_ratio": test_r2 / train_r2}
    """
    obs_keys = OBSERVABLE_KEYS
    X_tr = np.stack([obs_train[k] for k in obs_keys], axis=1)
    T_tr = min(X_tr.shape[0], z_train.shape[0])
    X_tr, z_tr = X_tr[:T_tr], z_train[:T_tr]

    X_te = np.stack([obs_test[k] for k in obs_keys], axis=1)
    T_te = min(X_te.shape[0], z_test.shape[0])
    X_te, z_te = X_te[:T_te], z_test[:T_te]

    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_te = scaler.transform(X_te)

    train_r2 = []
    test_r2 = []
    for d in range(z_tr.shape[1]):
        y_tr = z_tr[:, d]
        if y_tr.std() < 1e-9:
            train_r2.append(0.0)
            test_r2.append(0.0)
            continue
        model = Ridge(alpha=alpha).fit(Xs_tr, y_tr)
        train_r2.append(float(model.score(Xs_tr, y_tr)))
        # Test on z_test using same scaler + model
        if d < z_te.shape[1]:
            y_te = z_te[:, d]
            test_r2.append(float(model.score(Xs_te, y_te)))
        else:
            test_r2.append(0.0)

    tr_mean = float(np.mean(train_r2))
    te_mean = float(np.mean(test_r2))
    return {
        "train_r2_mean": tr_mean,
        "test_r2_mean": te_mean,
        "transfer_ratio": te_mean / max(abs(tr_mean), 1e-9),
        "per_dim_train_r2": train_r2,
        "per_dim_test_r2": test_r2,
    }
