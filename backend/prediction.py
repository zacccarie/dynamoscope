"""Prédiction trajectoire latente : f(z[t]) -> z[t+1].
Mesure surprise = prediction error. Lien world models + free energy + JEPA."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class LatentMLP(nn.Module):
    """MLP 2-couches pour prédire next latent depuis current."""
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        h = min(hidden, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual : x + delta
        return x + self.net(x)


def train_model(
    latents: np.ndarray, epochs: int = 150, lr: float = 1e-3,
    hidden: int = 256, device: str = "cpu",
) -> tuple[LatentMLP, torch.Tensor, torch.Tensor]:
    """Train et retourne (model, mean, std) pour reuse."""
    n, d = latents.shape
    x = torch.from_numpy(latents[:-1]).float().to(device)
    y = torch.from_numpy(latents[1:]).float().to(device)
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    x_n = (x - mean) / std
    y_n = (y - mean) / std
    model = LatentMLP(d, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_n)
        loss = ((pred - y_n) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model, mean, std


def _project_via_knn(pred_arr: np.ndarray, all_latents: np.ndarray, coords_3d: np.ndarray, k: int = 5) -> list[list[float]]:
    all_norm = all_latents / np.maximum(np.linalg.norm(all_latents, axis=1, keepdims=True), 1e-9)
    pred_norm = pred_arr / np.maximum(np.linalg.norm(pred_arr, axis=1, keepdims=True), 1e-9)
    k_eff = min(k, all_latents.shape[0])
    out: list[list[float]] = []
    for p in pred_norm:
        sims = all_norm @ p
        top_k = np.argpartition(-sims, k_eff - 1)[:k_eff]
        w = np.exp(sims[top_k] * 10)
        w /= w.sum()
        coord = (w[:, None] * coords_3d[top_k]).sum(axis=0)
        out.append(coord.tolist())
    return out


def counterfactual_rollouts(
    latents: np.ndarray,
    coords_3d: np.ndarray,
    start_idx: int,
    horizon: int = 20,
    n_samples: int = 8,
    noise_scale: float = 0.05,
    epochs: int = 120,
    hidden: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Cone of futures : N rollouts depuis latent perturbé. Visualise diversité MLP imaginaire."""
    n, d = latents.shape
    if not (0 <= start_idx < n - 1):
        raise ValueError("Invalid start_idx")
    horizon = max(1, min(horizon, 100))

    model, mean, std = train_model(latents, epochs=epochs, hidden=hidden, device=device)
    rng = np.random.RandomState(seed)
    # Noise calibre sur std des latents (perturbation relative à la dispersion)
    sigma = float(latents.std()) * noise_scale

    z_base = torch.from_numpy(latents[start_idx:start_idx+1]).float().to(device)
    rollouts: list[list[list[float]]] = []
    with torch.no_grad():
        for s in range(n_samples):
            # Perturbation Gaussienne sur latent initial
            noise = torch.from_numpy(rng.randn(1, d).astype(np.float32) * sigma).to(device)
            z = z_base + noise
            cur = (z - mean) / std
            preds = []
            for _ in range(horizon):
                cur = model(cur)
                preds.append(((cur * std + mean).cpu().numpy()[0]))
            pred_arr = np.stack(preds, axis=0)
            ghost = _project_via_knn(pred_arr, latents, coords_3d)
            rollouts.append(ghost)

    return {
        "start_idx": int(start_idx),
        "horizon": int(horizon),
        "n_samples": int(n_samples),
        "noise_scale": float(noise_scale),
        "rollouts": rollouts,
    }


def rollout(
    latents: np.ndarray,
    coords_3d: np.ndarray,
    start_idx: int,
    horizon: int = 20,
    epochs: int = 150,
    hidden: int = 256,
    device: str = "cpu",
) -> dict:
    """Rollout MLP itératif depuis start_idx. Projette ghost en 3D via k-NN weighted.
    Compare aussi à trajectoire réelle si disponible."""
    n, d = latents.shape
    if not (0 <= start_idx < n - 1):
        raise ValueError(f"start_idx {start_idx} out of [0, {n-1})")
    horizon = max(1, min(horizon, n - start_idx - 1, 100))

    model, mean, std = train_model(latents, epochs=epochs, hidden=hidden, device=device)

    # Iterative rollout
    z = torch.from_numpy(latents[start_idx:start_idx+1]).float().to(device)
    z_n = (z - mean) / std
    pred_latents: list[np.ndarray] = []
    with torch.no_grad():
        cur = z_n.clone()
        for _ in range(horizon):
            cur = model(cur)
            denorm = (cur * std + mean).cpu().numpy()[0]
            pred_latents.append(denorm)
    pred_arr = np.stack(pred_latents, axis=0)

    # Projette en 3D via k-NN weighted (cosine sim avec latents tous)
    all_norm = latents / np.maximum(np.linalg.norm(latents, axis=1, keepdims=True), 1e-9)
    pred_norm = pred_arr / np.maximum(np.linalg.norm(pred_arr, axis=1, keepdims=True), 1e-9)
    ghost_3d = []
    k = min(5, n)
    for p in pred_norm:
        sims = all_norm @ p
        top_k = np.argpartition(-sims, k - 1)[:k]
        w = np.exp(sims[top_k] * 10)
        w /= w.sum()
        coord = (w[:, None] * coords_3d[top_k]).sum(axis=0)
        ghost_3d.append(coord.tolist())

    # Per-step error vs actual (si horizon points still in trajectory)
    actual_latents = latents[start_idx+1 : start_idx+1+horizon]
    step_errors = []
    for i in range(min(len(actual_latents), len(pred_arr))):
        err = float(np.linalg.norm(pred_arr[i] - actual_latents[i]))
        step_errors.append(err)

    return {
        "start_idx": int(start_idx),
        "horizon": int(horizon),
        "ghost_3d": ghost_3d,
        "step_errors": step_errors,
        "n_actual_compared": len(step_errors),
    }


def fit_predictor(
    latents: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 256,
    device: str = "cpu",
) -> dict:
    """Train MLP minimal sur trajectoire. Retourne predictions + errors."""
    n, d = latents.shape
    if n < 8:
        raise ValueError("Trajectory too short")

    x = torch.from_numpy(latents[:-1]).float().to(device)
    y = torch.from_numpy(latents[1:]).float().to(device)

    # Standardise pour stabilite
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    x_n = (x - mean) / std
    y_n = (y - mean) / std

    model = LatentMLP(d, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(x_n)
        loss = ((pred - y_n) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # Inference + per-frame errors
    model.eval()
    with torch.no_grad():
        pred_n = model(x_n)
        # De-standardize
        pred = pred_n * std + mean
        # Per-step L2 error in original latent space
        err = ((pred - y) ** 2).sum(dim=1).sqrt().cpu().numpy()

    # Baseline : prediction "identite" (z[t+1] = z[t])
    baseline_err = np.linalg.norm(latents[1:] - latents[:-1], axis=1)

    # Normalisation pour comparaison
    err_norm = err / np.maximum(baseline_err.mean(), 1e-9)
    baseline_norm = baseline_err / np.maximum(baseline_err.mean(), 1e-9)
    # Surprise per-frame : prepend 0 pour alignment longueur n
    surprise = np.concatenate([[0.0], err]).astype(np.float32)
    baseline_surprise = np.concatenate([[0.0], baseline_err]).astype(np.float32)

    predictability = float(1.0 - min(1.0, err.mean() / max(baseline_err.mean(), 1e-9)))

    return {
        "loss_curve": losses,
        "final_loss": float(losses[-1]),
        "mean_error": float(err.mean()),
        "std_error": float(err.std()),
        "max_error": float(err.max()),
        "mean_baseline": float(baseline_err.mean()),
        "predictability": predictability,
        "surprise": surprise.tolist(),
        "baseline_surprise": baseline_surprise.tolist(),
        "epochs": epochs,
    }
