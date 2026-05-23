"""ClassicalAlignmentLoss : distill classical Takens trajectory into neural latent.

Idée centrale :
- Classical Takens trajectory = oracle structure dynamique (model-free)
- Neural latent trajectory = perceptual encoder (compression sémantique)
- Loss = align neural distance-geometry à classical distance-geometry
- Procrustes-invariant : pas de matching point-à-point, juste structure relative

Si ça marche : neural encoder gagne dynamics-richness du classical
SANS perdre semantic invariance (recon loss préservée en parallèle).

Implementation :
1. Calcule distance matrices d_neural, d_classical (sur fenêtre commune T)
2. Normalize each by mean
3. Loss = 1 − Pearson correlation entre upper-triangle distances

Pearson correlation est invariant aux rotations, échelles, translations →
on aligne la structure géométrique, pas l'embedding absolu.

Alternative : sliced-Wasserstein sur distances (mode "sliced_w"), comme
TopologyPreservationLoss. Choix par défaut = distance_corr.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ClassicalAlignmentLoss(nn.Module):
    """Align neural latent geometry to classical Takens reference geometry.

    Args:
        mode: "distance_corr" (default, Pearson on pairwise) | "sliced_w".
        n_slices: pour sliced_w.
        weight: poids final.
    """

    def __init__(
        self,
        mode: str = "distance_corr",
        n_slices: int = 16,
        weight: float = 1.0,
    ):
        super().__init__()
        if mode not in ("distance_corr", "sliced_w"):
            raise ValueError(f"mode must be distance_corr|sliced_w, got {mode}")
        self.mode = mode
        self.n_slices = n_slices
        self.weight = weight

    def forward(
        self,
        z_neural: torch.Tensor,
        z_classical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_neural: (T_n, D_n) — gradient-tracking neural latent.
            z_classical: (T_c, D_c) — pre-computed classical Takens trajectory.
                Pas de gradient (oracle fixe).
        Returns:
            scalar loss.

        Si T_n != T_c (classical truncate par delay embed), on aligne sur
        min(T_n, T_c) premiers points. C'est ok car classical = z[:(N-(m-1)τ)],
        donc même origine temporelle que neural[: ce nombre].
        """
        if z_neural.dim() != 2 or z_classical.dim() != 2:
            raise ValueError(
                f"expected 2D tensors, got {z_neural.shape}, {z_classical.shape}"
            )
        T = min(z_neural.shape[0], z_classical.shape[0])
        if T < 4:
            return z_neural.sum() * 0.0

        z_n = z_neural[:T]
        z_c = z_classical[:T].detach()

        d_n = torch.cdist(z_n, z_n)
        d_c = torch.cdist(z_c, z_c).to(z_n.device).to(z_n.dtype)

        # Normalize par mean (échelle invariante)
        d_n = d_n / d_n.mean().clamp_min(1e-9)
        d_c = d_c / d_c.mean().clamp_min(1e-9)

        if self.mode == "distance_corr":
            mask = torch.triu(
                torch.ones(T, T, device=z_n.device, dtype=torch.bool), diagonal=1
            )
            x = d_n[mask]
            y = d_c[mask]
            x_c = x - x.mean()
            y_c = y - y.mean()
            num = (x_c * y_c).sum()
            den = (x_c.norm() * y_c.norm()).clamp_min(1e-9)
            corr = num / den
            return self.weight * (1.0 - corr)

        # sliced_w
        idx_i, idx_j = torch.triu_indices(T, T, offset=1, device=z_n.device)
        d_n_pairs = d_n[idx_i, idx_j]
        d_c_pairs = d_c[idx_i, idx_j]
        slices = []
        for _ in range(self.n_slices):
            perm = torch.randperm(d_n_pairs.shape[0], device=z_n.device)
            a = torch.sort(d_n_pairs[perm])[0]
            b = torch.sort(d_c_pairs[perm])[0]
            slices.append((a - b).abs().mean())
        return self.weight * torch.stack(slices).mean()


def compute_classical_target(
    frames_np,
    observable: str = "motion",
    m: int = 3,
    tau: int | None = None,
) -> torch.Tensor:
    """Helper : génère classical Takens target tensor depuis frames numpy.

    Args:
        frames_np: (N, H, W, 3) uint8 ou float [0,1].
        observable: clé observable.
        m: dimension plongement.
        tau: délai, None = auto.
    Returns:
        (T, m) tensor float32 sur CPU.
    """
    from ..observables import compute_observables
    from ..embedding import auto_tau, delay_embed

    obs = compute_observables(frames_np)
    series = obs[observable]
    if tau is None:
        tau = auto_tau(series)
    coords = delay_embed(series, m=m, tau=tau)
    return torch.from_numpy(coords).float()
