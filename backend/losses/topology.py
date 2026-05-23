"""TopologyPreservationLoss : proxy différentiable pour préservation topologique.

Vraie Wasserstein-PD non-diff (matching combinatoire). Approche pragmatique :

Distance matrix preservation entre input space (raw features ou frames idx) et
latent space. Si encoder préserve la métrique relative, alors la structure
topologique (clusters, loops, holes) est largement préservée (lemme nervures).

Méthode :
1. Compute pairwise distances dans z (current) et dans z_ref ou index
2. Rank correlation Spearman approximé par soft-ranking
   OU MSE direct entre distances normalisées
3. Loss = 1 − ρ ou MSE

Pour cas plus puissant : sliced Wasserstein sur persistence diagram approché
via subsampling de paires + sorted distances (Carriere et al. 2017).
Provided as optional sliced_w mode.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TopologyPreservationLoss(nn.Module):
    """Préservation de structure métrique entre input et latent.

    Args:
        mode: "distance_corr" (Spearman-like) | "sliced_w" (sliced-Wasserstein)
        n_slices: pour sliced_w, nb directions de projection.
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

    def forward(self, z: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (T, D) latent trajectory (gradient-tracking).
            z_ref: (T, D_ref) reference embedding (pas de gradient).
        Returns:
            scalar loss différentiable w.r.t. z.
        """
        if z.shape[0] != z_ref.shape[0]:
            raise ValueError(f"T mismatch: {z.shape[0]} vs {z_ref.shape[0]}")

        d_lat = torch.cdist(z, z)
        d_ref = torch.cdist(z_ref.detach(), z_ref.detach())

        # Normalize pour comparer échelles différentes
        d_lat_norm = d_lat / d_lat.mean().clamp_min(1e-9)
        d_ref_norm = d_ref / d_ref.mean().clamp_min(1e-9)

        if self.mode == "distance_corr":
            # Pearson correlation sur distances flatten (upper triangle)
            T = z.shape[0]
            mask = torch.triu(
                torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1
            )
            x = d_lat_norm[mask]
            y = d_ref_norm[mask]
            x_c = x - x.mean()
            y_c = y - y.mean()
            num = (x_c * y_c).sum()
            den = (x_c.norm() * y_c.norm()).clamp_min(1e-9)
            corr = num / den
            loss = 1.0 - corr
            return self.weight * loss

        # sliced_w : projet distances en 1D, sort, L1 entre dist sorted
        T = z.shape[0]
        idx_i, idx_j = torch.triu_indices(T, T, offset=1, device=z.device)
        d_lat_pairs = d_lat_norm[idx_i, idx_j]  # (P,)
        d_ref_pairs = d_ref_norm[idx_i, idx_j]
        # Project sur n_slices directions aléatoires (1D donc trivial : signe)
        # Pour 1D : sliced-Wasserstein = L1 entre sorted vectors
        slices = []
        for _ in range(self.n_slices):
            # Random permutation sub-sample
            perm = torch.randperm(d_lat_pairs.shape[0], device=z.device)[
                : d_lat_pairs.shape[0]
            ]
            a = torch.sort(d_lat_pairs[perm])[0]
            b = torch.sort(d_ref_pairs[perm])[0]
            slices.append((a - b).abs().mean())
        loss = torch.stack(slices).mean()
        return self.weight * loss
