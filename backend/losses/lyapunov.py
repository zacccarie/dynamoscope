"""LyapunovMatchingLoss : proxy différentiable pour exposant de Lyapunov.

Rosenstein 1993 standard non-diff (k-NN + log-slope fit). Notre proxy :

1. Pour chaque pas t, identifier paire (i, j) avec distance latente minimale
2. Mesurer croissance log ‖z_{i+τ} − z_{j+τ}‖ / ‖z_i − z_j‖
3. Loss = MSE entre pente empirique et target_lyapunov fourni

Gradient flows back through z, donc through encoder params.

Use cases :
- Forcer un encoder à reproduire le chaos d'un système connu (régularisation)
- Réduire chaos parasite : target_lyapunov=0 → drive trajectory vers ordered
- Augmenter chaos : target_lyapunov>0 → drive vers butterfly effect
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LyapunovMatchingLoss(nn.Module):
    """Differentiable Lyapunov estimator + matching loss.

    Args:
        target_lyapunov: valeur cible. 0 = pas de chaos, > 0 = chaotic.
        tau_range: horizons utilisés pour le fit log-linéaire.
        k_pairs: nombre de paires nearest-neighbour considérées par fenêtre.
        weight: multiplicateur appliqué à la loss finale.
    """

    def __init__(
        self,
        target_lyapunov: float = 0.0,
        tau_range: tuple[int, int] = (1, 8),
        k_pairs: int = 16,
        weight: float = 1.0,
    ):
        super().__init__()
        self.target = float(target_lyapunov)
        self.tau_min, self.tau_max = tau_range
        self.k_pairs = k_pairs
        self.weight = weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (T, D) tensor trajectoire latente.
        Returns:
            scalar loss différentiable.
        """
        if z.dim() != 2:
            raise ValueError(f"expected (T, D), got shape {tuple(z.shape)}")
        T, D = z.shape
        if T < self.tau_max + 2:
            return z.sum() * 0.0  # no-op si trop court

        # Distance matrix initiale (frames trop proches en temps exclues)
        # Theiler window = 1 pour éviter trivial neighbours
        dist0 = torch.cdist(z, z)  # (T, T)
        eye_mask = torch.eye(T, device=z.device, dtype=torch.bool)
        # Exclure |i-j| <= 1 (Theiler) ET t+tau_max hors bornes
        time_idx = torch.arange(T, device=z.device)
        diff_time = (time_idx.unsqueeze(0) - time_idx.unsqueeze(1)).abs()
        mask = eye_mask | (diff_time <= 1)
        # Exclure paires dont j + tau_max >= T (i.e. impossible de mesurer growth)
        valid_j = time_idx + self.tau_max < T
        valid_i = time_idx + self.tau_max < T
        mask = mask | (~valid_i.unsqueeze(1)) | (~valid_j.unsqueeze(0))

        # k_pairs plus proches paires non masquées
        dist_masked = dist0.masked_fill(mask, float("inf"))
        flat = dist_masked.flatten()
        k = min(self.k_pairs, flat.numel())
        _, top_idx = torch.topk(flat, k, largest=False)
        i_idx = top_idx // T
        j_idx = top_idx % T
        d0 = dist0[i_idx, j_idx].clamp_min(1e-9)  # (k,)

        # Compute log-growth log(d_tau / d_0) pour chaque tau
        log_growth_per_tau = []
        for tau in range(self.tau_min, self.tau_max + 1):
            d_tau = torch.norm(z[i_idx + tau] - z[j_idx + tau], dim=1).clamp_min(1e-9)
            lg = torch.log(d_tau / d0)
            log_growth_per_tau.append(lg.mean())
        log_growth = torch.stack(log_growth_per_tau)  # (n_tau,)

        # Linear fit log_growth ≈ λ · tau via least-squares fermé
        taus = torch.arange(
            self.tau_min, self.tau_max + 1, device=z.device, dtype=z.dtype
        )
        tau_mean = taus.mean()
        lg_mean = log_growth.mean()
        cov = ((taus - tau_mean) * (log_growth - lg_mean)).sum()
        var = ((taus - tau_mean) ** 2).sum().clamp_min(1e-9)
        lyap_est = cov / var

        loss = (lyap_est - self.target) ** 2
        return self.weight * loss
