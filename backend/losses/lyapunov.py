"""LyapunovMatchingLoss : Rosenstein 1993 différentiable.

Implementation fidèle à Rosenstein, Collins & De Luca (1993) :
"A practical method for calculating largest Lyapunov exponents from
small data sets" — Physica D.

Algo :
1. Pour chaque anchor i, trouver plus proche voisin j avec Theiler window
   (|i-j| > W où W ~ période moyenne — éviter recurrences)
2. Tracker <ln d(i,j)(t)> averaged sur tous anchors, où d est distance
   euclidienne entre trajectoires
3. Slope de <ln d(t)> vs t dans région linéaire ≈ λ (par unité de t)

Avantages vs version précédente :
- Tous les anchors valides utilisés (pas top-k qui sature)
- Theiler window évite paires temporellement proches mais artificielles
- Mean of logs (correct Rosenstein) vs log of mean

Différentiable : grad flow via z_seq[anchors + τ] et z_seq[neighbors + τ].
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LyapunovMatchingLoss(nn.Module):
    """Differentiable Rosenstein-style Lyapunov estimator + matching loss.

    Args:
        target_lyapunov: λ cible (par sample unit).
        tau_range: (tau_min, tau_max) horizons pour le fit linéaire.
        theiler_window: |i-j| > W exclu (typiquement period moyen ou
            quelques samples). 10 = bon défaut pour séquences ~100+ pts.
        min_anchors: nb min d'anchors valides ; sinon retourne no-op.
        weight: multiplicateur final.
    """

    def __init__(
        self,
        target_lyapunov: float = 0.0,
        tau_range: tuple[int, int] = (1, 8),
        theiler_window: int = 10,
        min_anchors: int = 8,
        weight: float = 1.0,
    ):
        super().__init__()
        self.target = float(target_lyapunov)
        self.tau_min, self.tau_max = tau_range
        self.theiler = theiler_window
        self.min_anchors = min_anchors
        self.weight = weight

    def estimate(self, z: torch.Tensor) -> torch.Tensor:
        """Retourne λ_per_sample estimé via Rosenstein. Différentiable.

        Returns:
            scalar tensor : slope of <ln d(t)> averaged over anchor pairs.
            Si trop court ou pas assez d'anchors valides, retourne 0.0.
        """
        if z.dim() != 2:
            raise ValueError(f"expected (T, D), got shape {tuple(z.shape)}")
        T, D = z.shape
        if T < self.tau_max + self.theiler + 2:
            return z.sum() * 0.0

        # 1) Distance matrix avec Theiler mask + bord
        dist = torch.cdist(z, z)
        time_idx = torch.arange(T, device=z.device)
        diff_time = (time_idx.unsqueeze(0) - time_idx.unsqueeze(1)).abs()
        valid_max = T - self.tau_max
        mask = diff_time <= self.theiler
        mask = mask | (time_idx.unsqueeze(0) >= valid_max)
        mask = mask | (time_idx.unsqueeze(1) >= valid_max)
        dist_m = dist.masked_fill(mask, float("inf"))

        # 2) Pour chaque anchor i valide, nearest neighbor j
        nn_dist, j_idx = dist_m.min(dim=1)
        valid_mask = nn_dist < float("inf")
        if int(valid_mask.sum().item()) < self.min_anchors:
            return z.sum() * 0.0

        anchors = time_idx[valid_mask]
        neighbors = j_idx[valid_mask]

        # 3) Track <ln d(t)> per τ
        log_d_per_tau = []
        for tau in range(self.tau_min, self.tau_max + 1):
            d_tau = torch.norm(
                z[anchors + tau] - z[neighbors + tau], dim=1
            ).clamp_min(1e-12)
            log_d_per_tau.append(torch.log(d_tau).mean())
        log_d = torch.stack(log_d_per_tau)

        # 4) Linear fit slope
        taus = torch.arange(
            self.tau_min, self.tau_max + 1, device=z.device, dtype=z.dtype
        )
        tau_mean = taus.mean()
        ld_mean = log_d.mean()
        cov = ((taus - tau_mean) * (log_d - ld_mean)).sum()
        var = ((taus - tau_mean) ** 2).sum().clamp_min(1e-9)
        return cov / var

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (T, D) trajectoire latente.
        Returns:
            scalar loss = weight * (λ_per_sample_estimé − target_lyapunov)².
        """
        lyap_est = self.estimate(z)
        loss = (lyap_est - self.target) ** 2
        return self.weight * loss
