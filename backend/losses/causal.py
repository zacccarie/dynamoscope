"""CausalSparsityLoss : L1 sur matrice autorégressive Granger.

Vraie causalité (transfer entropy KSG) non-diff. Proxy linéaire :

1. Fit z_{t+1} ≈ A z_t + b via least-squares (closed-form, diff)
2. Loss combine :
   - Fit error : ‖z_{t+1} − (A z_t + b)‖²  (encourage Markov 1st order)
   - L1 penalty : λ ‖A‖_1   (sparse causal graph)

Sparse A signifie : peu de variables latentes influencent peu d'autres
→ structure causale lisible.

Use cases :
- Encoder pour Dynamoscope causal panel : sparse A donne PCMCI cleaner
- World models : impose Markovianité d'ordre 1 dans latent space
- Disentanglement : sparse A favorise indépendance entre dimensions
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CausalSparsityLoss(nn.Module):
    """Penalty sparse autorégressif sur trajectoire latente.

    Args:
        l1_weight: poids du terme L1 sur A.
        fit_weight: poids du terme erreur de fit Markovien.
        order: ordre autorégressif (1 = z_{t+1} ~ z_t).
        weight: poids global.
    """

    def __init__(
        self,
        l1_weight: float = 0.01,
        fit_weight: float = 1.0,
        order: int = 1,
        weight: float = 1.0,
    ):
        super().__init__()
        if order < 1:
            raise ValueError(f"order >= 1 required, got {order}")
        self.l1_weight = l1_weight
        self.fit_weight = fit_weight
        self.order = order
        self.weight = weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (T, D) trajectoire latente.
        Returns:
            scalar loss différentiable.
        """
        if z.dim() != 2:
            raise ValueError(f"expected (T, D), got shape {tuple(z.shape)}")
        T, D = z.shape
        if T < self.order + 2:
            return z.sum() * 0.0

        # Build lag matrix X = [z_{t-order+1}, ..., z_t] et target Y = z_{t+1}
        # Pour order=1 : X = z[:-1], Y = z[1:]
        X_blocks = []
        for lag in range(self.order):
            X_blocks.append(z[lag : T - self.order + lag])
        X = torch.cat(X_blocks, dim=1)  # (T-order, D*order)
        Y = z[self.order :]  # (T-order, D)

        # Add bias column
        ones = torch.ones(X.shape[0], 1, device=z.device, dtype=z.dtype)
        Xb = torch.cat([X, ones], dim=1)  # (T-order, D*order+1)

        # Least-squares closed-form via torch.linalg.lstsq (differentiable backwards)
        # Beta : (D*order+1, D)
        result = torch.linalg.lstsq(Xb, Y)
        beta = result.solution  # may be (D*order+1, D) or padded; handle both
        if beta.shape[0] != Xb.shape[1]:
            beta = beta[: Xb.shape[1]]

        Y_pred = Xb @ beta
        fit_err = ((Y - Y_pred) ** 2).mean()

        # L1 sur matrice A (exclure terme de bias dernière ligne)
        A = beta[:-1]  # (D*order, D)
        l1 = A.abs().mean()

        loss = self.fit_weight * fit_err + self.l1_weight * l1
        return self.weight * loss
