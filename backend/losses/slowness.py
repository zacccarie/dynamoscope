"""SFASlownessRegularizer : Wiskott-Sejnowski 2002 slowness principle.

Formulation classique SFA :
    minimize ⟨(Δy)²⟩  s.t.  ⟨y⟩ = 0, ⟨y²⟩ = 1, ⟨y_i y_j⟩ = 0 (i ≠ j)

Notre version régularisateur :
    L = mean‖z_{t+1} − z_t‖² / mean(var(z))

Le dénominateur agit comme penalty implicite contre collapse (z = const minimise
numérateur mais aussi dénominateur → ratio diverge).

Use cases :
- Encourager latents à capturer features lentement variant (semantic content,
  scene identity) plutôt que features rapidement variant (motion, noise)
- Force "scene memory" : utile pour world models où état doit persister
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SFASlownessRegularizer(nn.Module):
    """Penalty sur vitesse de variation des latents.

    Args:
        order: 1 = première dérivée Δz, 2 = seconde dérivée Δ²z (acceleration).
        normalize_by_variance: True = divise par var(z) pour éviter collapse.
        weight: poids final.
    """

    def __init__(
        self,
        order: int = 1,
        normalize_by_variance: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        self.order = order
        self.normalize = normalize_by_variance
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
        T = z.shape[0]
        if T < self.order + 2:
            return z.sum() * 0.0

        diff = z[1:] - z[:-1]  # (T-1, D)
        if self.order == 2:
            diff = diff[1:] - diff[:-1]  # (T-2, D)

        sq_norm = (diff ** 2).sum(dim=1).mean()

        if self.normalize:
            var = z.var(dim=0, unbiased=False).mean().clamp_min(1e-9)
            loss = sq_norm / var
        else:
            loss = sq_norm

        return self.weight * loss
