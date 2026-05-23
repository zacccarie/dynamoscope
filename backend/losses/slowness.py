"""SFASlownessRegularizer : Wiskott-Sejnowski 2002 slowness principle.

Formulation classique SFA :
    minimize ⟨(Δy)²⟩  s.t.  ⟨y⟩ = 0, ⟨y²⟩ = 1

Notre version régularisateur :

    mode="standardize" (default) :
        z_std = (z − mean(z)) / std(z)   per-dim
        L = mean‖Δ z_std‖²
        → invariant à scale et offset, var(z) ne peut PAS croître pour
        échapper à la pénalité (échec mode connu de "ratio").

    mode="ratio" (legacy) :
        L = mean‖Δz‖² / mean(var(z))
        → simple mais peut être contourné en gonflant var(z).

Use cases :
- Encourager latents capturant features lentement variant (scene identity,
  semantic content) plutôt que rapide (motion).
- "Memory" pour world models où état doit persister.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SFASlownessRegularizer(nn.Module):
    """Penalty sur vitesse de variation des latents.

    Args:
        order: 1 = première dérivée Δz, 2 = seconde dérivée Δ²z.
        mode: "standardize" (default, robust) | "ratio" (legacy).
        weight: poids final.
    """

    def __init__(
        self,
        order: int = 1,
        mode: str = "standardize",
        weight: float = 1.0,
        normalize_by_variance: bool | None = None,  # legacy kwarg
    ):
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        # Legacy compat
        if normalize_by_variance is not None:
            mode = "ratio" if normalize_by_variance else "raw"
        if mode not in ("standardize", "ratio", "raw"):
            raise ValueError(f"mode must be standardize|ratio|raw, got {mode}")
        self.order = order
        self.mode = mode
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

        if self.mode == "standardize":
            # Per-dim mean-0 var-1 normalization. var(z) growth ne peut plus
            # réduire la pénalité — projection robust sur sphère unitaire.
            mean = z.mean(dim=0, keepdim=True)
            std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
            z_use = (z - mean) / std
        else:
            z_use = z

        diff = z_use[1:] - z_use[:-1]
        if self.order == 2:
            diff = diff[1:] - diff[:-1]

        sq_norm = (diff ** 2).sum(dim=1).mean()

        if self.mode == "ratio":
            var = z.var(dim=0, unbiased=False).mean().clamp_min(1e-9)
            sq_norm = sq_norm / var

        return self.weight * sq_norm
