"""PCGrad : Projecting Conflicting Gradients (Yu et al. NeurIPS 2020).

Multi-task gradient surgery. Pour chaque paire (g_i, g_j) où i ≠ j :
si cos(g_i, g_j) < 0 → projeter g_i orthogonal à g_j (retirer composante
qui pointe vers gradient conflictuel).

Permet de gérer tug-of-war entre losses incompatibles. Notre cas :
phase_c + distill conflictent (cf §7.7). PCGrad teste si projection
ortho résout le conflit.

Algorithme :
1. Compute gradient g_i for each task
2. Pour chaque g_i :
   - For each g_j (j ≠ i, ordre randomisé) :
     - If cos(g_i, g_j) < 0 : g_i ← g_i − (g_i · g_j / ‖g_j‖²) g_j
3. Sum projected g_i
4. Apply

Référence : Yu, Kumar, Gupta, Levine, Hausman, Finn 2020.
"Gradient Surgery for Multi-Task Learning". NeurIPS.
"""
from __future__ import annotations
import random
import torch


def flatten_grad(params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten all param.grad into single vector. Returns clone."""
    out = []
    for p in params:
        if p.grad is None:
            out.append(torch.zeros_like(p).view(-1))
        else:
            out.append(p.grad.view(-1).clone())
    return torch.cat(out)


def unflatten_grad(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Re-split flat tensor into per-param shape tensors."""
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(flat[offset : offset + n].view_as(p))
        offset += n
    return out


def pcgrad_step(
    losses: list[torch.Tensor],
    params: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    clip_norm: float | None = 1.0,
) -> dict:
    """One PCGrad-projected gradient step.

    Args:
        losses: liste scalar tensors (un par task).
        params: liste params à mettre à jour.
        optimizer: Adam ou autre.
        clip_norm: gradient clipping après projection.
    Returns:
        dict {n_conflicts: nb projections appliquées, total_grad_norm: avant clip}.
    """
    n_tasks = len(losses)
    # 1. Compute flat gradient per task
    flat_grads = []
    for i, loss in enumerate(losses):
        optimizer.zero_grad()
        retain = (i < n_tasks - 1)
        loss.backward(retain_graph=retain)
        flat_grads.append(flatten_grad(params))

    # 2. Project each gradient onto orthogonal complement of conflicting ones
    n_conflicts = 0
    projected = [g.clone() for g in flat_grads]
    for i in range(n_tasks):
        # Random order over other tasks
        order = list(range(n_tasks))
        order.remove(i)
        random.shuffle(order)
        for j in order:
            g_i = projected[i]
            g_j = flat_grads[j]
            dot = (g_i * g_j).sum()
            if dot < 0:
                norm_sq_j = (g_j * g_j).sum().clamp_min(1e-12)
                projected[i] = g_i - (dot / norm_sq_j) * g_j
                n_conflicts += 1

    # 3. Sum projected gradients
    final_flat = torch.stack(projected).sum(dim=0)
    total_norm = float(final_flat.norm().item())

    # 4. Assign to .grad
    unflat = unflatten_grad(final_flat, params)
    for p, g in zip(params, unflat):
        p.grad = g.clone()

    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, clip_norm)

    optimizer.step()
    return {"n_conflicts": n_conflicts, "total_grad_norm": total_norm}
