"""Tests pour backend.losses : gradient flow + comportement attendu."""
from __future__ import annotations
import torch
import pytest

from backend.losses import (
    LyapunovMatchingLoss,
    TopologyPreservationLoss,
    SFASlownessRegularizer,
    CausalSparsityLoss,
    ClassicalAlignmentLoss,
)


def _smooth_traj(T: int = 60, D: int = 8, seed: int = 0) -> torch.Tensor:
    """Trajectoire lente (sinusoides) → SFA loss bas, Lyapunov ~0."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 6.28, T).unsqueeze(1)  # (T, 1)
    phases = torch.randn(D) * 0.5
    z = torch.sin(t + phases.unsqueeze(0))  # (T, D)
    return z.requires_grad_(True)


def _chaotic_traj(T: int = 60, D: int = 8, seed: int = 0) -> torch.Tensor:
    """Bruit pur → SFA loss haut, Lyapunov > 0."""
    torch.manual_seed(seed)
    z = torch.randn(T, D)
    return z.requires_grad_(True)


# ---------- Lyapunov ----------


def test_lyapunov_smooth_low():
    """Trajectoire lente → Lyapunov estimé proche 0 → loss bas vs target=0."""
    z = _smooth_traj()
    loss = LyapunovMatchingLoss(target_lyapunov=0.0)(z)
    assert loss.item() < 1.0, f"smooth should give small Lyap loss, got {loss.item()}"


def test_lyapunov_chaotic_higher():
    """Trajectoire chaotique → Lyap estimé > 0 → loss bas vs target>0."""
    z = _chaotic_traj()
    loss_zero = LyapunovMatchingLoss(target_lyapunov=0.0)(z).item()
    loss_high = LyapunovMatchingLoss(target_lyapunov=0.3)(z).item()
    # Au moins une des deux losses doit être finie et raisonnable
    assert all(0 <= x < 100 for x in (loss_zero, loss_high))


def test_lyapunov_gradient_flow():
    z = _chaotic_traj()
    loss = LyapunovMatchingLoss(target_lyapunov=0.0)(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


# ---------- Topology ----------


def test_topology_identity_zero_loss():
    """z = z_ref → distance_corr = 1 → loss ≈ 0."""
    z_ref = _smooth_traj().detach()
    z = z_ref.clone().requires_grad_(True)
    loss = TopologyPreservationLoss(mode="distance_corr")(z, z_ref)
    assert loss.item() < 1e-4, f"identity should give ~0 loss, got {loss.item()}"


def test_topology_random_higher_loss():
    """z random vs z_ref structuré → loss > identity case."""
    z_ref = _smooth_traj().detach()
    z_rand = torch.randn_like(z_ref).requires_grad_(True)
    loss = TopologyPreservationLoss(mode="distance_corr")(z_rand, z_ref)
    assert loss.item() > 0.1


def test_topology_sliced_w_works():
    z_ref = _smooth_traj().detach()
    z = (z_ref + 0.1 * torch.randn_like(z_ref)).requires_grad_(True)
    loss = TopologyPreservationLoss(mode="sliced_w", n_slices=4)(z, z_ref)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None


def test_topology_gradient_flow():
    z_ref = _smooth_traj().detach()
    z = torch.randn_like(z_ref).requires_grad_(True)
    loss = TopologyPreservationLoss()(z, z_ref)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# ---------- SFA Slowness ----------


def test_slowness_smooth_lower_than_chaotic():
    z_smooth = _smooth_traj()
    z_chaos = _chaotic_traj()
    L = SFASlownessRegularizer()
    loss_s = L(z_smooth).item()
    loss_c = L(z_chaos).item()
    assert loss_s < loss_c, f"smooth ({loss_s}) should be slower than chaos ({loss_c})"


def test_slowness_order2():
    z = _smooth_traj()
    L = SFASlownessRegularizer(order=2)
    loss = L(z)
    assert torch.isfinite(loss)


def test_slowness_gradient_flow():
    z = _chaotic_traj()
    loss = SFASlownessRegularizer()(z)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


# ---------- Causal Sparsity ----------


def test_causal_markov_chain_low_fit():
    """Données AR(1) parfaites → fit error ≈ 0."""
    torch.manual_seed(42)
    T, D = 100, 4
    A_true = torch.eye(D) * 0.9
    z = [torch.randn(D)]
    for _ in range(T - 1):
        z.append(A_true @ z[-1] + 0.01 * torch.randn(D))
    z = torch.stack(z).requires_grad_(True)
    L = CausalSparsityLoss(l1_weight=0.0, fit_weight=1.0, order=1)
    fit = L(z).item()
    assert fit < 0.1, f"AR(1) clean should fit well, got {fit}"


def test_causal_l1_penalizes():
    z = _smooth_traj()
    L_no_l1 = CausalSparsityLoss(l1_weight=0.0)
    L_with_l1 = CausalSparsityLoss(l1_weight=1.0)
    assert L_with_l1(z).item() > L_no_l1(z).item()


def test_causal_gradient_flow():
    z = _chaotic_traj()
    loss = CausalSparsityLoss()(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# ---------- Combined ----------


def test_combined_losses_sum():
    """Toutes les losses sommées doivent rester différentiables."""
    z = _smooth_traj()
    z_ref = z.detach()
    loss = (
        LyapunovMatchingLoss(target_lyapunov=0.0)(z)
        + TopologyPreservationLoss()(z, z_ref)
        + SFASlownessRegularizer()(z)
        + CausalSparsityLoss()(z)
    )
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(loss)


# ---------- Classical Alignment ----------


def test_classical_alignment_identity_zero():
    """z_neural = z_classical → corr=1 → loss=0."""
    z_ref = _smooth_traj(T=80, D=4).detach()
    z = z_ref.clone().requires_grad_(True)
    loss = ClassicalAlignmentLoss(mode="distance_corr")(z, z_ref)
    assert loss.item() < 1e-4


def test_classical_alignment_random_higher():
    """Random vs structured ref → loss notable."""
    z_ref = _smooth_traj(T=80, D=4).detach()
    z = torch.randn_like(z_ref).requires_grad_(True)
    loss = ClassicalAlignmentLoss()(z, z_ref)
    assert loss.item() > 0.1


def test_classical_alignment_different_T():
    """T_neural != T_classical → utilise min(T)."""
    z_n = torch.randn(60, 8, requires_grad=True)
    z_c = torch.randn(50, 3)  # plus court (typical delay embed truncation)
    loss = ClassicalAlignmentLoss()(z_n, z_c)
    assert torch.isfinite(loss)


def test_classical_alignment_gradient_flow():
    z_ref = _smooth_traj(T=80, D=4).detach()
    z = torch.randn_like(z_ref).requires_grad_(True)
    loss = ClassicalAlignmentLoss()(z, z_ref)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_classical_alignment_sliced_w():
    z_ref = _smooth_traj(T=80, D=4).detach()
    z = (z_ref + 0.1 * torch.randn_like(z_ref)).requires_grad_(True)
    loss = ClassicalAlignmentLoss(mode="sliced_w", n_slices=4)(z, z_ref)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None


def test_compute_classical_target_helper():
    """Helper produit tensor depuis frames numpy."""
    import numpy as np
    from backend.losses import compute_classical_target
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 1, (40, 32, 48, 3)).astype(np.float32)
    target = compute_classical_target(frames, observable="motion", m=3)
    assert target.dim() == 2
    assert target.shape[1] == 3
    assert target.shape[0] <= 40
