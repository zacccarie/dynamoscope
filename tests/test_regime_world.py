"""Smoke tests pour regime_world MVP."""
from __future__ import annotations
import numpy as np
import torch

from backend.regime_world import (
    RegimeWorldModel, make_dataset, split_train_eval,
    TrainConfig, train, encode_eval, regime_conditional_loss,
    GENERATORS_PER_REGIME,
)
from backend.regime_world.model import (
    PerceptualEncoder, FastSlowSSM, DescriptorHeads, RegimeRouter,
)


def test_encoder_shape():
    enc = PerceptualEncoder(d_in=3, d_fast=16)
    x = torch.randn(64, 3)
    z = enc(x)
    assert z.shape == (64, 16)


def test_fastslow_shape():
    ssm = FastSlowSSM(d_fast=16, d_slow=8, slow_stride=4)
    z_fast = torch.randn(64, 16)
    f_out, s_out = ssm(z_fast)
    assert f_out.shape == (64, 16)
    # 64 // 4 = 16 slow steps
    assert s_out.shape == (16, 8)


def test_descriptor_heads():
    dh = DescriptorHeads(d_slow=8)
    z_slow = torch.randn(16, 8)
    d = dh(z_slow)
    assert d.shape == (4,)
    assert torch.isfinite(d).all()


def test_regime_router_simplex():
    rr = RegimeRouter(d_d=4, n_regimes=3)
    d = torch.randn(4)
    r = rr(d)
    assert r.shape == (3,)
    assert torch.isclose(r.sum(), torch.tensor(1.0), atol=1e-5)
    assert (r >= 0).all()


def test_full_model_forward():
    model = RegimeWorldModel(d_in=3, d_fast=16, d_slow=8, slow_stride=4)
    x = torch.randn(32, 3)
    out = model(x)
    assert "z_fast" in out
    assert "z_slow" in out
    assert "r" in out
    assert out["z_slow"].shape[1] == 8
    assert out["r"].shape == (3,)
    assert out["x_recon"].shape == (32, 3)


def test_model_params_minimal():
    """MVP must be minimal : < 100K params."""
    model = RegimeWorldModel(d_in=3, d_fast=32, d_slow=32, slow_stride=4)
    n = model.count_params()
    assert n < 100_000, f"too big for MVP: {n}"


def test_regime_loss_finite():
    model = RegimeWorldModel(d_in=3, d_fast=16, d_slow=8, slow_stride=4)
    x = torch.randn(32, 3)
    out = model(x)
    ld = regime_conditional_loss(out, x)
    assert torch.isfinite(ld["total"])
    assert ld["total"].requires_grad


def test_gradient_flow():
    model = RegimeWorldModel(d_in=3, d_fast=16, d_slow=8, slow_stride=4)
    x = torch.randn(32, 3)
    out = model(x)
    ld = regime_conditional_loss(out, x)
    ld["total"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad : {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad : {name}"


def test_synth_zoo_3_regimes_present():
    ds = make_dataset(n_per_regime=4, T=64)
    regimes = set(s.regime for s in ds)
    assert regimes == {"smooth", "periodic", "chaotic"}
    assert len(ds) == 12


def test_synth_traj_shape_consistent():
    ds = make_dataset(n_per_regime=2, T=64)
    for s in ds:
        assert s.traj.shape[0] == 64
        assert s.traj.shape[1] >= 2


def test_train_smoke_short():
    """Train 3 epochs, verify no NaN, loss finite."""
    ds = make_dataset(n_per_regime=3, T=64)
    cfg = TrainConfig(n_epochs=3, lr=3e-4, device="cpu",
                      w_dyn=0.5, w_slow=0.5, w_recur=0.3, w_lyap=0.2)
    model = RegimeWorldModel(d_in=3, d_fast=16, d_slow=8, slow_stride=4)
    # All samples must have matching d_in; pad to 3D if not
    for s in ds:
        if s.traj.shape[1] < 3:
            pad = np.zeros((s.traj.shape[0], 3 - s.traj.shape[1]))
            s.traj = np.concatenate([s.traj, pad], axis=1)
        elif s.traj.shape[1] > 3:
            s.traj = s.traj[:, :3]
    history = train(model, ds, cfg, verbose=False)
    assert len(history) == 3
    assert np.isfinite(history[-1].loss_total)


def test_encode_eval_smoke():
    ds = make_dataset(n_per_regime=2, T=64)
    model = RegimeWorldModel(d_in=3, d_fast=16, d_slow=8, slow_stride=4)
    for s in ds:
        if s.traj.shape[1] < 3:
            pad = np.zeros((s.traj.shape[0], 3 - s.traj.shape[1]))
            s.traj = np.concatenate([s.traj, pad], axis=1)
        elif s.traj.shape[1] > 3:
            s.traj = s.traj[:, :3]
    out = encode_eval(model, ds)
    assert out["z_slow_pooled"].shape[0] == len(ds)
    assert out["r"].shape == (len(ds), 3)
    assert out["regime_true_idx"].shape == (len(ds),)
