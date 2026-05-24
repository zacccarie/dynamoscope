"""RegimeWorldModel — MVP Phase 1.

State : z_t = (z_fast_t, z_slow_t, d_t_dyn, r_t)

Components :
1. Encoder      : raw_traj (T, d_in) → z_fast (T, D_fast)
                  (for synth zoo : MLP wrapper over raw obs ; for video :
                  swap in frozen DINOv2 / V-JEPA)
2. FastSlow SSM : GRU fast (per-step) + GRU slow (every k steps)
3. DescriptorHeads : (slowness, recurrence_score, lyap_proxy) from windowed z_slow
4. RegimeRouter : MLP(d_dyn) → softmax Δ³ over {smooth, periodic, chaotic}
5. RegimeMixture Dynamics : 3 transition modules Φ_k mixed by r_t

Output :
    z_fast (T, D_fast)
    z_slow (T_s, D_slow)
    d_dyn  (T_s, D_d)
    r      (T_s, 3)
    z_slow_pred (T_s, D_slow) — predicted next-step under regime mixture

Design constraint (per spec) : MINIMAL. Small MLPs, no Mamba/S5 yet
(reduce risk, no external deps). Each Phi_k = 2-layer MLP.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


REGIMES = ["smooth", "periodic", "chaotic"]


class PerceptualEncoder(nn.Module):
    """MLP over raw obs (for synth). For video : replace by frozen V-JEPA wrapper."""

    def __init__(self, d_in: int, d_fast: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, d_fast),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (T, d_in) → (T, D_fast)
        return self.net(x)


class FastSlowSSM(nn.Module):
    """Fast GRU per-step + Slow GRU every k steps.

    Slow is the regime-sensitive representation per spec.
    """

    def __init__(self, d_fast: int = 32, d_slow: int = 32,
                 slow_stride: int = 4):
        super().__init__()
        self.fast_cell = nn.GRUCell(d_fast, d_fast)
        # Slow takes pooled fast over window of `slow_stride` steps
        self.slow_cell = nn.GRUCell(d_fast, d_slow)
        self.slow_stride = slow_stride
        self.d_fast = d_fast
        self.d_slow = d_slow

    def forward(self, z_fast_seq: torch.Tensor):
        """
        z_fast_seq : (T, D_fast)
        Returns :
          z_fast_hidden  (T, D_fast)
          z_slow         (T_s, D_slow) with T_s = T // stride
        """
        T = z_fast_seq.shape[0]
        device = z_fast_seq.device
        h_fast = torch.zeros(self.d_fast, device=device)
        h_slow = torch.zeros(self.d_slow, device=device)
        fast_out = []
        slow_out = []
        fast_window = []
        for t in range(T):
            h_fast = self.fast_cell(z_fast_seq[t:t + 1], h_fast.unsqueeze(0)).squeeze(0)
            fast_out.append(h_fast)
            fast_window.append(h_fast)
            if (t + 1) % self.slow_stride == 0:
                pooled = torch.stack(fast_window, dim=0).mean(dim=0)
                h_slow = self.slow_cell(pooled.unsqueeze(0), h_slow.unsqueeze(0)).squeeze(0)
                slow_out.append(h_slow)
                fast_window = []
        if not slow_out:
            slow_out.append(h_slow)
        return torch.stack(fast_out, dim=0), torch.stack(slow_out, dim=0)


class DescriptorHeads(nn.Module):
    """Compute lightweight dynamical descriptors from windowed slow latent.

    Output : (D_d,) per window.
    - slowness : mean ‖Δz_slow‖² (low = slow)
    - recur_score : mean pairwise prox (high = recurrent)
    - lyap_proxy : log-divergence of two perturbed trajectories (proxy)
    - var_proxy : trace of covariance (energy)
    """

    def __init__(self, d_slow: int = 32):
        super().__init__()
        self.d_d = 4  # 4 descriptors

    def forward(self, z_slow: torch.Tensor) -> torch.Tensor:
        """
        z_slow : (T_s, D_slow)
        Returns d_dyn : (D_d,) — global per-trajectory summary
                (broadcast back to T_s if needed downstream)
        """
        T_s, D = z_slow.shape
        if T_s < 2:
            return torch.zeros(4, device=z_slow.device, dtype=z_slow.dtype)
        # slowness
        diff = z_slow[1:] - z_slow[:-1]
        slowness = (diff ** 2).sum(dim=1).mean()
        # recurrence proxy : pairwise soft proximity
        d_ij = torch.cdist(z_slow, z_slow)
        eps = d_ij.mean().clamp_min(1e-6)
        R = torch.sigmoid((eps - d_ij) / (0.5 * eps))
        # Mask diagonal + temporal neighbors
        mask = torch.ones_like(R)
        for k in range(-2, 3):
            mask = mask - torch.diag(torch.ones(T_s - abs(k), device=R.device), diagonal=k)
        mask = mask.clamp_min(0)
        recur_score = (R * mask).sum() / mask.sum().clamp_min(1)
        # lyap proxy : pairwise distance growth correlation with time-lag
        # Approx : variance of log-distance (chaos → high variance)
        lyap_proxy = torch.log(d_ij + 1e-6).var()
        # var proxy
        var_proxy = z_slow.var(dim=0).sum()
        return torch.stack([slowness, recur_score, lyap_proxy, var_proxy])


class RegimeRouter(nn.Module):
    """MLP(d_dyn) → softmax over 3 regimes.

    Anti-collapse : entropy regularizer added in loss.
    """

    def __init__(self, d_d: int = 4, hidden: int = 32, n_regimes: int = 3,
                 temperature: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_d, hidden), nn.SiLU(),
            nn.Linear(hidden, n_regimes),
        )
        self.temperature = temperature

    def forward(self, d_dyn: torch.Tensor) -> torch.Tensor:
        logits = self.net(d_dyn)
        return F.softmax(logits / self.temperature, dim=-1)


class TransitionModule(nn.Module):
    """Single regime-specific transition Φ_k : z_t → z_{t+1}."""

    def __init__(self, d_slow: int = 32, hidden: int = 64,
                 init_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_slow, hidden), nn.SiLU(),
            nn.Linear(hidden, d_slow),
        )
        # Different init scale for different regimes (inductive bias)
        with torch.no_grad():
            for p in self.net.parameters():
                p.mul_(init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Residual : Φ(z) = z + delta
        return z + self.net(z)


class RegimeMixtureDynamics(nn.Module):
    """Mix 3 transitions weighted by regime distribution r."""

    def __init__(self, d_slow: int = 32, n_regimes: int = 3):
        super().__init__()
        # Inductive biases per regime :
        #   smooth   : small delta (init small)
        #   periodic : medium with rotational tendency
        #   chaotic  : larger delta
        scales = [0.3, 0.6, 1.0]
        self.transitions = nn.ModuleList([
            TransitionModule(d_slow=d_slow, init_scale=scales[k])
            for k in range(n_regimes)
        ])

    def forward(self, z_slow: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        z_slow : (T_s, D_slow)
        r      : (3,) regime distribution (global) OR (T_s, 3) per-step
        Returns z_pred : (T_s, D_slow)  = mixture transition
        """
        preds = torch.stack([phi(z_slow) for phi in self.transitions], dim=0)
        # preds : (n_regimes, T_s, D_slow)
        if r.dim() == 1:
            r_b = r.view(-1, 1, 1)  # broadcast
            return (preds * r_b).sum(dim=0)
        else:
            # r : (T_s, n_regimes) → (n_regimes, T_s, 1)
            r_b = r.transpose(0, 1).unsqueeze(-1)
            return (preds * r_b).sum(dim=0)


class RegimeWorldModel(nn.Module):
    """Assembled : encoder + fast/slow SSM + descriptors + router + mixture dynamics."""

    def __init__(self, d_in: int = 3, d_fast: int = 32, d_slow: int = 32,
                 slow_stride: int = 4, n_regimes: int = 3,
                 router_temperature: float = 1.0):
        super().__init__()
        self.encoder = PerceptualEncoder(d_in, d_fast)
        self.ssm = FastSlowSSM(d_fast, d_slow, slow_stride)
        self.desc_heads = DescriptorHeads(d_slow)
        self.router = RegimeRouter(d_d=4, n_regimes=n_regimes,
                                    temperature=router_temperature)
        self.dynamics = RegimeMixtureDynamics(d_slow, n_regimes)
        self.d_in = d_in
        self.d_fast = d_fast
        self.d_slow = d_slow
        # Decoder for reconstruction baseline
        self.decoder = nn.Sequential(
            nn.Linear(d_fast, 64), nn.SiLU(),
            nn.Linear(64, d_in),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (T, d_in) raw observable trajectory
        Returns dict with z_fast, z_slow, d_dyn, r, z_slow_pred, x_recon
        """
        z_fast_enc = self.encoder(x)              # (T, D_fast)
        z_fast, z_slow = self.ssm(z_fast_enc)     # (T, D_fast), (T_s, D_slow)
        d_dyn = self.desc_heads(z_slow)           # (4,)
        r = self.router(d_dyn)                    # (3,)
        z_slow_pred = self.dynamics(z_slow, r)    # (T_s, D_slow)
        x_recon = self.decoder(z_fast_enc)        # (T, d_in)
        return {
            "z_fast": z_fast,
            "z_slow": z_slow,
            "d_dyn": d_dyn,
            "r": r,
            "z_slow_pred": z_slow_pred,
            "x_recon": x_recon,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
