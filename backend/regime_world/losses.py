"""Regime-conditional losses pour RegimeWorldModel.

Per spec :
  L_total =
      L_recon
      + r_smooth * L_recon_extra (priorité reconstruction stable)
      + r_periodic * (L_slow + L_recurrence)
      + r_chaotic * L_lyap_proxy
      + entropy_reg * H(r)        — anti-collapse

L_slow / L_recurrence / L_lyap_proxy operate on z_slow.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def loss_recon(x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    return ((x_recon - x_target) ** 2).mean()


def loss_dynamics_one_step(z_slow: torch.Tensor, z_slow_pred: torch.Tensor) -> torch.Tensor:
    """Predicted z_t doit approximer z_{t+1} sur z_slow trajectory."""
    if z_slow.shape[0] < 2:
        return z_slow.sum() * 0.0
    return ((z_slow_pred[:-1] - z_slow[1:].detach()) ** 2).mean()


def loss_slowness(z_slow: torch.Tensor) -> torch.Tensor:
    """Standardized slowness : mean ‖Δz_std‖²."""
    if z_slow.shape[0] < 2:
        return z_slow.sum() * 0.0
    z_norm = (z_slow - z_slow.mean(0, keepdim=True)) / z_slow.std(0, keepdim=True).clamp_min(1e-6)
    diff = z_norm[1:] - z_norm[:-1]
    return (diff ** 2).sum(dim=1).mean()


def loss_recurrence(z_slow: torch.Tensor, eps_quantile: float = 0.2,
                    target_DET: float = 0.7) -> torch.Tensor:
    """Encourage récurrence forte (DET élevé) via soft recurrence matrix.

    Used for periodic regime. Cycle = high diagonal recurrence.
    """
    T_s = z_slow.shape[0]
    if T_s < 8:
        return z_slow.sum() * 0.0
    d_ij = torch.cdist(z_slow, z_slow)
    # Theiler window 2
    mask_offdiag = torch.ones(T_s, T_s, device=z_slow.device, dtype=torch.bool)
    for k in range(-2, 3):
        mask_offdiag = mask_offdiag & ~torch.eye(T_s, dtype=torch.bool,
                                                 device=z_slow.device).roll(k, dims=0)
    eps_val = torch.quantile(d_ij[mask_offdiag], eps_quantile)
    R = torch.sigmoid((eps_val - d_ij) / (0.3 * eps_val))
    # Soft DET : weighted sum on near-diagonals offsets ≥ 2
    det_score = 0.0
    n_diags = 0
    for off in range(2, min(T_s // 2, 16)):
        diag = torch.diagonal(R, offset=off)
        # Mean of diag = average recurrence on this off-diag
        det_score = det_score + diag.mean()
        n_diags += 1
    det_score = det_score / max(n_diags, 1)
    # Distance from target
    return (det_score - target_DET) ** 2


def loss_lyap_proxy(z_slow: torch.Tensor, tau: int = 3,
                    target_log_growth: float = 0.0) -> torch.Tensor:
    """Light Lyapunov-style proxy : log-growth of nearest-pair distances.

    For chaotic regime, we WANT positive log-growth (divergence) — but per
    spec keep this simple, not full Rosenstein. Just penalize collapse
    toward constant trajectory (negative growth).
    """
    T_s = z_slow.shape[0]
    if T_s < tau + 4:
        return z_slow.sum() * 0.0
    d_ij = torch.cdist(z_slow[:-tau], z_slow[:-tau])
    # Theiler diag fill
    mask = torch.eye(d_ij.shape[0], dtype=torch.bool, device=z_slow.device)
    d_ij = d_ij.masked_fill(mask, float("inf"))
    nn_d0, nn_idx = d_ij.min(dim=1)
    valid = nn_d0 < float("inf")
    if valid.sum() < 4:
        return z_slow.sum() * 0.0
    anchors = torch.arange(z_slow.shape[0] - tau, device=z_slow.device)[valid]
    neighbors = nn_idx[valid]
    d0 = nn_d0[valid].clamp_min(1e-9)
    d_tau = torch.norm(z_slow[anchors + tau] - z_slow[neighbors + tau], dim=1).clamp_min(1e-9)
    log_growth = (torch.log(d_tau) - torch.log(d0)).mean()
    return (log_growth - target_log_growth) ** 2


def loss_multistep_rollout(z_slow: torch.Tensor, dynamics, r: torch.Tensor,
                            K: int = 4, gamma: float = 0.9,
                            n_starts: int = 4) -> torch.Tensor:
    """K-step rollout loss : itérer dynamics K fois et matcher futur z_slow.

    Addresses F10 : 1-step training ne contraint pas la stabilité long horizon.
    Pick `n_starts` random offsets in z_slow, roll out K steps from each,
    compare predicted to ground truth with γ^k discount.

    z_slow : (T_s, D_slow) ground-truth latents
    dynamics : RegimeMixtureDynamics instance
    r : (3,) regime distribution from router
    K : rollout horizon
    gamma : discount per step (≤ 1, late steps weighted less to stabilize grad)
    n_starts : how many random start points per sample
    """
    T_s = z_slow.shape[0]
    if T_s < K + 2:
        return z_slow.sum() * 0.0
    max_start = T_s - K - 1
    # Determinist starts if max_start small; otherwise random
    if max_start <= n_starts:
        starts = list(range(max_start + 1))
    else:
        starts = torch.randint(0, max_start + 1, (n_starts,)).tolist()
    total = z_slow.sum() * 0.0
    for start in starts:
        z_pred = z_slow[start].unsqueeze(0)  # (1, D)
        loss_chain = z_slow.sum() * 0.0
        for k in range(K):
            z_pred = dynamics(z_pred, r)  # (1, D)
            target = z_slow[start + 1 + k].unsqueeze(0).detach()
            step_loss = ((z_pred - target) ** 2).mean()
            loss_chain = loss_chain + (gamma ** k) * step_loss
        total = total + loss_chain / K
    return total / max(len(starts), 1)


def loss_load_balance(r_batch: torch.Tensor) -> torch.Tensor:
    """Shazeer 2017 (simplified) load-balance pour K > 3 unsupervised experts.

    r_batch : (B, K) router outputs (softmax weights) per sample.
    Minimise K * ||mean_gate||² ; min = 1 si chaque expert utilisé à 1/K.
    Plus elevé = un expert monopolise tout (collapse).
    """
    if r_batch.dim() == 1:
        r_batch = r_batch.unsqueeze(0)
    f = r_batch.mean(dim=0)  # (K,) — gate fraction par expert
    K = f.shape[0]
    return K * (f * f).sum()


def loss_regime_supervised(r: torch.Tensor, regime_idx: int) -> torch.Tensor:
    """Cross-entropy : router output vs ground-truth regime label.

    r : (3,) softmax distribution.
    regime_idx : 0=smooth, 1=periodic, 2=chaotic.

    Used in semi-supervised training to break router collapse.
    """
    p = r.clamp_min(1e-9)
    target = torch.tensor(regime_idx, device=r.device)
    log_p = p.log()
    return -log_p[target]


def loss_entropy_regularizer(r: torch.Tensor, target_entropy: float = 0.8
                              ) -> torch.Tensor:
    """Encourage non-degenerate regime distribution.

    target_entropy ~ 0.8 (over 3 regimes, max ln(3) ≈ 1.10).
    Push H(r) toward target — too high = always uniform, too low = collapse.
    """
    p = r.clamp_min(1e-9)
    H = -(p * p.log()).sum()
    return (H - target_entropy) ** 2


def regime_conditional_loss(
    out: dict, x: torch.Tensor,
    w_recon: float = 1.0,
    w_dyn: float = 1.0,
    w_slow: float = 1.0,
    w_recur: float = 1.0,
    w_lyap: float = 1.0,
    w_entropy: float = 0.1,
    w_regime_sup: float = 0.0,
    w_multistep: float = 0.0,
    K_multistep: int = 4,
    gamma_multistep: float = 0.9,
    n_starts_multistep: int = 4,
    w_load_balance: float = 0.0,
    dynamics_module=None,
    regime_idx: int | None = None,
    target_log_growth_chaotic: float = 0.1,
    target_DET_periodic: float = 0.7,
) -> dict:
    """Sum of regime-conditional terms.

    Returns dict {total, recon, dyn, slow_weighted, recur_weighted, lyap_weighted,
    entropy_reg, r}.
    """
    r = out["r"]  # (K,) — K=3 (taxonomic) ou K>3 (unsupervised MoE)
    K = r.shape[-1] if r.dim() > 0 else 3
    taxonomic = (K == 3)

    l_recon = loss_recon(out["x_recon"], x) * w_recon
    l_dyn = loss_dynamics_one_step(out["z_slow"], out["z_slow_pred"]) * w_dyn

    if taxonomic:
        r_smooth, r_periodic, r_chaotic = r[0], r[1], r[2]
        # Slowness : weight by (r_smooth + r_periodic), penalize when system is
        # smooth or periodic (both prefer slow latents)
        l_slow_raw = loss_slowness(out["z_slow"])
        l_slow = (r_smooth + r_periodic) * l_slow_raw * w_slow
        # Recurrence : weighted by r_periodic
        l_recur_raw = loss_recurrence(out["z_slow"], target_DET=target_DET_periodic)
        l_recur = r_periodic * l_recur_raw * w_recur
        # Lyap proxy : weighted by r_chaotic
        l_lyap_raw = loss_lyap_proxy(out["z_slow"],
                                      target_log_growth=target_log_growth_chaotic)
        l_lyap = r_chaotic * l_lyap_raw * w_lyap
    else:
        # K > 3 : aucune correspondence régime ↔ expert. Pas de gated losses.
        # Le modèle apprend la spécialisation purement via dyn + recon + ms.
        zero = r.sum() * 0.0
        l_slow = zero; l_recur = zero; l_lyap = zero

    # Entropy regularizer (anti-collapse). Vise distribution non-dégenérée.
    l_entropy = loss_entropy_regularizer(r) * w_entropy

    # Optional supervised regime CE (semi-supervised mode, K=3 only)
    if taxonomic and w_regime_sup > 0 and regime_idx is not None:
        l_regime_sup = loss_regime_supervised(r, regime_idx) * w_regime_sup
    else:
        l_regime_sup = r.sum() * 0.0  # zero, keeps grad graph

    # Load balance (Shazeer 2017, simplified) — pour K > 3 MoE-style.
    # r est (K,) ici (un sample); le loss attend (B, K) idéalement mais
    # accepte single sample via unsqueeze. Effet faible sur 1 sample, l'agrégat
    # batch-level émerge via accumulation epoch-wise du gradient.
    if w_load_balance > 0:
        l_load = loss_load_balance(r) * w_load_balance
    else:
        l_load = r.sum() * 0.0

    # Multi-step rollout loss (F10 fix). Requires dynamics_module to iterate.
    if w_multistep > 0 and dynamics_module is not None:
        l_multistep_raw = loss_multistep_rollout(
            out["z_slow"], dynamics_module, r,
            K=K_multistep, gamma=gamma_multistep,
            n_starts=n_starts_multistep,
        )
        l_multistep = w_multistep * l_multistep_raw
    else:
        l_multistep = r.sum() * 0.0

    total = (l_recon + l_dyn + l_slow + l_recur + l_lyap + l_entropy
             + l_regime_sup + l_multistep + l_load)
    # Pour l'export r_smooth/periodic/chaotic : K=3 garde la semantique ;
    # K>3 reporte les 3 premières gates (juste pour télémétrie, pas sémantique).
    if taxonomic:
        r0, r1, r2 = float(r[0].item()), float(r[1].item()), float(r[2].item())
    else:
        r0 = float(r[0].item())
        r1 = float(r[1].item()) if K > 1 else 0.0
        r2 = float(r[2].item()) if K > 2 else 0.0
    return {
        "total": total,
        "recon": l_recon,
        "dyn": l_dyn,
        "slow_weighted": l_slow,
        "recur_weighted": l_recur,
        "lyap_weighted": l_lyap,
        "entropy_reg": l_entropy,
        "regime_sup": l_regime_sup,
        "multistep": l_multistep,
        "load_balance": l_load,
        "r_smooth": r0,
        "r_periodic": r1,
        "r_chaotic": r2,
    }
