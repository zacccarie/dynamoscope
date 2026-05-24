"""Three-stage trainer pour RegimeWorldModel.

Stage 1 (warmup, ~30% epochs) : recon + dyn only.
Stage 2 (~30% epochs)         : + slow + recurrence (uniformly weighted).
Stage 3 (~40% epochs)         : + regime-conditional weights actifs.

Per spec : avoid joint optimization from epoch 0 to prevent collapse.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from .model import RegimeWorldModel
from .losses import regime_conditional_loss


@dataclass
class TrainConfig:
    n_epochs: int = 60
    lr: float = 3e-4
    device: str = "cpu"  # mps adds overhead for tiny models
    w_dyn: float = 1.0
    w_slow: float = 1.0
    w_recur: float = 0.5
    w_lyap: float = 0.3
    w_entropy: float = 0.1
    w_regime_sup: float = 0.0  # 0 = unsupervised, > 0 = semi-supervised
    target_log_growth_chaotic: float = 0.1
    target_DET_periodic: float = 0.7


@dataclass
class EpochStats:
    epoch: int
    stage: int
    loss_total: float
    loss_recon: float
    loss_dyn: float
    loss_slow: float
    loss_recur: float
    loss_lyap: float
    loss_entropy: float
    mean_r_smooth: float
    mean_r_periodic: float
    mean_r_chaotic: float


def stage_weights(epoch: int, n_epochs: int, cfg: TrainConfig) -> dict:
    """Schedule loss weights by stage."""
    s1 = int(0.3 * n_epochs)
    s2 = int(0.6 * n_epochs)
    # Supervised regime weight : ramp up from stage 1 (if enabled)
    w_sup = cfg.w_regime_sup
    if epoch < s1:
        # recon + dyn + (sup if any) — warmup
        return {"w_dyn": cfg.w_dyn, "w_slow": 0.0, "w_recur": 0.0,
                "w_lyap": 0.0, "w_entropy": cfg.w_entropy * 0.5,
                "w_regime_sup": w_sup}
    elif epoch < s2:
        return {"w_dyn": cfg.w_dyn, "w_slow": cfg.w_slow,
                "w_recur": cfg.w_recur, "w_lyap": cfg.w_lyap * 0.3,
                "w_entropy": cfg.w_entropy, "w_regime_sup": w_sup}
    else:
        return {"w_dyn": cfg.w_dyn, "w_slow": cfg.w_slow,
                "w_recur": cfg.w_recur, "w_lyap": cfg.w_lyap,
                "w_entropy": cfg.w_entropy, "w_regime_sup": w_sup}


def train_one_epoch(model, dataset, optim, weights, cfg) -> dict:
    """Train one epoch over a list of TrajSample. Returns aggregated stats."""
    from .synth import REGIME_TO_IDX
    model.train()
    device = next(model.parameters()).device
    accum = {k: 0.0 for k in ["total", "recon", "dyn", "slow_weighted",
                              "recur_weighted", "lyap_weighted", "entropy_reg",
                              "regime_sup"]}
    rs_accum = [0.0, 0.0, 0.0]
    n = 0
    for sample in dataset:
        x = torch.from_numpy(sample.traj).float().to(device)
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        out = model(x)
        regime_idx = REGIME_TO_IDX[sample.regime]
        loss_dict = regime_conditional_loss(
            out, x,
            w_recon=1.0, w_dyn=weights["w_dyn"],
            w_slow=weights["w_slow"], w_recur=weights["w_recur"],
            w_lyap=weights["w_lyap"], w_entropy=weights["w_entropy"],
            w_regime_sup=weights.get("w_regime_sup", 0.0),
            regime_idx=regime_idx,
            target_log_growth_chaotic=cfg.target_log_growth_chaotic,
            target_DET_periodic=cfg.target_DET_periodic,
        )
        optim.zero_grad()
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        for k in accum:
            v = loss_dict[k]
            accum[k] += float(v.item() if hasattr(v, "item") else v)
        rs_accum[0] += loss_dict["r_smooth"]
        rs_accum[1] += loss_dict["r_periodic"]
        rs_accum[2] += loss_dict["r_chaotic"]
        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()} | {
        "r_smooth": rs_accum[0] / max(n, 1),
        "r_periodic": rs_accum[1] / max(n, 1),
        "r_chaotic": rs_accum[2] / max(n, 1),
    }


def train(model: RegimeWorldModel, dataset, cfg: TrainConfig,
          verbose: bool = True) -> list[EpochStats]:
    device = torch.device(cfg.device)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.n_epochs):
        w = stage_weights(epoch, cfg.n_epochs, cfg)
        s1 = int(0.3 * cfg.n_epochs)
        s2 = int(0.6 * cfg.n_epochs)
        stage = 1 if epoch < s1 else (2 if epoch < s2 else 3)
        stats = train_one_epoch(model, dataset, optim, w, cfg)
        es = EpochStats(
            epoch=epoch, stage=stage,
            loss_total=stats["total"], loss_recon=stats["recon"],
            loss_dyn=stats["dyn"], loss_slow=stats["slow_weighted"],
            loss_recur=stats["recur_weighted"], loss_lyap=stats["lyap_weighted"],
            loss_entropy=stats["entropy_reg"],
            mean_r_smooth=stats["r_smooth"], mean_r_periodic=stats["r_periodic"],
            mean_r_chaotic=stats["r_chaotic"],
        )
        history.append(es)
        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"[ep {epoch:02d} stg{stage}] tot={es.loss_total:.4f}  "
                  f"rec={es.loss_recon:.4f}  slow={es.loss_slow:.4f}  "
                  f"recur={es.loss_recur:.4f}  lyap={es.loss_lyap:.4f}  "
                  f"r=({es.mean_r_smooth:.2f},{es.mean_r_periodic:.2f},"
                  f"{es.mean_r_chaotic:.2f})")
    return history


@torch.no_grad()
def encode_eval(model: RegimeWorldModel, dataset) -> dict:
    """Eval-time : encode chaque sample, return latents + regime predictions."""
    model.eval()
    device = next(model.parameters()).device
    out = {"z_slow_pooled": [], "d_dyn": [], "r": [],
           "regime_true": [], "regime_true_idx": [], "sample_idx": []}
    from .synth import REGIME_TO_IDX
    for i, sample in enumerate(dataset):
        x = torch.from_numpy(sample.traj).float().to(device)
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        o = model(x)
        out["z_slow_pooled"].append(o["z_slow"].mean(dim=0).cpu().numpy())
        out["d_dyn"].append(o["d_dyn"].cpu().numpy())
        out["r"].append(o["r"].cpu().numpy())
        out["regime_true"].append(sample.regime)
        out["regime_true_idx"].append(REGIME_TO_IDX[sample.regime])
        out["sample_idx"].append(i)
    out["z_slow_pooled"] = np.stack(out["z_slow_pooled"])
    out["d_dyn"] = np.stack(out["d_dyn"])
    out["r"] = np.stack(out["r"])
    out["regime_true_idx"] = np.array(out["regime_true_idx"])
    return out
