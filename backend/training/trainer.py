"""RSSMTrainer : training loop mini RSSM avec Dynamoscope losses.

Workflow :
1. Sample procedural videos (datasets.py)
2. Encoder(frames) → z trajectory
3. Compute losses :
   - Reconstruction : ‖decoder(z) − frames‖²
   - One-step dynamics : ‖z_{t+1} − dynamics(z_t)‖²
   - Dynamoscope (optional) : SFA + Causal + Lyapunov target
4. Backprop + Adam step

Permettre comparaison ablation : baseline (recon+dyn only) vs +dynamoscope.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .mini_rssm import MiniRSSM
from ..losses import (
    LyapunovMatchingLoss,
    SFASlownessRegularizer,
    CausalSparsityLoss,
    TopologyPreservationLoss,
)


@dataclass
class TrainConfig:
    """Hyperparams pour training mini RSSM."""

    embed_dim: int = 128
    hidden_dim: int = 256
    in_channels: int = 3
    lr: float = 3e-4
    n_epochs: int = 20
    frames_per_clip: int = 32
    # Loss weights (0 désactive)
    w_recon: float = 1.0
    w_dynamics: float = 1.0
    w_slowness: float = 0.0
    w_causal: float = 0.0
    w_lyapunov: float = 0.0
    lyapunov_target: float = 0.0
    w_topology: float = 0.0
    device: str = "auto"  # auto picks mps > cuda > cpu


@dataclass
class EpochStats:
    epoch: int
    loss_total: float
    loss_recon: float
    loss_dynamics: float
    loss_slowness: float
    loss_causal: float
    loss_lyapunov: float
    loss_topology: float


class RSSMTrainer:
    """Trainer pour MiniRSSM. Provide list of (T, C, H, W) tensors as dataset."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)
        self.model = MiniRSSM(
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.slowness = SFASlownessRegularizer(weight=cfg.w_slowness)
        self.causal = CausalSparsityLoss(weight=cfg.w_causal)
        self.lyap = LyapunovMatchingLoss(
            target_lyapunov=cfg.lyapunov_target, weight=cfg.w_lyapunov
        )
        self.topo = TopologyPreservationLoss(weight=cfg.w_topology)
        self.history: list[EpochStats] = []

    @staticmethod
    def _pick_device(spec: str) -> torch.device:
        if spec != "auto":
            return torch.device(spec)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _step(self, frames: torch.Tensor) -> EpochStats:
        """One gradient step on one clip (T, C, H, W)."""
        cfg = self.cfg
        frames = frames.to(self.device)
        z, z_pred, recon = self.model(frames)

        loss_recon = ((recon - frames) ** 2).mean() * cfg.w_recon
        # Dynamics : z_pred[t] should predict z[t+1]
        if z.shape[0] >= 2 and cfg.w_dynamics > 0:
            loss_dyn = ((z_pred[:-1] - z[1:].detach()) ** 2).mean() * cfg.w_dynamics
        else:
            loss_dyn = torch.tensor(0.0, device=self.device)

        # Dynamoscope losses (require z 2D)
        loss_slow = self.slowness(z) if cfg.w_slowness > 0 else torch.tensor(0.0, device=self.device)
        loss_caus = self.causal(z) if cfg.w_causal > 0 else torch.tensor(0.0, device=self.device)
        loss_lyap = self.lyap(z) if cfg.w_lyapunov > 0 else torch.tensor(0.0, device=self.device)
        # Topology requires reference embedding. Use z.detach() initialement,
        # i.e. encourage stability through training plus que matching cible externe.
        # Skip si w_topology = 0
        if cfg.w_topology > 0:
            # Use raw frame pixels as anatomic reference
            ref = frames.view(frames.shape[0], -1)
            loss_topo = self.topo(z, ref)
        else:
            loss_topo = torch.tensor(0.0, device=self.device)

        total = loss_recon + loss_dyn + loss_slow + loss_caus + loss_lyap + loss_topo
        self.optim.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        return EpochStats(
            epoch=-1,
            loss_total=total.item(),
            loss_recon=loss_recon.item(),
            loss_dynamics=loss_dyn.item(),
            loss_slowness=loss_slow.item(),
            loss_causal=loss_caus.item(),
            loss_lyapunov=loss_lyap.item(),
            loss_topology=loss_topo.item(),
        )

    def train(self, clips: list[torch.Tensor], verbose: bool = True) -> list[EpochStats]:
        """Train n_epochs avec dataset clips.

        Args:
            clips: liste tensors (T, C, H, W) [0,1] sur CPU.
            verbose: log per epoch.
        """
        for epoch in range(self.cfg.n_epochs):
            stats_list = []
            for clip in clips:
                s = self._step(clip)
                stats_list.append(s)
            # Average per epoch
            avg = EpochStats(
                epoch=epoch,
                loss_total=float(np.mean([s.loss_total for s in stats_list])),
                loss_recon=float(np.mean([s.loss_recon for s in stats_list])),
                loss_dynamics=float(np.mean([s.loss_dynamics for s in stats_list])),
                loss_slowness=float(np.mean([s.loss_slowness for s in stats_list])),
                loss_causal=float(np.mean([s.loss_causal for s in stats_list])),
                loss_lyapunov=float(np.mean([s.loss_lyapunov for s in stats_list])),
                loss_topology=float(np.mean([s.loss_topology for s in stats_list])),
            )
            self.history.append(avg)
            if verbose:
                print(
                    f"[ep {epoch:02d}] tot={avg.loss_total:.4f}  "
                    f"rec={avg.loss_recon:.4f}  dyn={avg.loss_dynamics:.4f}  "
                    f"slow={avg.loss_slowness:.4f}  caus={avg.loss_causal:.4f}  "
                    f"lyap={avg.loss_lyapunov:.4f}  topo={avg.loss_topology:.4f}"
                )
        return self.history

    @torch.no_grad()
    def encode_video(self, frames_np: np.ndarray) -> np.ndarray:
        """Eval-time : encode video frames (N, H, W, 3) → (N, embed_dim)."""
        self.model.eval()
        t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(self.device)
        if t.max() > 1.5:
            t = t / 255.0
        z = self.model.encoder(t)
        self.model.train()
        return z.cpu().numpy()


def video_to_tensor(frames_np: np.ndarray, target_size: int = 64) -> torch.Tensor:
    """Helper : (N, H, W, 3) numpy → (N, 3, 64, 64) tensor [0,1]."""
    import cv2

    if frames_np.shape[1] != target_size:
        frames_np = np.stack(
            [
                cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_AREA)
                for f in frames_np
            ]
        )
    if frames_np.dtype == np.uint8:
        frames_np = frames_np.astype(np.float32) / 255.0
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    return t
