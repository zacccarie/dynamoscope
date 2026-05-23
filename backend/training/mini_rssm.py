"""Mini RSSM : CNN encoder + GRU dynamics + transposed CNN decoder.

Deterministic version Dreamer-V1 (Hafner 2019) sans stochastic state,
sans reward predictor. Goal : minimal trainable world model pour démo
feedback loop avec Dynamoscope losses.

Architecture (64×64 grayscale ou 64×64×3 RGB) :
    Encoder : 4 conv strided (stride 2) → flatten → linear → z (128d)
    Dynamics : GRU(z_dim → hidden_dim) → next_z via linear
    Decoder : linear → 4 deconv strided (stride 2) → reconstructed frame

Params count : ~500K (vs Dreamer-V3 ~200M) → trainable sur CPU/MPS.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniRSSMEncoder(nn.Module):
    """CNN encoder 64×64×C → embed_dim."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        # 64 → 32 → 16 → 8 → 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(256 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 64, 64) → (B, embed_dim)
        h = self.conv(x)
        return self.proj(self.flatten(h))


class MiniRSSMDecoder(nn.Module):
    """Transposed CNN decoder : embed_dim → 64×64×C reconstruction."""

    def __init__(self, out_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # [0,1] images
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class MiniRSSMDynamics(nn.Module):
    """GRU-based latent dynamics : z_t → z_{t+1} prediction."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self.proj_z = nn.Linear(hidden_dim, embed_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self, z_seq: torch.Tensor, h0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            z_seq: (T, embed_dim) latent sequence
            h0: optional initial hidden state (hidden_dim,)
        Returns:
            z_pred: (T, embed_dim) one-step-ahead predictions
                    z_pred[t] = f(z_seq[t-1], h_{t-1})
        """
        T = z_seq.shape[0]
        h = (
            h0
            if h0 is not None
            else torch.zeros(self.hidden_dim, device=z_seq.device, dtype=z_seq.dtype)
        )
        z_preds = []
        for t in range(T):
            h = self.gru(z_seq[t].unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            z_preds.append(self.proj_z(h))
        return torch.stack(z_preds, dim=0)


class MiniRSSM(nn.Module):
    """Mini Dreamer-style RSSM : encoder + dynamics + decoder.

    Args:
        in_channels: 3 RGB, 1 grayscale
        embed_dim: latent dim z
        hidden_dim: GRU hidden state dim
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = MiniRSSMEncoder(in_channels, embed_dim)
        self.dynamics = MiniRSSMDynamics(embed_dim, hidden_dim)
        self.decoder = MiniRSSMDecoder(in_channels, embed_dim)
        self.embed_dim = embed_dim

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, C, H, W) → z: (T, embed_dim)"""
        return self.encoder(frames)

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pass : encode → dynamics → decode.

        Args:
            frames: (T, C, H, W) input video
        Returns:
            z: (T, embed_dim) encoded latents
            z_pred: (T, embed_dim) one-step dynamics predictions
            recon: (T, C, H, W) reconstructed frames from z
        """
        z = self.encoder(frames)
        z_pred = self.dynamics(z)
        recon = self.decoder(z)
        return z, z_pred, recon

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
