"""Encodeurs : ResNet50 + ViT-B (torchvision)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Literal


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


class ResNet50Encoder:
    """Penultimate features 2048d."""
    name = "resnet50"
    latent_dim = 2048

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        backbone.eval().to(self.device)
        self.model = backbone

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """frames: (N, H, W, 3) float32 [0,1] -> (N, 2048) float32."""
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            t = _normalize(t)
            feats = self.model(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)


class ViTEncoder:
    """ViT-B/16 IMAGENET1K_V1, dernier token CLS = 768d."""
    name = "vit_b_16"
    latent_dim = 768

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # hook avant classifier head
        self.model.heads = nn.Identity()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            t = _normalize(t)
            feats = self.model(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)


class DinoV2Encoder:
    """DINOv2 self-supervised (Meta 2023). Pas de biais ImageNet classes."""
    name = "dinov2_vits14"
    latent_dim = 384

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        self.model.eval().to(self.device)
        # DINOv2 prefere 224 mais patch_size=14, doit etre multiple
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            # Resize 224 -> 224 (deja 224), normalise
            t = (t - self._mean) / self._std
            feats = self.model(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)


class CLIPEncoder:
    """CLIP ViT-B/32 (OpenAI). Image + texte dans espace commun 512d.
    Permet query semantique text -> frames."""
    name = "clip_vit_b32"
    latent_dim = 512

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai",
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Normalisation CLIP-specific (OpenAI stats)
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            t = (t - self._mean) / self._std
            feats = self.model.encode_image(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens).float().cpu().numpy()
        return feats


ENCODER_REGISTRY = {
    "resnet50": ResNet50Encoder,
    "vit_b_16": ViTEncoder,
    "dinov2_vits14": DinoV2Encoder,
    "clip_vit_b32": CLIPEncoder,
}


def get_encoder(name: Literal["resnet50", "vit_b_16", "dinov2_vits14", "clip_vit_b32"]):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}")
    return ENCODER_REGISTRY[name]()
