"""Encoders : transforment frames pixels → vecteurs latents haute-dim.

Chaque encoder = vue particulière du visuel, conditionnée par son pretraining :
- ResNet50/ViT-B : supervised ImageNet → biais vers classes ImageNet
- DINOv2 : self-supervised → invariances naturelles sans labels
- CLIP : contrastive vision-language → espace commun avec texte

Choisir encoder = choisir quelle "perception" appliquer à la vidéo.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Literal


def get_device() -> torch.device:
    """Sélectionne le meilleur device GPU dispo : MPS (Apple Silicon) > CUDA > CPU.

    Apple MPS = Metal Performance Shaders, accélération GPU sur Mac M1/M2/M3.
    Bf16/fp32 supporté sur MPS depuis PyTorch 2.0+.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Stats normalisation ImageNet (mean, std par canal RGB)
# Critique : tous les modèles supervised attendent images normalisées avec ces stats
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


class ResNet50Encoder:
    """ResNet50 supervised ImageNet · 2048d output.

    Architecture résiduelle CNN (He et al. 2016). 25M params.
    Penultimate features (avant classifier) = 2048d = vecteur descriptif image.
    Biais : entraîné pour discriminer 1000 classes ImageNet → features activent sur
    catégories sémantiques (chien/voiture/etc.) plutôt que sur traits abstraits.
    """
    name = "resnet50"
    latent_dim = 2048

    def __init__(self, device: torch.device | None = None):
        """Charge ResNet50 pretrained, remplace classifier par Identity pour extraire features."""
        self.device = device or get_device()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Identity = retire couche classification, garde features 2048d penultimate
        backbone.fc = nn.Identity()
        backbone.eval().to(self.device)
        self.model = backbone

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Encode batch de frames → features 2048d.

        Args:
            frames: (N, H, W, 3) float [0,1] RGB
            batch_size: taille batch pour limiter VRAM

        Returns:
            (N, 2048) features float32
        """
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            # NHWC numpy → NCHW torch (channel-first attendu par PyTorch)
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            t = _normalize(t)
            feats = self.model(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)


class ViTEncoder:
    """Vision Transformer ViT-B/16 supervised ImageNet · 768d output.

    Architecture attention-only (Dosovitskiy et al. 2020). 86M params.
    Patches 16×16 + transformer encoder, CLS token = représentation globale.
    Plus expressif que ResNet50 mais features moins compactes par dim.
    """
    name = "vit_b_16"
    latent_dim = 768

    def __init__(self, device: torch.device | None = None):
        """Charge ViT-B/16 pretrained, remplace head classification par Identity."""
        self.device = device or get_device()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Identity()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """Encode frames → CLS token 768d. Batch size plus petit (transformer ↑ VRAM)."""
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
    """DINOv2 self-supervised (Meta 2023) · 384d output.

    Apprentissage student-teacher sans labels (auto-distillation).
    Pas de biais classes ImageNet : features capturent invariances visuelles "naturelles"
    (objets, contexte, géométrie) plutôt que catégories supervisées.
    Excellent pour vidéo car continuité visuelle mappe à continuité latente.
    """
    name = "dinov2_vits14"
    latent_dim = 384

    def __init__(self, device: torch.device | None = None):
        """Charge DINOv2-S via torch.hub. Auto-download 84MB checkpoint au premier usage."""
        self.device = device or get_device()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        self.model.eval().to(self.device)
        # Stats ImageNet : DINOv2 réutilise même normalisation
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """Encode frames → CLS token DINOv2 384d. Self-supervised features."""
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            t = (t - self._mean) / self._std
            feats = self.model(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)


class CLIPEncoder:
    """CLIP ViT-B/32 (OpenAI 2021) · 512d image + texte espace partagé.

    Entrainé par contrastive learning sur 400M paires (image, caption web).
    Image features et text features dans même espace → permet :
    - Recherche sémantique : "find frames matching 'sunset'"
    - Zero-shot classification : embed candidate labels, compare cosine sim
    - Cross-modal alignment : audio/texte/image dans espace commun
    """
    name = "clip_vit_b32"
    latent_dim = 512

    def __init__(self, device: torch.device | None = None):
        """Charge CLIP ViT-B/32 via open_clip. 151M params, ~360MB download."""
        self.device = device or get_device()
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai",
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Stats CLIP-specific (différentes ImageNet !) : OpenAI computed sur leur dataset
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """Encode frames image → CLIP image features 512d (espace partagé avec texte)."""
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
        """Encode liste de strings → text features 512d dans même espace que images.

        Permet cosine_similarity(image_feat, text_feat) = score "matching" sémantique.
        Usage : query video frames matching 'dog playing' → top-k frames similaires.
        """
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens).float().cpu().numpy()
        return feats


class VideoMAEEncoder:
    """VideoMAE-base (Tong et al. 2022) · 768d output per-frame via clip context.

    Premier encoder réellement temporel : attend clips de 16 frames,
    produit features qui agrègent contexte temporel (mouvement, transitions).

    Architecture : Masked Autoencoder sur tubelet (T=2, P=16) patches.
    Pretrained Kinetics-400 self-supervised → invariances motion natives.

    Stratégie per-frame : pour chaque frame i, construit clip 16-frame
    centré sur i (avec padding clamp aux bords), forward, mean-pool spatial+temporel
    → 1 feature 768d par frame d'entrée.
    """
    name = "videomae_base"
    latent_dim = 768

    def __init__(self, device: torch.device | None = None):
        """Charge VideoMAE-base via transformers. ~340MB checkpoint."""
        self.device = device or get_device()
        from transformers import VideoMAEModel, VideoMAEImageProcessor
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.model.eval().to(self.device)
        self.proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 4, clip_size: int = 16) -> np.ndarray:
        """Per-frame encoding via 16-frame centered context window.

        Pour chaque frame i, construit clip [i-7..i+8] clamped aux bords du tableau.
        Forward batch_size clips à la fois. Mean-pool over spatial-temporal tokens.

        Args:
            frames: (N, H, W, 3) float [0,1] RGB
            batch_size: nombre de clips traités ensemble (limite VRAM)
            clip_size: longueur temporelle clip (16 = standard VideoMAE)

        Returns:
            (N, 768) features per-frame avec contexte temporel intégré
        """
        n = frames.shape[0]
        frames_u8 = (frames * 255).clip(0, 255).astype(np.uint8)
        out: list[np.ndarray] = []
        half = clip_size // 2

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_clips = []
            for i in range(batch_start, batch_end):
                idxs = [min(max(0, i - half + 1 + k), n - 1) for k in range(clip_size)]
                clip = [frames_u8[j] for j in idxs]
                batch_clips.append(clip)
            inp = self.proc(batch_clips, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            h = self.model(**inp).last_hidden_state  # (B, T_patches, 768)
            feats = h.mean(dim=1).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)


# Registry : encoders disponibles, accessibles par nom string via /api endpoints
ENCODER_REGISTRY = {
    "resnet50": ResNet50Encoder,
    "vit_b_16": ViTEncoder,
    "dinov2_vits14": DinoV2Encoder,
    "clip_vit_b32": CLIPEncoder,
    "videomae_base": VideoMAEEncoder,
}


def get_encoder(name: Literal["resnet50", "vit_b_16", "dinov2_vits14", "clip_vit_b32", "videomae_base"]):
    """Factory function : instancie encoder par nom string.

    Lazy-loaded : modèle chargé en mémoire seulement quand requis.
    Cached côté caller (main.py `_ENCODERS_CACHE`) pour éviter rechargements.
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}")
    return ENCODER_REGISTRY[name]()
