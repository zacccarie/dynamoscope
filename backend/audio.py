"""Modalité audio : wav2vec2 features alignées avec frames vidéo.

Pipeline : ffmpeg extract audio → numpy float32 → wav2vec2 encode windows 1s → (N_frames, 768d).
Permet analyse cross-modale visuel↔audio (TE/causal/joint embedding).
"""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch


# Singletons : modèle wav2vec2 + feature extractor (chargés une seule fois)
_WAV2VEC = None
_WAV2VEC_EXT = None


def _load_wav2vec(device: torch.device):
    """Lazy-load wav2vec2-base (94M params, ~360MB).

    Self-supervised model entraîné sur 960h LibriSpeech (Baevski et al. 2020).
    Features 768d capturent structure phonémique + prosodique du signal audio.
    Singleton pattern : 1 seul load par session pour éviter re-init coûteux.
    """
    global _WAV2VEC, _WAV2VEC_EXT
    if _WAV2VEC is None:
        from transformers import Wav2Vec2Model, AutoFeatureExtractor
        _WAV2VEC = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _WAV2VEC.eval().to(device)
        _WAV2VEC_EXT = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    return _WAV2VEC, _WAV2VEC_EXT


def extract_audio(video_path: str | Path, sr: int = 16000) -> np.ndarray | None:
    """Extrait piste audio mono d'une vidéo via ffmpeg.

    16kHz mono = format standard wav2vec2 (entraîné sur 16kHz).
    Retourne None si vidéo n'a pas d'audio (silent fail propagé up-stack).

    Args:
        video_path: chemin fichier vidéo
        sr: sample rate cible (Hz)

    Returns:
        np.float32 1D array [-1, 1] ou None
    """
    out_path = Path(tempfile.gettempdir()) / f"dynamo_audio_{Path(video_path).stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1",          # mono
        "-ar", str(sr),      # sample rate
        "-f", "wav", "-loglevel", "error",
        str(out_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        if res.returncode != 0:
            return None
    except Exception:
        return None
    if not out_path.exists() or out_path.stat().st_size < 1000:
        return None
    import wave
    with wave.open(str(out_path), "rb") as wf:
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    # Décode selon bit-depth : 16-bit signed PCM (standard) ou 8-bit unsigned
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        audio = np.frombuffer(raw, dtype=np.float32)
    try:
        out_path.unlink()
    except Exception:
        pass
    return audio


def encode_audio_aligned(
    video_path: str | Path,
    frame_indices: list[int],
    fps: float,
    device: torch.device,
    sr: int = 16000,
) -> np.ndarray | None:
    """Encode audio en (N_frames, 768d) avec alignement temporel sur indices de frames.

    Stratégie : pour chaque frame_idx, fenêtre de 1s centrée sur le timestamp du frame
    (timestamp = idx / fps). wav2vec2 forward → mean pooling sur dim temporelle = 1 vecteur 768d.

    Pourquoi window 1s : suffisant pour capturer contexte phonémique + transitions audio.
    Pourquoi mean pool : agrège info temporelle fine en représentation par-frame.

    Returns: (N, 768) ou None si pas d'audio dans vidéo.
    """
    audio = extract_audio(video_path, sr=sr)
    if audio is None or len(audio) < sr // 4:
        return None
    model, ext = _load_wav2vec(device)
    n_frames = len(frame_indices)
    out = np.zeros((n_frames, 768), dtype=np.float32)

    win = int(0.5 * sr)  # 0.5s avant + après = 1s total context
    for i, idx in enumerate(frame_indices):
        # Timestamp réel de cette frame dans la vidéo
        t_center = idx / max(fps, 1e-6)
        s_center = int(t_center * sr)
        s0 = max(0, s_center - win)
        s1 = min(len(audio), s_center + win)
        chunk = audio[s0:s1]
        if len(chunk) < sr // 8:
            continue  # window trop courte (bord début/fin vidéo)
        try:
            inp = ext(chunk, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                # last_hidden_state : (1, T, 768) où T = chunk_len // 320
                h = model(**{k: v.to(device) for k, v in inp.items()}).last_hidden_state
            # Mean pool temporal axis → (768,) seul vecteur par frame
            out[i] = h.mean(dim=1).squeeze(0).float().cpu().numpy()
        except Exception:
            continue
    return out
