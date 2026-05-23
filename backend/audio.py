"""Modalite audio : extract audio + wav2vec2 features alignées avec frames."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch


_WAV2VEC = None
_WAV2VEC_EXT = None


def _load_wav2vec(device: torch.device):
    global _WAV2VEC, _WAV2VEC_EXT
    if _WAV2VEC is None:
        from transformers import Wav2Vec2Model, AutoFeatureExtractor
        _WAV2VEC = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _WAV2VEC.eval().to(device)
        _WAV2VEC_EXT = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    return _WAV2VEC, _WAV2VEC_EXT


def extract_audio(video_path: str | Path, sr: int = 16000) -> np.ndarray | None:
    """Extract audio mono via ffmpeg. Retourne np.float32 ou None si pas d'audio."""
    out_path = Path(tempfile.gettempdir()) / f"dynamo_audio_{Path(video_path).stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sr), "-f", "wav", "-loglevel", "error",
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
    # Lit le wav
    import wave
    with wave.open(str(out_path), "rb") as wf:
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
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
    """Encode audio en (N_frames, 768) aligne sur frame_indices.
    Chaque feature = wav2vec moyenne sur fenetre 1s autour du frame timestamp."""
    audio = extract_audio(video_path, sr=sr)
    if audio is None or len(audio) < sr // 4:
        return None
    model, ext = _load_wav2vec(device)
    n_frames = len(frame_indices)
    out = np.zeros((n_frames, 768), dtype=np.float32)

    # Fenetre 1s autour de chaque frame_idx (~ 0.5s avant + apres)
    win = int(0.5 * sr)
    for i, idx in enumerate(frame_indices):
        t_center = idx / max(fps, 1e-6)
        s_center = int(t_center * sr)
        s0 = max(0, s_center - win)
        s1 = min(len(audio), s_center + win)
        chunk = audio[s0:s1]
        if len(chunk) < sr // 8:
            continue  # trop court
        try:
            inp = ext(chunk, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                h = model(**{k: v.to(device) for k, v in inp.items()}).last_hidden_state
            out[i] = h.mean(dim=1).squeeze(0).float().cpu().numpy()
        except Exception:
            continue
    return out
