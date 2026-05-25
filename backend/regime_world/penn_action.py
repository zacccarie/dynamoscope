"""Penn Action dataset loader for RegimeWorldModel Phase 2 real-video.

Penn Action (Zhang et al. 2013, ICCV) :
- 2326 video sequences, 15 actions, joint annotations
- Source : https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz
- Layout after extraction :
    Penn_Action/frames/{seq_id:04d}/{frame_idx:06d}.jpg
    Penn_Action/labels/{seq_id:04d}.mat   ← action, x, y, bbox, nframes, train

Pipeline :
1. PennActionLoader : scan dataset, filter by action class
2. load_video_frames(seq_id) : read JPEGs, return (T, H, W, 3) uint8
3. encode_penn_action_video(...) : encode + cache DINOv2 features
4. build_penn_action_dataset(actions=[...], n_per_class=...) :
   returns list[VideoSample] compatible with existing trainer.
"""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Default expected paths — user can override via env vars.
PENN_ROOT = Path(os.environ.get(
    "PENN_ACTION_ROOT",
    "/Users/zac/Downloads/penn_action_data/Penn_Action",
))
CACHE_DIR = Path("cache/penn_action_dinov2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# 15 Penn Action classes — names as STORED in .mat files (verified scan).
# Different from official paper naming (squats → squat, bowling → bowl, etc.)
ACTIONS = [
    "baseball_pitch", "baseball_swing", "bench_press", "bowl",
    "clean_and_jerk", "golf_swing", "jumping_jacks", "jump_rope",
    "pullup", "pushup", "situp", "squat",
    "strum_guitar", "tennis_forehand", "tennis_serve",
]


@dataclass
class PennActionSample:
    """Penn Action video sample : features + metadata.

    Compatible with VideoSample interface used by trainer (has .traj).
    """
    features: np.ndarray   # (T, D_enc) DINOv2 features per frame
    action: str            # action class name
    seq_id: int            # original Penn Action sequence id
    n_frames_original: int

    @property
    def traj(self) -> np.ndarray:
        return self.features

    @traj.setter
    def traj(self, value: np.ndarray) -> None:
        self.features = value

    @property
    def regime(self) -> str:
        """Alias for trainer compat ; we use action name as regime label."""
        return self.action


def _ensure_dataset_exists() -> None:
    if not PENN_ROOT.exists():
        raise FileNotFoundError(
            f"Penn Action root not found at {PENN_ROOT}. Download from "
            f"https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz and extract."
        )
    if not (PENN_ROOT / "frames").exists():
        raise FileNotFoundError(f"{PENN_ROOT}/frames/ missing.")
    if not (PENN_ROOT / "labels").exists():
        raise FileNotFoundError(f"{PENN_ROOT}/labels/ missing.")


def list_sequences_by_action() -> dict[str, list[int]]:
    """Scan all labels/*.mat, return dict action_name → list of seq_ids."""
    _ensure_dataset_exists()
    from scipy.io import loadmat
    labels_dir = PENN_ROOT / "labels"
    by_action = {a: [] for a in ACTIONS}
    for mat_file in sorted(labels_dir.glob("*.mat")):
        seq_id = int(mat_file.stem)
        try:
            mat = loadmat(mat_file, squeeze_me=True)
            action = str(mat.get("action", ""))
            if action in by_action:
                by_action[action].append(seq_id)
        except Exception as e:
            print(f"  [warn] could not parse {mat_file.name}: {e}")
    return by_action


def load_video_frames(seq_id: int, max_frames: int = 64,
                       sample_uniform: bool = True) -> np.ndarray:
    """Load JPEG frames for sequence seq_id → (T, H, W, 3) uint8.

    sample_uniform : if True and the video has > max_frames, uniformly
    sub-sample to max_frames (else use first max_frames).
    """
    import cv2
    _ensure_dataset_exists()
    frames_dir = PENN_ROOT / "frames" / f"{seq_id:04d}"
    if not frames_dir.exists():
        raise FileNotFoundError(f"{frames_dir} missing.")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_total = len(frame_paths)
    if n_total == 0:
        raise ValueError(f"No JPEGs in {frames_dir}")
    if n_total > max_frames:
        if sample_uniform:
            idx = np.linspace(0, n_total - 1, max_frames).astype(int)
            frame_paths = [frame_paths[i] for i in idx]
        else:
            frame_paths = frame_paths[:max_frames]
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.stack(frames)


def _cache_key(seq_id: int, max_frames: int, encoder_name: str) -> str:
    h = hashlib.md5(f"penn_{seq_id}_{max_frames}_{encoder_name}".encode()).hexdigest()[:12]
    return f"penn_{seq_id:04d}_{max_frames}_{encoder_name}_{h}.npy"


def encode_penn_action_video(seq_id: int, encoder,
                              encoder_name: str = "dinov2_vits14",
                              max_frames: int = 64,
                              target_size: int = 224) -> np.ndarray:
    """Load frames + DINOv2 encode + cache. Returns (T, 384) features."""
    cache_path = CACHE_DIR / _cache_key(seq_id, max_frames, encoder_name)
    if cache_path.exists():
        return np.load(cache_path)
    frames = load_video_frames(seq_id, max_frames=max_frames)
    # Reuse the existing video_data encoder helper for consistency.
    from .video_data import encode_video_dinov2
    feats = encode_video_dinov2(frames, encoder, target_size=target_size)
    np.save(cache_path, feats)
    return feats


def build_penn_action_dataset(
    actions: list[str] = None,
    n_per_class: int = 20,
    max_frames: int = 64,
    encoder_name: str = "dinov2_vits14",
    seed: int = 0,
    verbose: bool = True,
) -> list[PennActionSample]:
    """Build dataset : pick `n_per_class` videos per action class.

    actions : if None, uses all 15.
    """
    _ensure_dataset_exists()
    if actions is None:
        actions = ACTIONS
    from ..encoder import get_encoder
    if verbose:
        print(f"  loading encoder {encoder_name}...")
    encoder = get_encoder(encoder_name)
    by_action = list_sequences_by_action()
    rng = np.random.default_rng(seed)
    out = []
    for action in actions:
        seqs = by_action.get(action, [])
        if not seqs:
            print(f"  [warn] no sequences for action '{action}'")
            continue
        if n_per_class < len(seqs):
            seqs = list(rng.choice(seqs, size=n_per_class, replace=False))
        if verbose:
            print(f"    [{action:<18}] {len(seqs)} seqs")
        for seq_id in seqs:
            try:
                feats = encode_penn_action_video(
                    int(seq_id), encoder,
                    encoder_name=encoder_name, max_frames=max_frames,
                )
                # Original frame count from .mat
                n_orig = max_frames  # approximation; could load from .mat
                out.append(PennActionSample(
                    features=feats, action=action,
                    seq_id=int(seq_id), n_frames_original=n_orig,
                ))
            except Exception as e:
                print(f"    [skip] seq {seq_id}: {e}")
    if verbose:
        print(f"  -> {len(out)} Penn Action samples encoded")
    return out


def windows_from_penn_sample(sample: PennActionSample, window: int = 32,
                              stride: int = 8) -> list[PennActionSample]:
    """Sub-window slicing for training."""
    out = []
    T = sample.features.shape[0]
    for start in range(0, T - window + 1, stride):
        feats_w = sample.features[start:start + window]
        out.append(PennActionSample(
            features=feats_w, action=sample.action,
            seq_id=sample.seq_id,
            n_frames_original=sample.n_frames_original,
        ))
    return out
