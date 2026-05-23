"""Cross-paradigm observable probe — 3 angles d'analyse.

Question : que captent les neural encoders ? Est-ce du physique (observables
classiques) ou du supra-physique ? Linear probe répond.

3 angles :
A. Interpretability : top observables prédisant chaque neural dim
B. Generalization : probe trained on bouncing_balls → eval on rotating_shapes
C. Comparative encoders : DINOv2 vs ResNet50 vs V-JEPA vs mini-RSSM vs Takens

Sur procedural videos. Output : results/observable_probe.json + ASCII tables.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.observables import compute_observables, OBSERVABLE_KEYS
from backend.models.registry import get_model
from backend.cross_paradigm import observable_probe, observable_probe_transfer
from backend.ingestion import sample_frames


VIDEOS = {
    "bouncing_balls": "videos/synthetic/bouncing_balls.mp4",
    "rotating_shapes": "videos/synthetic/rotating_shapes.mp4",
    "double_pendulum": "videos/synthetic/double_pendulum_render.mp4",
    "color_morph": "videos/synthetic/color_morph.mp4",
}

ENCODERS = [
    "resnet50",
    "dinov2_vits14",
    "vjepa2_vitl",
    # observable-based pour comparaison (devrait avoir R² élevé par construction)
    "delay_motion_auto_m3",
    "pca_obs_m3",
]


def encode_video(video_path: str, encoder_name: str, max_frames: int = 60):
    """Returns (latents, observables) for same set of sampled frames."""
    frames, _ = sample_frames(video_path, max_frames=max_frames)
    obs = compute_observables(frames)
    model = get_model(encoder_name)
    latents = model.produce_trajectory(frames)
    return latents, obs


def angle_A_interpretability(video_path: str) -> dict:
    """Per-encoder interpretability sur 1 vidéo. Rapport top obs."""
    out = {}
    for enc in ENCODERS:
        try:
            z, obs = encode_video(video_path, enc, max_frames=60)
            probe = observable_probe(z, obs)
            out[enc] = {
                "n_dims": probe["n_dims_total"],
                "mean_r2": probe["mean_r2"],
                "max_r2": probe["max_r2"],
                "median_r2": probe["median_r2"],
                "n_high_r2": probe["n_dims_r2_above_0.5"],
                "n_very_high_r2": probe["n_dims_r2_above_0.8"],
                "top_dims": [],
            }
            # Top-5 dims par R²
            ranked = sorted(
                enumerate(zip(probe["per_dim_r2"], probe["per_dim_top_obs"])),
                key=lambda x: -x[1][0],
            )[:5]
            for dim_idx, (r2, top_obs) in ranked:
                out[enc]["top_dims"].append({
                    "dim": dim_idx,
                    "r2": r2,
                    "top_obs": top_obs[:2],
                })
        except Exception as e:
            out[enc] = {"error": str(e)}
    return out


def angle_B_generalization() -> dict:
    """Probe trained sur bouncing → test sur rotating + double_pendulum + morph."""
    out = {}
    train_video = "bouncing_balls"
    test_videos = ["rotating_shapes", "double_pendulum", "color_morph"]

    for enc in ENCODERS:
        out[enc] = {}
        try:
            z_train, obs_train = encode_video(VIDEOS[train_video], enc, max_frames=60)
            for test_v in test_videos:
                z_test, obs_test = encode_video(VIDEOS[test_v], enc, max_frames=60)
                # Align dims if needed (encoder gives fixed latent_dim regardless of input)
                tr = observable_probe_transfer(z_train, obs_train, z_test, obs_test)
                out[enc][test_v] = {
                    "train_r2_mean": tr["train_r2_mean"],
                    "test_r2_mean": tr["test_r2_mean"],
                    "transfer_ratio": tr["transfer_ratio"],
                }
        except Exception as e:
            out[enc]["error"] = str(e)
    return out


def angle_C_comparative(video_path: str) -> dict:
    """Compare encoders by mean R² on same video. Higher = more 'physical'."""
    out = {}
    for enc in ENCODERS:
        try:
            z, obs = encode_video(video_path, enc, max_frames=60)
            probe = observable_probe(z, obs)
            out[enc] = {
                "latent_dim": probe["n_dims_total"],
                "mean_r2": probe["mean_r2"],
                "max_r2": probe["max_r2"],
                "frac_high_r2": probe["n_dims_r2_above_0.5"] / max(probe["n_dims_total"], 1),
            }
        except Exception as e:
            out[enc] = {"error": str(e)}
    return out


def main():
    print("=" * 75)
    print("CROSS-PARADIGM OBSERVABLE PROBE — 3 angles")
    print("=" * 75)
    t0 = time.time()
    results = {}

    # ANGLE A : interpretability on multiple videos, average
    print("\n[A] Interpretability per encoder (mean R² across 4 procedural videos)")
    results["A_interpretability"] = {}
    enc_acc = {e: {"r2s": [], "n_high": [], "n_dims": []} for e in ENCODERS}
    for vname, vpath in VIDEOS.items():
        if not Path(vpath).exists():
            continue
        print(f"  video: {vname}")
        ts = time.time()
        a = angle_A_interpretability(vpath)
        results["A_interpretability"][vname] = a
        for enc, stats in a.items():
            if "error" in stats:
                continue
            enc_acc[enc]["r2s"].append(stats["mean_r2"])
            enc_acc[enc]["n_high"].append(stats["n_high_r2"])
            enc_acc[enc]["n_dims"].append(stats["n_dims"])
        print(f"    ({time.time() - ts:.1f}s)")

    print(f"\n  {'encoder':<24}{'mean R²':>10}{'high-R² dims':>16}{'/total':>10}")
    for enc in ENCODERS:
        if not enc_acc[enc]["r2s"]:
            print(f"  {enc:<24}{'(failed)':>10}")
            continue
        mr2 = float(np.mean(enc_acc[enc]["r2s"]))
        mh = float(np.mean(enc_acc[enc]["n_high"]))
        md = float(np.mean(enc_acc[enc]["n_dims"]))
        print(f"  {enc:<24}{mr2:>10.3f}{mh:>16.1f}{md:>10.0f}")

    # ANGLE B : generalization (train on bouncing, test elsewhere)
    print("\n[B] Generalization (probe trained on bouncing_balls)")
    results["B_generalization"] = angle_B_generalization()
    print(f"\n  {'encoder':<24}{'test video':<22}{'train R²':>10}{'test R²':>10}{'ratio':>10}")
    for enc in ENCODERS:
        if "error" in results["B_generalization"][enc]:
            continue
        for test_v, stats in results["B_generalization"][enc].items():
            print(f"  {enc:<24}{test_v:<22}{stats['train_r2_mean']:>10.3f}"
                  f"{stats['test_r2_mean']:>10.3f}{stats['transfer_ratio']:>10.2f}")

    # ANGLE C : comparative ranking on each video
    print("\n[C] Comparative encoders — mean R² per encoder/video")
    results["C_comparative"] = {}
    print(f"\n  {'video':<22}{'encoder':<24}{'latent_dim':>12}{'mean R²':>10}{'frac high':>12}")
    for vname, vpath in VIDEOS.items():
        if not Path(vpath).exists():
            continue
        c = angle_C_comparative(vpath)
        results["C_comparative"][vname] = c
        for enc in ENCODERS:
            stats = c.get(enc, {})
            if "error" in stats:
                continue
            print(f"  {vname:<22}{enc:<24}{stats['latent_dim']:>12}"
                  f"{stats['mean_r2']:>10.3f}{stats['frac_high_r2']:>12.2f}")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    Path("results").mkdir(exist_ok=True)
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(results, open("results/observable_probe.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/observable_probe.json")


if __name__ == "__main__":
    main()
