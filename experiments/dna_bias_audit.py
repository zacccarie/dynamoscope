"""DNA bias audit : décompose composite score par axe × type model.

Question : pourquoi classical observables battent neural sur DNA ?
Hypothèses :
- H1 : DNA pondère lourdement axes que classical excelle (topology, causality)
- H2 : Neural encoders perdent dynamics signal lors compression sémantique
- H3 : Classical observables sont biaisés *vers* dynamics par construction
  (delay embedding implique structure dynamique explicite)
- H4 : DNA est sain mais évalue *dynamics-richness* (not perceptual quality)
  → classical sont effectivement meilleurs sur ce critère

Output : pour chaque (model, video) toutes valeurs d'axes brutes.
Compare classical vs neural sur chaque axe séparément + relative contribution
to composite.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.benchmark_models import run_model_on_video


VIDEOS = [
    "videos/synthetic/bouncing_balls.mp4",
    "videos/synthetic/double_pendulum_render.mp4",
    "videos/synthetic/rotating_shapes.mp4",
    "videos/synthetic/color_morph.mp4",
    "videos/synthetic/reaction_diffusion.mp4",
    "videos/synthetic/game_of_life.mp4",
]

CLASSICAL_MODELS = [
    "delay_motion_auto_m3",
    "delay_brightness_auto_m3",
    "delay_entropy_auto_m3",
    "pca_obs_m3",
    "direct_bme",
]
NEURAL_MODELS = [
    "dinov2_vits14",
    "resnet50",
]


def main():
    from backend.models.registry import get_model

    rows = []
    for video in VIDEOS:
        for name in CLASSICAL_MODELS + NEURAL_MODELS:
            m = get_model(name)
            r = run_model_on_video(m, video, max_frames=60, reducer="pca")
            if r.get("status") != "ok":
                continue
            row = {
                "video": Path(video).stem,
                "model": name,
                "type": "classical" if name in CLASSICAL_MODELS else "neural",
                "dna_score": r.get("dna_score"),
            }
            if "dna_axes" in r:
                for k, v in r["dna_axes"].items():
                    row[f"axis_{k}"] = v
            rows.append(row)
            print(f"  {Path(video).stem:<28}{name:<28}DNA={r.get('dna_score', 0):.2f}")

    # Aggregate per (type, axis)
    axes_keys = [k for k in rows[0] if k.startswith("axis_")]
    summary = {}
    for typ in ("classical", "neural"):
        summary[typ] = {}
        for ax in axes_keys:
            vals = [r[ax] for r in rows if r["type"] == typ and ax in r]
            summary[typ][ax] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    # Compute per-axis advantage : classical - neural
    print("\n" + "=" * 80)
    print("AXIS-LEVEL CLASSICAL vs NEURAL")
    print("=" * 80)
    print(f"{'axis':<22}{'weight':>10}{'classical':>15}{'neural':>15}{'delta':>15}{'weighted_delta':>20}")
    print("-" * 100)

    # weights from dna.py
    WEIGHTS = {
        "topology": 0.27,
        "causality": 0.18,
        "spectral": 0.16,
        "regime_confidence": 0.10,
        "predictability": 0.09,
        "chaos": 0.09,
        "complexity": 0.06,
        "structure": 0.05,
    }
    weighted_deltas = {}
    for ax in axes_keys:
        ax_name = ax.replace("axis_", "")
        w = WEIGHTS.get(ax_name, 0)
        c_mean = summary["classical"][ax]["mean"]
        n_mean = summary["neural"][ax]["mean"]
        delta = c_mean - n_mean
        wd = delta * w * 100
        weighted_deltas[ax_name] = wd
        print(f"{ax_name:<22}{w:>10.2f}{c_mean:>15.4f}{n_mean:>15.4f}{delta:>+15.4f}{wd:>+20.2f}")

    total_delta = sum(weighted_deltas.values())
    print("-" * 100)
    print(f"{'sum weighted Δ':<22}{'':>10}{'':>15}{'':>15}{'':>15}{total_delta:>+20.2f}")
    print(f"({'actual DNA gap from leaderboard'}: ~6-7 pts)")

    # Save artifact
    Path("results").mkdir(exist_ok=True)
    out = {
        "rows": rows,
        "summary_by_type": summary,
        "weighted_deltas_classical_minus_neural": weighted_deltas,
        "interpretation": _interpret(weighted_deltas),
    }
    json.dump(out, open("results/dna_bias_audit.json", "w"), indent=2)
    print(f"\n[saved] results/dna_bias_audit.json")


def _interpret(deltas: dict) -> dict:
    """Identify which axes drive classical advantage and which favor neural."""
    sorted_ax = sorted(deltas.items(), key=lambda x: -x[1])
    return {
        "favors_classical": [k for k, v in sorted_ax if v > 0.3],
        "favors_neural": [k for k, v in sorted_ax if v < -0.3],
        "neutral": [k for k, v in sorted_ax if -0.3 <= v <= 0.3],
        "top_driver_classical": sorted_ax[0] if sorted_ax else None,
        "top_driver_neural": sorted_ax[-1] if sorted_ax else None,
    }


if __name__ == "__main__":
    main()
