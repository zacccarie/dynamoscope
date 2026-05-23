"""Cross-model benchmark : same video → N models → comparative metrics matrix.

Produit table standardisée (model × video × metric) avec :
- DNA composite score + 7 axes
- Dynamics : Lyapunov, correlation dim
- Topology : H1 count, persistence entropy
- Smoothness : mean perceptual velocity
- Timings : encode + reduce + analyze

Permet :
- Leaderboard cross-model par régime dynamique
- Détection encoder bias (qui smooth, qui chaotic)
- Statistical tests pour différencier models significativement
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np

from .models.registry import get_model, MODEL_REGISTRY
from .models.base import TrajectoryProducer
from .ingestion import sample_frames
from .reducer import reduce_3d
from .dna import compute_dna
from .dynamics import analyse_trajectory
from .topology import persistent_homology


def _safe_metric(fn, default=None):
    """Run metric function, return default on exception."""
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)} if default is None else default


def run_model_on_video(
    model: TrajectoryProducer,
    video_path: str | Path,
    max_frames: int = 60,
    reducer: str = "umap",
) -> dict:
    """Pipeline complet : video → model → trajectory → metrics."""
    t0 = time.time()
    frames, indices = sample_frames(video_path, max_frames=max_frames)
    t_sample = time.time() - t0

    t0 = time.time()
    try:
        latents = model.produce_trajectory(frames)
    except NotImplementedError as e:
        return {"model": model.name, "video": Path(video_path).name, "error": str(e), "status": "stub"}
    t_encode = time.time() - t0

    t0 = time.time()
    coords = reduce_3d(latents, method=reducer)
    t_reduce = time.time() - t0

    # Compute metrics
    t0 = time.time()
    dyn = _safe_metric(lambda: analyse_trajectory(coords))
    ph = _safe_metric(lambda: persistent_homology(coords, max_dim=1))
    dna = _safe_metric(lambda: compute_dna(latents, coords))
    t_metrics = time.time() - t0

    # Smoothness : mean perceptual velocity (cosine distance frame-à-frame)
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    units = latents / np.maximum(norms, 1e-9)
    cos_sim = np.sum(units[:-1] * units[1:], axis=1).clip(-1, 1)
    mean_perceptual_v = float(np.mean(1.0 - cos_sim))

    result = {
        "model": model.name,
        "category": model.category,
        "latent_dim": int(latents.shape[1]),
        "video": Path(video_path).name,
        "n_frames": int(latents.shape[0]),
        "mean_perceptual_velocity": round(mean_perceptual_v, 6),
        "timings": {
            "sample_s": round(t_sample, 3),
            "encode_s": round(t_encode, 3),
            "reduce_s": round(t_reduce, 3),
            "metrics_s": round(t_metrics, 3),
            "total_s": round(t_sample + t_encode + t_reduce + t_metrics, 3),
        },
        "status": "ok",
    }

    if isinstance(dyn, dict) and "lyapunov" in dyn:
        result["lyapunov"] = dyn["lyapunov"]
        result["correlation_dim"] = dyn["correlation_dim"]
        result["rqa_det"] = dyn["rqa"]["DET"]

    if isinstance(ph, dict) and "diagrams" in ph:
        result["h0_count"] = ph["diagrams"][0]["count"]
        result["h1_count"] = ph["diagrams"][1]["count"] if len(ph["diagrams"]) > 1 else 0
        result["h1_entropy"] = ph["persistence_entropy"].get("H1", 0.0)

    if isinstance(dna, dict) and "composite_score" in dna:
        result["dna_score"] = dna["composite_score"]
        result["dna_label"] = dna["label"]
        result["dna_axes"] = dna["axes"]

    return result


def cross_model_benchmark(
    video_paths: list[str | Path],
    model_names: list[str],
    max_frames: int = 60,
    reducer: str = "umap",
    skip_stubs: bool = True,
) -> dict:
    """Run all (video × model) combinations. Return matrix + summary.

    Args:
        video_paths: liste vidéos benchmark
        model_names: liste noms models depuis MODEL_REGISTRY
        max_frames: frames sampling per video
        reducer: umap | pca | isomap
        skip_stubs: ignore world model stubs (NotImplementedError)

    Returns:
        - rows : liste de dicts (un par paire video × model)
        - matrix : dict {model: {video: dna_score}}
        - summary : aggregated stats per model
    """
    # Load models once
    loaded_models = {}
    for name in model_names:
        try:
            m = get_model(name)
            loaded_models[name] = m
        except Exception as e:
            print(f"[benchmark] skip {name}: {e}")

    rows: list[dict] = []
    matrix: dict[str, dict[str, float]] = {m: {} for m in loaded_models}

    for v in video_paths:
        v = Path(v)
        if not v.exists():
            continue
        for name, model in loaded_models.items():
            result = run_model_on_video(model, v, max_frames=max_frames, reducer=reducer)
            if result.get("status") == "stub" and skip_stubs:
                continue
            rows.append(result)
            if "dna_score" in result:
                matrix[name][v.name] = result["dna_score"]

    # Summary per model
    summary = {}
    for name in loaded_models:
        scores = [r["dna_score"] for r in rows if r["model"] == name and "dna_score" in r]
        smoothness = [r["mean_perceptual_velocity"] for r in rows if r["model"] == name]
        times = [r["timings"]["total_s"] for r in rows if r["model"] == name]
        if scores:
            summary[name] = {
                "n_runs": len(scores),
                "dna_mean": round(float(np.mean(scores)), 2),
                "dna_std": round(float(np.std(scores)), 2),
                "dna_range": [round(float(np.min(scores)), 2), round(float(np.max(scores)), 2)],
                "smoothness_mean": round(float(np.mean(smoothness)), 6),
                "compute_avg_s": round(float(np.mean(times)), 2),
            }

    return {
        "rows": rows,
        "matrix": matrix,
        "summary": summary,
        "n_videos": len(video_paths),
        "n_models": len(loaded_models),
        "skipped_stubs": skip_stubs,
    }


def render_leaderboard(benchmark_result: dict) -> str:
    """ASCII leaderboard rendering."""
    summary = benchmark_result["summary"]
    if not summary:
        return "No results."
    # Sort models by DNA mean descending
    sorted_models = sorted(summary.items(), key=lambda x: -x[1]["dna_mean"])
    lines = []
    lines.append(f"{'MODEL':<22}{'RUNS':>6}{'DNA mean':>11}{'DNA std':>10}{'smooth':>10}{'time/video':>12}")
    lines.append("-" * 71)
    for name, s in sorted_models:
        lines.append(
            f"{name:<22}{s['n_runs']:>6}{s['dna_mean']:>11.2f}{s['dna_std']:>10.2f}"
            f"{s['smoothness_mean']:>10.5f}{s['compute_avg_s']:>11.2f}s"
        )
    return "\n".join(lines)
