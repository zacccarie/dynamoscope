"""CLI batch tool : process vidéos sans UI, dump JSON.

Usage examples :
  python -m backend.cli process video.mp4 --encoder dinov2_vits14 --out result.json
  python -m backend.cli batch ./videos --encoder resnet50 --out-dir ./results
  python -m backend.cli dna video.mp4
  python -m backend.cli compare a.mp4 b.mp4
  python -m backend.cli system lorenz --analyses all
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from .ingestion import sample_frames, video_meta
from .encoder import get_encoder, ENCODER_REGISTRY
from .reducer import reduce_3d
from .dynamics import analyse_trajectory
from .spectral import dmd, power_spectrum
from .multiscale import multiscale_entropy, spectral_slope
from .causal import causal_summary
from .topology import persistent_homology
from .sindy import fit_sindy
from .emergence import effective_information_ladder
from .segmentation import find_boundaries
from .clustering import cluster_latents, transition_matrix
from .sfa import slow_feature_analysis
from .evolution import evolution_pipeline
from .dna import compute_dna
from .prediction import fit_predictor
from .systems import SYSTEMS


def encode_video(path: str | Path, encoder_name: str, max_frames: int, reducer: str) -> dict:
    """Run encoding + reduction sur 1 video. Retourne latents + coords + meta."""
    frames, indices = sample_frames(path, max_frames=max_frames)
    enc = get_encoder(encoder_name)
    t0 = time.time()
    latents = enc.encode(frames)
    t_encode = time.time() - t0
    t0 = time.time()
    coords = reduce_3d(latents, method=reducer)
    t_reduce = time.time() - t0
    return {
        "latents": latents,
        "coords": coords,
        "indices": indices,
        "meta": video_meta(path),
        "encoder": encoder_name,
        "reducer": reducer,
        "timings": {"encode_s": round(t_encode, 3), "reduce_s": round(t_reduce, 3)},
    }


def run_analyses(latents: np.ndarray, coords: np.ndarray, which: list[str]) -> dict:
    """Lance les analyses sélectionnées. `which` = liste noms ou ['all']."""
    if "all" in which:
        which = [
            "dynamics", "spectral", "multiscale", "causal", "topology",
            "emergence", "segments", "clusters", "sfa", "evolution", "dna",
        ]
    results: dict = {}
    if "dynamics" in which:
        results["dynamics"] = analyse_trajectory(coords)
    if "spectral" in which:
        results["spectral"] = {
            "dmd": dmd(coords, dt=1.0),
            "power": power_spectrum(coords),
        }
    if "multiscale" in which:
        results["multiscale"] = {
            "entropy": multiscale_entropy(coords),
            "spectral_slope": spectral_slope(coords),
        }
    if "causal" in which:
        results["causal"] = causal_summary(coords, lag=2)
    if "topology" in which:
        results["topology"] = persistent_homology(coords, max_dim=1)
    if "emergence" in which:
        results["emergence"] = effective_information_ladder(coords)
    if "segments" in which:
        results["segments"] = find_boundaries(latents, method="adaptive", sensitivity=1.0)
    if "clusters" in which:
        cl = cluster_latents(latents, min_cluster_size=5)
        # transition matrix sur labels
        cl["transitions"] = transition_matrix(cl["labels"])
        results["clusters"] = cl
    if "sfa" in which:
        results["sfa"] = slow_feature_analysis(latents, n_components=4)
    if "evolution" in which:
        results["evolution"] = evolution_pipeline(latents, window=24, stride=8)
    if "dna" in which:
        results["dna"] = compute_dna(latents, coords)
    if "predict" in which:
        results["predict"] = fit_predictor(latents, epochs=80, device="cpu")
    return results


def cmd_process(args: argparse.Namespace) -> int:
    """Process 1 video, dump JSON."""
    print(f"[process] {args.video} · encoder={args.encoder} · reducer={args.reducer}", file=sys.stderr)
    t0 = time.time()
    enc = encode_video(args.video, args.encoder, args.max_frames, args.reducer)
    analyses = run_analyses(enc["latents"], enc["coords"], args.analyses)
    elapsed = time.time() - t0

    bundle = {
        "source": str(args.video),
        "encoder": enc["encoder"],
        "reducer": enc["reducer"],
        "n_frames": int(enc["latents"].shape[0]),
        "latent_dim": int(enc["latents"].shape[1]),
        "coords_3d": enc["coords"].tolist(),
        "frame_indices": enc["indices"],
        "meta": enc["meta"],
        "timings": enc["timings"],
        "total_elapsed_s": round(elapsed, 2),
        "analyses": analyses,
    }
    out_path = Path(args.out) if args.out else Path(args.video).with_suffix(".analysis.json")
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"[process] saved → {out_path} ({out_path.stat().st_size // 1024} KB) · {elapsed:.1f}s total", file=sys.stderr)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Process all videos in folder."""
    folder = Path(args.folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
    videos = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not videos:
        print(f"[batch] no videos found in {folder}", file=sys.stderr)
        return 1
    print(f"[batch] found {len(videos)} videos", file=sys.stderr)
    success = 0
    for i, v in enumerate(videos):
        print(f"[batch] {i+1}/{len(videos)} · {v.name}", file=sys.stderr)
        try:
            enc = encode_video(v, args.encoder, args.max_frames, args.reducer)
            analyses = run_analyses(enc["latents"], enc["coords"], args.analyses)
            bundle = {
                "source": v.name,
                "encoder": enc["encoder"],
                "reducer": enc["reducer"],
                "n_frames": int(enc["latents"].shape[0]),
                "latent_dim": int(enc["latents"].shape[1]),
                "coords_3d": enc["coords"].tolist(),
                "frame_indices": enc["indices"],
                "meta": enc["meta"],
                "timings": enc["timings"],
                "analyses": analyses,
            }
            out_path = out_dir / (v.stem + ".analysis.json")
            out_path.write_text(json.dumps(bundle, indent=2))
            success += 1
        except Exception as e:
            print(f"[batch] FAIL {v.name}: {e}", file=sys.stderr)
    print(f"[batch] done · {success}/{len(videos)} success", file=sys.stderr)
    return 0 if success == len(videos) else 1


def cmd_dna(args: argparse.Namespace) -> int:
    """Compute composite DNA score seul (rapide)."""
    enc = encode_video(args.video, args.encoder, args.max_frames, args.reducer)
    dna = compute_dna(enc["latents"], enc["coords"])
    out = {
        "source": str(args.video),
        "encoder": args.encoder,
        "n_frames": int(enc["latents"].shape[0]),
        "dna": dna,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare 2 videos : shared UMAP fit + diff stats."""
    print(f"[compare] {args.video_a} vs {args.video_b}", file=sys.stderr)
    a = encode_video(args.video_a, args.encoder, args.max_frames, "umap")
    b = encode_video(args.video_b, args.encoder, args.max_frames, "umap")
    if a["latents"].shape[1] != b["latents"].shape[1]:
        print("[compare] encoder dim mismatch", file=sys.stderr)
        return 1
    union = np.concatenate([a["latents"], b["latents"]], axis=0)
    coords = reduce_3d(union, method="umap")
    coords_a = coords[: a["latents"].shape[0]]
    coords_b = coords[a["latents"].shape[0] :]
    n_min = min(len(coords_a), len(coords_b))
    pair_diffs = np.linalg.norm(coords_a[:n_min] - coords_b[:n_min], axis=1)
    out = {
        "video_a": str(args.video_a),
        "video_b": str(args.video_b),
        "encoder": args.encoder,
        "n_a": int(a["latents"].shape[0]),
        "n_b": int(b["latents"].shape[0]),
        "mean_pair_distance": float(pair_diffs.mean()),
        "centroid_distance": float(np.linalg.norm(coords_a.mean(axis=0) - coords_b.mean(axis=0))),
        "dna_a": compute_dna(a["latents"], coords_a),
        "dna_b": compute_dna(b["latents"], coords_b),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_system(args: argparse.Namespace) -> int:
    """Generate + analyse trajectoire système synthétique."""
    if args.system_id not in SYSTEMS:
        print(f"[system] unknown system: {args.system_id}", file=sys.stderr)
        return 1
    print(f"[system] generating {args.system_id}", file=sys.stderr)
    sys_out = SYSTEMS[args.system_id]()
    coords = np.array(sys_out["coords"])
    raw = np.array(sys_out["raw_coords"])
    analyses = run_analyses(raw, coords, args.analyses)
    out = {
        "system": args.system_id,
        "name": sys_out["name"],
        "type": sys_out["type"],
        "meta": sys_out["meta"],
        "n_points": int(coords.shape[0]),
        "coords_3d": coords.tolist(),
        "analyses": analyses,
    }
    out_path = Path(args.out) if args.out else Path(f"{args.system_id}.analysis.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[system] saved → {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="dynamoscope",
        description="Video → latent trajectory → emergence analysis (CLI mode)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args
    def add_common(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--encoder", default="resnet50", choices=list(ENCODER_REGISTRY.keys()))
        ap.add_argument("--reducer", default="umap", choices=["umap", "pca", "isomap"])
        ap.add_argument("--max-frames", type=int, default=120, dest="max_frames")
        ap.add_argument("--analyses", nargs="+", default=["dna"], help="Liste analyses ou 'all'")

    p_proc = sub.add_parser("process", help="Process 1 video, dump JSON")
    p_proc.add_argument("video", type=str)
    p_proc.add_argument("--out", type=str, default=None)
    add_common(p_proc)
    p_proc.set_defaults(func=cmd_process)

    p_batch = sub.add_parser("batch", help="Process all videos in folder")
    p_batch.add_argument("folder", type=str)
    p_batch.add_argument("--out-dir", default="./results", dest="out_dir")
    add_common(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_dna = sub.add_parser("dna", help="Compute composite DNA score only")
    p_dna.add_argument("video", type=str)
    add_common(p_dna)
    p_dna.set_defaults(func=cmd_dna)

    p_cmp = sub.add_parser("compare", help="Compare 2 videos (shared UMAP)")
    p_cmp.add_argument("video_a", type=str)
    p_cmp.add_argument("video_b", type=str)
    add_common(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    p_sys = sub.add_parser("system", help="Analyse synthetic dynamical system")
    p_sys.add_argument("system_id", choices=list(SYSTEMS.keys()))
    p_sys.add_argument("--out", type=str, default=None)
    p_sys.add_argument("--analyses", nargs="+", default=["all"])
    p_sys.set_defaults(func=cmd_system)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
