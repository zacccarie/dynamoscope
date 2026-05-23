"""FastAPI app : ingestion + encode + reduce + serve frontend."""
from __future__ import annotations
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .autotune import bayesian_search, random_search
from .audio import encode_audio_aligned
from .causal import causal_summary
from .causal_advanced import causal_summary_advanced, pcmci_discovery, te_matrix_ksg
from .clustering import cluster_latents, transition_matrix
from .dna import compute_dna
from .evolution import evolution_pipeline
from .live import create_session, end_session, get_session
from .prediction import counterfactual_rollouts, fit_predictor, rollout as rollout_fn
from .dynamics import analyse_trajectory
from .emergence import effective_information_ladder
from .encoder import ENCODER_REGISTRY, get_device, get_encoder
from .ingestion import sample_frames, save_thumbnails, video_meta
from .multiscale import multiscale_entropy, spectral_slope
from .reducer import lorenz_trajectory, reduce_3d
from .segmentation import find_boundaries
from .registry import (
    delete_experiment, get_experiment, list_experiments, save_experiment,
)
from .sfa import slow_feature_analysis
from .wavelets import wavelet_per_dim
from .benchmark import benchmark_summary
from .sindy import fit_sindy
from .spectral import dmd, power_spectrum
from .systems import SYSTEMS, list_systems
from .topology import persistent_homology, sliding_window_ph

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "videos"
CACHE_DIR = ROOT / "cache"
FRAMES_DIR = CACHE_DIR / "frames"
FRONTEND_DIR = ROOT / "frontend"
VIDEOS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Dynamoscope",
    version="0.6.0",
    description="Video → latent trajectory → emergence analysis. /docs for OpenAPI.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_ENCODERS_CACHE: dict[str, object] = {}


def _get_or_load_encoder(name: str):
    if name not in _ENCODERS_CACHE:
        _ENCODERS_CACHE[name] = get_encoder(name)
    return _ENCODERS_CACHE[name]


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "device": str(get_device()),
        "encoders": list(ENCODER_REGISTRY.keys()),
    }


@app.get("/api/demo/lorenz")
def demo_lorenz(n: int = 2000) -> JSONResponse:
    n = max(100, min(n, 10000))
    coords, raw, dt = lorenz_trajectory(n=n, return_raw=True)
    return JSONResponse(
        {
            "name": "Lorenz attractor (demo)",
            "n": int(coords.shape[0]),
            "coords": coords.tolist(),
            "raw_coords": raw.tolist(),
            "dt": float(dt),
            "encoder": "synthetic",
            "var_names": ["x", "y", "z"],
            "meta": {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3},
        }
    )


class DynamicsRequest(__import__("pydantic").BaseModel):
    coords: list[list[float]]


@app.post("/api/dynamics")
def dynamics_endpoint(req: DynamicsRequest) -> JSONResponse:
    """Analyse dynamique d'une trajectoire (coords envoyees par le client)."""
    coords = np.asarray(req.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 points")
    t0 = time.time()
    stats = analyse_trajectory(coords)
    stats["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(stats)


@app.post("/api/dynamics_native/{cache_key}")
def dynamics_native_endpoint(cache_key: str) -> JSONResponse:
    """Dynamics computed DIRECTLY on high-dim latents (bypass UMAP).

    Évite artefacts de projection : Lyapunov / corr_dim / RQA mesurés
    dans l'espace original (e.g., 2048d ResNet50). Référence ground-truth
    vs analyse sur projection 3D UMAP.
    """
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    stats = analyse_trajectory(latents)
    stats["compute_s"] = round(time.time() - t0, 3)
    stats["space"] = f"native_{latents.shape[1]}d"
    return JSONResponse(stats)


@app.post("/api/topology_native/{cache_key}")
def topology_native_endpoint(cache_key: str, max_dim: int = 1) -> JSONResponse:
    """Persistent homology DIRECTLY on high-dim latents.

    Bypasse projection 3D UMAP. PH dans espace original = topology
    sans déformation. Plus coûteux mais ground-truth.
    """
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    if latents.shape[0] > 300:
        # Subsample pour rester tractable en haute-dim
        idx = np.linspace(0, latents.shape[0] - 1, 300).astype(int)
        latents = latents[idx]
    t0 = time.time()
    global_ph = persistent_homology(latents, max_dim=max_dim, max_n=300)
    out = {
        "global": global_ph,
        "space": f"native_{latents.shape[1]}d",
        "compute_s": round(time.time() - t0, 3),
    }
    return JSONResponse(out)


class SindyRequest(__import__("pydantic").BaseModel):
    coords: list[list[float]]
    dt: float = 1.0
    order: int = 2
    threshold: float = 0.05
    var_names: list[str] | None = None


@app.post("/api/sindy")
def sindy_endpoint(req: SindyRequest) -> JSONResponse:
    """SINDy : trouve equations differentielles depuis trajectoire."""
    coords = np.asarray(req.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 30:
        raise HTTPException(status_code=400, detail="Need >= 30 points")
    t0 = time.time()
    out = fit_sindy(
        coords,
        dt=req.dt,
        order=req.order,
        threshold=req.threshold,
        var_names=req.var_names,
    )
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/emergence")
def emergence_endpoint(req: DynamicsRequest) -> JSONResponse:
    """Effective information ladder + phi_id approx (Hoel)."""
    coords = np.asarray(req.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 30:
        raise HTTPException(status_code=400, detail="Need >= 30 points")
    t0 = time.time()
    out = effective_information_ladder(coords)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/causal")
def causal_endpoint(req: DynamicsRequest) -> JSONResponse:
    """Granger + TE + CCM (basique). Pour PCMCI+ + KSG, utiliser /api/causal_advanced."""
    coords = np.asarray(req.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 20:
        raise HTTPException(status_code=400, detail="Need >= 20 points")
    t0 = time.time()
    out = causal_summary(coords, lag=2)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/causal_advanced")
def causal_advanced_endpoint(req: DynamicsRequest, tau_max: int = 3, pc_alpha: float = 0.05) -> JSONResponse:
    """PCMCI+ multi-variate causal discovery + KSG TE k-NN estimator.

    Stronger than basic Granger : FDR control, non-linear options,
    continuous estimator (no binning artifacts).
    """
    series = np.asarray(req.coords, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] < 30:
        raise HTTPException(status_code=400, detail="Need >= 30 points")
    if series.shape[1] > 8:
        # Reduce to top 6 by variance for tractability
        var = series.var(axis=0)
        keep = np.argsort(var)[-6:]
        series = series[:, keep]
    t0 = time.time()
    out = causal_summary_advanced(series, tau_max=tau_max, pc_alpha=pc_alpha)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/topology")
def topology_endpoint(req: DynamicsRequest) -> JSONResponse:
    """Persistent homology global + sliding window."""
    coords = np.asarray(req.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 20:
        raise HTTPException(status_code=400, detail="Need >= 20 points")
    t0 = time.time()
    global_ph = persistent_homology(coords, max_dim=1, max_n=400)
    sw = sliding_window_ph(coords, window=80, stride=40, max_dim=1)
    out = {
        "global": global_ph,
        "sliding": sw,
        "compute_s": round(time.time() - t0, 3),
    }
    return JSONResponse(out)


@app.post("/api/spectral")
def spectral_endpoint(req: DynamicsRequest) -> JSONResponse:
    """DMD/Koopman + power spectrum + multiscale entropy ladder."""
    coords = np.asarray(req.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 16:
        raise HTTPException(status_code=400, detail="Need at least 16 points")
    t0 = time.time()
    out = {
        "dmd": dmd(coords, dt=1.0),
        "power": power_spectrum(coords),
        "multiscale": multiscale_entropy(coords),
        "spectral_slope": round(spectral_slope(coords), 4),
        "compute_s": round(time.time() - t0, 3),
    }
    return JSONResponse(out)


@app.get("/api/encoders")
def list_encoders() -> dict:
    return {"available": list(ENCODER_REGISTRY.keys()), "device": str(get_device())}


# ──────────────────────────────────────────────────────────────────
# Phase 6 : systems zoo + experiment registry + v1 versioning
# ──────────────────────────────────────────────────────────────────


@app.get("/api/v1/systems")
def systems_list() -> dict:
    return {"systems": list_systems()}


@app.get("/api/v1/systems/{system_id}")
def systems_generate(system_id: str, n: int | None = None) -> JSONResponse:
    if system_id not in SYSTEMS:
        raise HTTPException(status_code=404, detail=f"Unknown system: {system_id}")
    kwargs = {"n": n} if n else {}
    out = SYSTEMS[system_id](**kwargs)
    out["n"] = len(out["coords"])
    out["encoder"] = "synthetic"
    # Vitesse phase space vraie depuis raw_coords (vitesse physique)
    raw = np.asarray(out["raw_coords"], dtype=np.float64)
    diffs = np.linalg.norm(raw[1:] - raw[:-1], axis=1)
    eucl_v = np.concatenate([[0.0], diffs]).astype(np.float32)
    out["perceptual_velocity"] = eucl_v.tolist()
    out["euclidean_velocity"] = eucl_v.tolist()
    out["velocity_metric"] = "phase_space_euclidean"
    return JSONResponse(out)


@app.get("/api/v1/experiments")
def experiments_list(limit: int = 50) -> dict:
    return {"experiments": list_experiments(limit=limit)}


@app.get("/api/v1/experiments/{exp_id}")
def experiments_get(exp_id: str) -> JSONResponse:
    exp = get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return JSONResponse(exp)


class ExperimentSaveRequest(__import__("pydantic").BaseModel):
    source: str
    source_kind: str  # "video" | "system" | "lorenz"
    payload: dict
    encoder: str | None = None
    tags: list[str] | None = None


@app.post("/api/v1/experiments")
def experiments_save(req: ExperimentSaveRequest) -> dict:
    exp_id = save_experiment(
        source=req.source,
        source_kind=req.source_kind,
        payload=req.payload,
        encoder=req.encoder,
        tags=req.tags,
    )
    return {"id": exp_id, "ok": True}


@app.delete("/api/v1/experiments/{exp_id}")
def experiments_delete(exp_id: str) -> dict:
    ok = delete_experiment(exp_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"ok": True}


def _video_hash(video_path: Path, encoder: str, n_frames: int) -> str:
    stat = video_path.stat()
    key = f"{video_path.name}-{stat.st_size}-{stat.st_mtime}-{encoder}-{n_frames}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


_LATENTS_CACHE: dict[str, np.ndarray] = {}
_CLIP_LATENTS_CACHE: dict[str, np.ndarray] = {}  # CLIP features paralleles pour query/labels
_COORDS_CACHE: dict[str, np.ndarray] = {}        # dernière projection 3D pour rollout ghost
_AUDIO_LATENTS_CACHE: dict[str, np.ndarray] = {} # wav2vec2 features (N, 768)


@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    encoder: str = "resnet50",
    max_frames: int = 200,
    reducer: str = "umap",
) -> JSONResponse:
    if encoder not in ENCODER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown encoder: {encoder}")

    # Save upload
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = VIDEOS_DIR / f"upload_{int(time.time())}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cache_key = _video_hash(dest, encoder, max_frames)
    # Cache key inclut le reducer pour stocker plusieurs projections du meme latents
    cache_file = CACHE_DIR / f"{cache_key}__{reducer}.json"
    if cache_file.exists():
        return JSONResponse(json.loads(cache_file.read_text()))

    t0 = time.time()
    frames, indices = sample_frames(dest, max_frames=max_frames)
    t_sample = time.time() - t0

    # Sauvegarde miniatures pour player synchronise
    thumb_dir = FRAMES_DIR / cache_key
    thumb_names = save_thumbnails(frames, thumb_dir, width=240, quality=72)
    frame_urls = [f"/frames/{cache_key}/{n}" for n in thumb_names]

    t0 = time.time()
    enc = _get_or_load_encoder(encoder)
    latents = enc.encode(frames)
    t_encode = time.time() - t0

    # Vitesse perceptuelle : cosine distance frame-a-frame dans espace latent ORIGINAL
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    units = latents / norms
    cos_sim = np.sum(units[:-1] * units[1:], axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    perceptual_v = np.concatenate([[0.0], 1.0 - cos_sim]).astype(np.float32)
    # Vitesse euclidienne brute pour comparaison
    eucl_v = np.concatenate([[0.0], np.linalg.norm(latents[1:] - latents[:-1], axis=1)]).astype(np.float32)

    t0 = time.time()
    coords = reduce_3d(latents, method=reducer)
    t_reduce = time.time() - t0
    # Stocke latents pour re-reduction sans re-encode
    _LATENTS_CACHE[cache_key] = latents
    _COORDS_CACHE[cache_key] = coords
    # Auto-encode CLIP parallele (sauf si deja CLIP) pour activer query/labels universels
    if encoder != "clip_vit_b32":
        try:
            clip_enc = _get_or_load_encoder("clip_vit_b32")
            _CLIP_LATENTS_CACHE[cache_key] = clip_enc.encode(frames)
        except Exception:
            pass  # CLIP optionnel
    else:
        _CLIP_LATENTS_CACHE[cache_key] = latents

    meta = video_meta(dest)
    payload = {
        "name": file.filename or dest.name,
        "n": int(coords.shape[0]),
        "coords": coords.tolist(),
        "frame_indices": indices,
        "frame_urls": frame_urls,
        "cache_key": cache_key,
        "encoder": encoder,
        "latent_dim": int(latents.shape[1]),
        "perceptual_velocity": perceptual_v.tolist(),
        "euclidean_velocity": eucl_v.tolist(),
        "velocity_metric": "cosine_distance",
        "reducer": reducer,
        "meta": meta,
        "timings": {
            "sample_s": round(t_sample, 3),
            "encode_s": round(t_encode, 3),
            "reduce_s": round(t_reduce, 3),
        },
    }
    # Encode audio aligne en background-blocking (apres meta defini)
    try:
        audio_feats = encode_audio_aligned(
            dest, frame_indices=indices, fps=meta.get("fps", 30.0), device=get_device(),
        )
        if audio_feats is not None:
            _AUDIO_LATENTS_CACHE[cache_key] = audio_feats
            payload["has_audio"] = True
            payload["audio_dim"] = int(audio_feats.shape[1])
    except Exception:
        pass

    cache_file.write_text(json.dumps(payload))
    return JSONResponse(payload)


@app.get("/api/audio/{cache_key}")
def audio_trajectory(cache_key: str, reducer: str = "umap") -> JSONResponse:
    """Retourne trajectoire 3D projetée depuis features audio wav2vec2."""
    if cache_key not in _AUDIO_LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No audio for this cache_key")
    feats = _AUDIO_LATENTS_CACHE[cache_key]
    coords = reduce_3d(feats, method=reducer)
    return JSONResponse({
        "coords": coords.tolist(),
        "n": int(coords.shape[0]),
        "audio_dim": int(feats.shape[1]),
        "reducer": reducer,
    })


@app.get("/api/cross_modal/{cache_key}")
def cross_modal_causal(cache_key: str) -> JSONResponse:
    """Cross-modal : compute TE/causal entre features visuelles et audio."""
    if cache_key not in _LATENTS_CACHE or cache_key not in _AUDIO_LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="Need both visual and audio latents")
    visual = _LATENTS_CACHE[cache_key]
    audio = _AUDIO_LATENTS_CACHE[cache_key]
    # Reduce both to 4D pour rendre causal_summary calculable
    from .reducer import reduce_3d as _r
    v_reduced = _r(visual, method="pca")
    a_reduced = _r(audio, method="pca")
    # Concatene 6D (3 visual + 3 audio)
    joint = np.concatenate([v_reduced, a_reduced], axis=1)
    out = causal_summary(joint, lag=2)
    # Renomme labels pour clarte
    labels = ["v0", "v1", "v2", "a0", "a1", "a2"]
    out["labels"] = labels[:joint.shape[1]]
    return JSONResponse(out)


@app.get("/api/nearest/{cache_key}")
def nearest_neighbors(cache_key: str, idx: int, k: int = 8) -> JSONResponse:
    """Voisins dans espace latent original (2048d), pas UMAP.
    Retourne indices + cosine distances."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    n = latents.shape[0]
    if not (0 <= idx < n):
        raise HTTPException(status_code=400, detail=f"idx {idx} out of [0, {n})")
    q = latents[idx:idx+1]
    # Cosine distance pour interpretabilite
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    ln = latents / np.maximum(np.linalg.norm(latents, axis=1, keepdims=True), 1e-9)
    sims = (ln @ qn.T).flatten()
    dists = 1.0 - sims
    # Exclude self
    order = np.argsort(dists)
    selected = [int(j) for j in order if j != idx][:k]
    return JSONResponse({
        "query_idx": idx,
        "neighbors": [
            {"idx": j, "cosine_dist": round(float(dists[j]), 4), "cosine_sim": round(float(sims[j]), 4)}
            for j in selected
        ],
    })


class TransitionsRequest(__import__("pydantic").BaseModel):
    labels: list[int]


@app.post("/api/transitions/{cache_key}")
def transitions_endpoint(cache_key: str, req: TransitionsRequest) -> JSONResponse:
    """Calcule transition matrix entre clusters sequentiels."""
    t0 = time.time()
    out = transition_matrix(req.labels)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


class ClusterLabelsRequest(__import__("pydantic").BaseModel):
    labels: list[int]      # per-frame cluster id (-1 = noise)
    vocab: list[str]       # candidate labels CLIP


@app.post("/api/label_clusters/{cache_key}")
def label_clusters_endpoint(cache_key: str, req: ClusterLabelsRequest) -> JSONResponse:
    """Pour chaque cluster, calcule centroid feature CLIP, retourne meilleur label vocab."""
    if cache_key not in _CLIP_LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No CLIP latents")
    if not req.vocab:
        raise HTTPException(status_code=400, detail="vocab vide")
    clip = _get_or_load_encoder("clip_vit_b32")
    clip_lat = _CLIP_LATENTS_CACHE[cache_key]
    if clip_lat.shape[0] != len(req.labels):
        raise HTTPException(status_code=400, detail=f"labels len {len(req.labels)} != n_frames {clip_lat.shape[0]}")

    # Encode vocab textes
    text_feats = clip.encode_text(req.vocab)
    tn = text_feats / np.maximum(np.linalg.norm(text_feats, axis=1, keepdims=True), 1e-9)
    in_ = clip_lat / np.maximum(np.linalg.norm(clip_lat, axis=1, keepdims=True), 1e-9)

    labels_arr = np.asarray(req.labels)
    out: dict[int, dict] = {}
    for cid in sorted(set(req.labels)):
        if cid == -1:
            continue
        mask = labels_arr == cid
        centroid = in_[mask].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-9)
        sims = (tn @ centroid).flatten()
        ranked = np.argsort(-sims)
        out[int(cid)] = {
            "cluster_id": int(cid),
            "size": int(mask.sum()),
            "best_label": req.vocab[int(ranked[0])],
            "best_sim": round(float(sims[int(ranked[0])]), 4),
            "top3": [
                {"label": req.vocab[int(i)], "sim": round(float(sims[int(i)]), 4)}
                for i in ranked[:3]
            ],
        }
    return JSONResponse({"clusters": list(out.values())})


@app.post("/api/live/start")
def live_start(encoder: str = "resnet50") -> JSONResponse:
    if encoder not in ENCODER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown encoder: {encoder}")
    _get_or_load_encoder(encoder)  # warm load
    sess = create_session(encoder_name=encoder, frames_dir=FRAMES_DIR)
    return JSONResponse({"session_id": sess.id, "encoder": encoder})


@app.post("/api/live/frame/{sid}")
async def live_frame(sid: str, file: UploadFile = File(...)) -> JSONResponse:
    """Recoit 1 frame JPEG via webcam, encode, append session, projette + retourne coords."""
    sess = get_session(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    # Decode image bytes
    import cv2
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Cannot decode frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # Resize to 224 carre
    sized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    frame_norm = (sized.astype(np.float32) / 255.0)[None, ...]  # (1, 224, 224, 3)

    # Save thumbnail
    t0 = time.time()
    thumb_name = f"{sess.n:04d}.jpg"
    cv2.imwrite(
        str(sess.frames_dir / thumb_name),
        cv2.resize(bgr, (240, int(240 * h / w)), interpolation=cv2.INTER_AREA),
        [int(cv2.IMWRITE_JPEG_QUALITY), 65],
    )

    # Encode
    enc = _get_or_load_encoder(sess.encoder_name)
    latent = enc.encode(frame_norm)[0]
    t_encode = time.time() - t0

    sess.append(latent, thumb_name)

    # Project current trajectory
    t0 = time.time()
    if sess.n >= 5:
        method = "umap" if sess.n >= 12 else "pca"
        coords = reduce_3d(sess.stacked_latents(), method=method, n_neighbors=min(15, sess.n - 1))
    else:
        # Trop peu de points, simple PCA
        coords = reduce_3d(sess.stacked_latents(), method="pca")
    t_reduce = time.time() - t0

    # Velocity perceptuelle live
    latents_all = sess.stacked_latents()
    norms = np.linalg.norm(latents_all, axis=1, keepdims=True)
    units = latents_all / np.maximum(norms, 1e-9)
    if latents_all.shape[0] > 1:
        cos_sim = np.sum(units[:-1] * units[1:], axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        perceptual_v = np.concatenate([[0.0], 1.0 - cos_sim]).astype(np.float32)
    else:
        perceptual_v = np.zeros(1, dtype=np.float32)

    return JSONResponse({
        "session_id": sid,
        "n": sess.n,
        "coords": coords.tolist(),
        "frame_urls": [f"/frames/live_{sid}/{p}" for p in sess.thumb_paths],
        "perceptual_velocity": perceptual_v.tolist(),
        "velocity_metric": "cosine_distance",
        "encoder": sess.encoder_name,
        "latent_dim": int(latents_all.shape[1]),
        "timings": {"encode_s": round(t_encode, 3), "reduce_s": round(t_reduce, 3)},
    })


@app.post("/api/live/stop/{sid}")
def live_stop(sid: str) -> JSONResponse:
    end_session(sid)
    return JSONResponse({"ok": True})


@app.post("/api/counterfactual/{cache_key}")
def counterfactual_endpoint(
    cache_key: str, start_idx: int, horizon: int = 20,
    n_samples: int = 8, noise_scale: float = 0.05,
    epochs: int = 100, hidden: int = 256,
) -> JSONResponse:
    """N rollouts depuis latent perturbé — cone of futures."""
    if cache_key not in _LATENTS_CACHE or cache_key not in _COORDS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents/coords")
    latents = _LATENTS_CACHE[cache_key]
    coords = _COORDS_CACHE[cache_key]
    t0 = time.time()
    device_str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = counterfactual_rollouts(
        latents, coords, start_idx=start_idx, horizon=horizon,
        n_samples=n_samples, noise_scale=noise_scale,
        epochs=epochs, hidden=hidden, device=device_str,
    )
    out["compute_s"] = round(time.time() - t0, 3)
    out["device"] = device_str
    return JSONResponse(out)


@app.post("/api/rollout/{cache_key}")
def rollout_endpoint(
    cache_key: str, start_idx: int, horizon: int = 20,
    epochs: int = 120, hidden: int = 256,
) -> JSONResponse:
    """Rollout MLP itératif depuis start_idx. Ghost trajectoire 3D."""
    if cache_key not in _LATENTS_CACHE or cache_key not in _COORDS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents/coords")
    latents = _LATENTS_CACHE[cache_key]
    coords = _COORDS_CACHE[cache_key]
    if latents.shape[0] < 10:
        raise HTTPException(status_code=400, detail="Need >= 10 frames")
    t0 = time.time()
    device_str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = rollout_fn(latents, coords, start_idx=start_idx, horizon=horizon,
                    epochs=epochs, hidden=hidden, device=device_str)
    out["compute_s"] = round(time.time() - t0, 3)
    out["device"] = device_str
    return JSONResponse(out)


@app.post("/api/predict/{cache_key}")
def predict_endpoint(cache_key: str, epochs: int = 150, hidden: int = 256) -> JSONResponse:
    """Train MLP latent predictor sur cached latents. Retourne surprise per-frame."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    if latents.shape[0] < 10:
        raise HTTPException(status_code=400, detail="Need >= 10 frames")
    t0 = time.time()
    device_str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = fit_predictor(latents, epochs=epochs, hidden=hidden, device=device_str)
    out["compute_s"] = round(time.time() - t0, 3)
    out["device"] = device_str
    return JSONResponse(out)


@app.post("/api/evolution/{cache_key}")
def evolution_endpoint(cache_key: str, window: int = 24, stride: int = 8) -> JSONResponse:
    """Sliding-window analytics : metrics evolution per time chunk."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    out = evolution_pipeline(latents, window=window, stride=stride)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/benchmark/{cache_key}")
def benchmark_endpoint(cache_key: str, dt: float = 1.0) -> JSONResponse:
    """Cross-tool benchmark : compare nos SINDy + DMD vs pysindy + scipy reference.

    Valide correctness numérique + positionne performances.
    Returns agreement metrics + speed comparison.
    """
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    if latents.shape[0] < 30:
        raise HTTPException(status_code=400, detail="Need >= 30 points")
    # Reduce dims for tractability if too high
    if latents.shape[1] > 8:
        var = latents.var(axis=0)
        keep = np.argsort(var)[-3:]  # top-3 PC pour SINDy/DMD
        latents = latents[:, keep]
    t0 = time.time()
    out = benchmark_summary(latents, dt=dt)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/wavelets/{cache_key}")
def wavelets_endpoint(
    cache_key: str, wavelet: str = "db4", max_level: int | None = None,
) -> JSONResponse:
    """Multi-resolution wavelet decomposition (PyWavelets DWT).

    Choices wavelet : db4 (default), haar, sym8, coif5.
    Returns energy per scale + dominant scale (Mallat-style multi-scale signature).
    """
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    out = wavelet_per_dim(latents, wavelet=wavelet, max_level=max_level)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/sfa/{cache_key}")
def sfa_endpoint(cache_key: str, n_components: int = 8, polynomial_expand: int = 1) -> JSONResponse:
    """Slow Feature Analysis. Retourne k composantes les plus lentes."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    out = slow_feature_analysis(latents, n_components=n_components, polynomial_expand=polynomial_expand)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/dna/{cache_key}")
def dna_endpoint(cache_key: str) -> JSONResponse:
    """Composite complexity score depuis tous les analytics."""
    if cache_key not in _LATENTS_CACHE or cache_key not in _COORDS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents/coords")
    latents = _LATENTS_CACHE[cache_key]
    coords = _COORDS_CACHE[cache_key]
    if latents.shape[0] < 20:
        raise HTTPException(status_code=400, detail="Need >= 20 frames")
    t0 = time.time()
    out = compute_dna(latents, coords)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/cluster/{cache_key}")
def cluster_endpoint(
    cache_key: str,
    min_cluster_size: int = 5,
    metric: str = "euclidean",
) -> JSONResponse:
    """HDBSCAN cluster discovery sur latents caches 2048d/512d/384d."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    out = cluster_latents(latents, min_cluster_size=min_cluster_size, metric=metric)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.get("/api/clip_query/{cache_key}")
def clip_text_query(cache_key: str, text: str, top_k: int = 10) -> JSONResponse:
    """Encode texte CLIP, cosine sim avec features CLIP paralleles.
    Marche pour TOUS encoders (CLIP auto-encode en background)."""
    if cache_key not in _CLIP_LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No CLIP latents (re-upload video)")
    latents = _CLIP_LATENTS_CACHE[cache_key]
    clip = _get_or_load_encoder("clip_vit_b32")
    txt_feats = clip.encode_text([text])
    # Normalise pour cosine
    tn = txt_feats / np.maximum(np.linalg.norm(txt_feats, axis=1, keepdims=True), 1e-9)
    ln = latents / np.maximum(np.linalg.norm(latents, axis=1, keepdims=True), 1e-9)
    sims = (ln @ tn.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return JSONResponse({
        "query": text,
        "results": [
            {"idx": int(j), "similarity": round(float(sims[j]), 4)}
            for j in order
        ],
        "similarities": sims.tolist(),
    })


@app.post("/api/segments/{cache_key}")
def segments_endpoint(
    cache_key: str,
    method: str = "adaptive",
    sensitivity: float = 1.0,
    min_segment: int = 4,
) -> JSONResponse:
    """Détecte boundaries depuis velocity perceptuelle. methods: adaptive | percentile | std."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    out = find_boundaries(latents, method=method, sensitivity=sensitivity, min_segment=min_segment)
    out["compute_s"] = round(time.time() - t0, 3)
    return JSONResponse(out)


@app.post("/api/autotune/{cache_key}")
def autotune_endpoint(
    cache_key: str,
    objective: str = "composite",
    n_trials: int = 20,
    method: str = "bayesian",
) -> JSONResponse:
    """Search params UMAP. method='bayesian' (gp_minimize) ou 'random'."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents")
    latents = _LATENTS_CACHE[cache_key]
    if latents.shape[0] < 10:
        raise HTTPException(status_code=400, detail="Need >= 10 frames")
    if method == "random":
        result = random_search(latents, n_trials=n_trials, objective=objective)
    else:
        try:
            result = bayesian_search(latents, n_trials=n_trials, objective=objective)
        except Exception:
            # Fallback random si skopt indisponible
            result = random_search(latents, n_trials=n_trials, objective=objective)
    return JSONResponse(result)


@app.get("/api/compare")
def compare_videos(
    key1: str, key2: str, reducer: str = "umap",
    n_neighbors: int = 15, min_dist: float = 0.1, metric: str = "cosine",
) -> JSONResponse:
    """Fit UMAP partagé sur union latents de 2 vidéos. Encoder doit matcher."""
    if key1 not in _LATENTS_CACHE or key2 not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="One or both cache_keys missing")
    a = _LATENTS_CACHE[key1]
    b = _LATENTS_CACHE[key2]
    if a.shape[1] != b.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Dim mismatch : {a.shape[1]} vs {b.shape[1]} (same encoder required)",
        )
    t0 = time.time()
    union = np.concatenate([a, b], axis=0)
    union_coords = reduce_3d(
        union, method=reducer,
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
    )
    coords_a = union_coords[: a.shape[0]]
    coords_b = union_coords[a.shape[0] :]

    # Distance moyenne entre trajectoires (Procrustes-like : juste mean pairwise sur min len)
    n_min = min(len(coords_a), len(coords_b))
    pair_diffs = np.linalg.norm(coords_a[:n_min] - coords_b[:n_min], axis=1)
    mean_dist = float(pair_diffs.mean())

    # Distance entre centroids
    centroid_dist = float(np.linalg.norm(coords_a.mean(axis=0) - coords_b.mean(axis=0)))

    return JSONResponse({
        "coords_a": coords_a.tolist(),
        "coords_b": coords_b.tolist(),
        "n_a": int(a.shape[0]),
        "n_b": int(b.shape[0]),
        "dim": int(a.shape[1]),
        "mean_pair_distance": mean_dist,
        "centroid_distance": centroid_dist,
        "compute_s": round(time.time() - t0, 3),
    })


@app.post("/api/reduce/{cache_key}")
def re_reduce(
    cache_key: str,
    reducer: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    smoothing: int = 0,
) -> JSONResponse:
    """Re-project latents cached avec params custom."""
    if cache_key not in _LATENTS_CACHE:
        raise HTTPException(status_code=404, detail="No cached latents for that key. Re-upload first.")
    latents = _LATENTS_CACHE[cache_key]
    t0 = time.time()
    coords = reduce_3d(
        latents, method=reducer,
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, smoothing=smoothing,
    )
    t_reduce = time.time() - t0
    _COORDS_CACHE[cache_key] = coords
    return JSONResponse({
        "coords": coords.tolist(),
        "n": int(coords.shape[0]),
        "reducer": reducer,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "smoothing": smoothing,
        "compute_s": round(t_reduce, 3),
    })


# Static frontend + frames thumbnails
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/videos_static", StaticFiles(directory=VIDEOS_DIR), name="videos_static")


@app.get("/")
def index() -> FileResponse:
    idx = FRONTEND_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="Frontend missing")
    return FileResponse(idx)


@app.get("/favicon.ico")
def favicon():
    return JSONResponse({}, status_code=204)
