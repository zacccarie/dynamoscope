# DYNAMOSCOPE

Video → latent trajectory → emergence analysis.

Experimental research tool to extract emergent structures, causal dynamics, and multi-scale patterns from video by treating it as a trajectory through state space.

## Pipeline

```
video / webcam / synthetic system
        ↓
  ingestion (frame sampling, thumbnails)
        ↓
  encoder (ResNet50 / ViT-B / DINOv2 / CLIP)  ← also wav2vec2 audio in parallel
        ↓
  reducer (UMAP / PCA / Isomap)  ← params tunable + auto-tune (random / bayesian)
        ↓
  3D trajectory
        ↓
  analysis modules
   ├── dynamics (Lyapunov, RQA, Takens, correlation dim)
   ├── spectral (DMD / Koopman, multiscale entropy)
   ├── causal (Granger, transfer entropy, CCM)
   ├── topology (persistent homology H0/H1, sliding window)
   ├── sindy (sparse equation discovery)
   ├── emergence (effective info, φ-id approx)
   ├── segments (shot boundary via perceptual velocity)
   ├── clusters (HDBSCAN auto-discovery + CLIP labels)
   ├── predict (latent MLP + surprise per frame)
   ├── rollout (n-step prediction + counterfactual cone)
   ├── sfa (Slow Feature Analysis)
   ├── evolution (sliding window metrics)
   ├── dna (composite complexity score 7-axes)
   └── transitions (Markov chain between clusters)
        ↓
  3D visualization (Three.js r170, bloom, axes overlay)
   ├── 5 view modes : phase space / time helix / Takens / polar / audio / SFA
   ├── 4 color modes : time / velocity UMAP / velocity perceptual / cluster / surprise
   └── 12 analysis tabs in right sidebar
        ↓
  workspace
   ├── save runs (SQLite registry)
   ├── load history
   ├── multi-video compare (shared UMAP fit)
   ├── HTML report export (standalone, embedded snapshots)
   └── CLIP text query frames
```

## Stack

**Backend** : Python 3.13, FastAPI, PyTorch 2.12 (MPS/CUDA/CPU), UMAP, sklearn, ripser, scikit-optimize, transformers, open_clip_torch.
**Frontend** : Three.js r170, vanilla JS modules, CSS Grid layout.
**Storage** : zarr-style cache + SQLite registry + JSON payloads per run.

## Run

### Web UI

```bash
uv venv --python 3.13
uv pip install -e .
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8770
```

Open http://127.0.0.1:8770

### CLI (headless batch)

```bash
# Process 1 video, full analyses
python -m backend.cli process video.mp4 --encoder dinov2_vits14 --analyses all --out result.json

# Batch process folder
python -m backend.cli batch ./videos --out-dir ./results --encoder resnet50

# Just composite DNA score
python -m backend.cli dna video.mp4

# Compare 2 videos
python -m backend.cli compare a.mp4 b.mp4

# Synthetic system
python -m backend.cli system lorenz --analyses all
```

### Tests

```bash
.venv/bin/python -m pytest tests/   # 50 tests, ~4s
```

## Modules backend

```
backend/
  main.py            FastAPI app, all routes, in-memory caches
  ingestion.py       frame sampling + thumbnails
  encoder.py         4 encoders : ResNet50, ViT-B, DINOv2, CLIP
  audio.py           wav2vec2 audio features
  reducer.py         UMAP / PCA / Isomap + smoothing
  dynamics.py        Lyapunov (Rosenstein), RQA, Takens embedding
  spectral.py        DMD / Koopman, power spectrum
  multiscale.py      coarse-grain entropy ladder, spectral slope
  causal.py          Granger, transfer entropy, CCM (Sugihara)
  topology.py        persistent homology, sliding window
  sindy.py           polynomial library + STLSQ sparse regression
  emergence.py       effective info, φ-id approx
  segmentation.py    shot boundaries via velocity peaks
  autotune.py        random + Bayesian GP-EI param search
  clustering.py      HDBSCAN + transition matrix
  prediction.py      LatentMLP + rollout + counterfactual cone
  sfa.py             Slow Feature Analysis (Wiskott-Sejnowski)
  evolution.py       sliding window metrics evolution
  dna.py             composite 7-axis complexity score
  live.py            webcam session management
  registry.py        SQLite experiments registry
  systems.py         6 canonical dynamical systems (Lorenz, Rössler, etc.)
```

## API

All routes under `/api`. OpenAPI auto-docs at `/docs`.

Key endpoints :
- `POST /api/process` upload video, encode, reduce, return coords
- `GET /api/v1/systems/{id}` synthetic system demo
- `POST /api/reduce/{cache_key}` re-project cached latents with new params
- `POST /api/dynamics|spectral|causal|topology|sindy|emergence|...` per-analysis
- `POST /api/dna/{cache_key}` composite score
- `POST /api/autotune/{cache_key}` Bayesian param search
- `POST /api/predict|rollout|counterfactual/{cache_key}` predictive analysis
- `GET /api/clip_query/{cache_key}?text=...` semantic video search
- `POST /api/live/start|frame|stop` real-time webcam pipeline
- `GET /api/compare?key1=A&key2=B` multi-video shared UMAP

## Theoretical anchors

| Concept | Author | Location in tool |
|---|---|---|
| Takens embedding | Takens 1981 | dynamics.py · view mode `takens_delay` |
| Information bottleneck | Tishby | implicit in encoder choice + reducer |
| Free energy principle | Friston | predict.py surprise + counterfactual |
| Causal emergence | Hoel | emergence.py φ_id |
| Renormalization group | Wilson, Mehta-Schwab | multiscale.py entropy ladder |
| Koopman theory | Koopman 1931 · Brunton | spectral.py |
| Geometric deep learning | Bronstein | encoders + scattering inheritance |
| Slow Feature Analysis | Wiskott-Sejnowski 2002 | sfa.py |
| Persistent homology | Edelsbrunner | topology.py |
| Predictive coding | Rao-Ballard · Friston | prediction.py rollout |
| SINDy | Brunton 2016 | sindy.py |
| JEPA | LeCun 2023 | counterfactual cone |
| HDBSCAN | Campello 2013 | clustering.py |

## Frontend layout

```
┌──── TOP BAR : brand · stats · device ─────────────────────────┐
│                                                                │
│  LEFT SIDEBAR        3D CANVAS (Three.js)        RIGHT SIDEBAR │
│   · video input        · trajectory                · tabs (12) │
│   · synthetic system   · ghost rollouts            · metrics   │
│   · view mode          · counterfactual cone       · canvases  │
│   · color mode         · cluster colors            · buttons   │
│   · embedding params   · marker actif                          │
│   · auto-tune                                                  │
│   · player + scrubber                                          │
│   · workspace (save/export/history/overlay)                    │
│                                                                │
├──── BOTTOM BAR : frame counter · scrubber · status ───────────┤
└────────────────────────────────────────────────────────────────┘
```

12 tabs analyse droite :
`dynamics` · `spectral` · `causal` · `topology` · `SINDy · φ` · `inspect` · `segments` · `clusters` · `predict` · `DNA` · `evolution` · `status`

## License

Research project. Personal use.
