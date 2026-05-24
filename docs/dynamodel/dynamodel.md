# Dynamodel — Organigramme + Schema exhaustif

> Architecture du **RegimeWorldModel** Phase 1 + Phase 2.
> Régime-conditional mixture-of-experts world model.

## Visualisations

- **`dynamodel.svg`** — diagramme vectoriel haute-fidélité (browser-renderable)
- **`dynamodel.mmd`** — diagramme Mermaid (rendu auto sur GitHub)
- **`dynamodel.tex`** — TikZ pour intégration dans paper LaTeX (§10)
- **`dynamodel.md`** — ce fichier (organigramme ASCII + explainer)

---

## Organigramme ASCII

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      INPUT                              │
                    │   frames (N, H, W, 3)  OR  raw obs trajectory (T, d_in)│
                    └────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
                    ┌────────────────────────────────────────────────────────┐
                    │              PERCEPTUAL ENCODER                         │
                    │   MLP 2-layer (Phase 1)  OR  DINOv2-S frozen (Phase 2) │
                    │              x_t  ──→  z_fast_t ∈ ℝ^D_fast              │
                    └─┬──────────────────────────────┬───────────────────────┘
                      │                              │
                      │ z_fast (per-step)            │ z_fast (per-step)
                      │                              │
                      ▼                              ▼
       ┌──────────────────────────┐     ┌──────────────────────────┐
       │      FAST GRU            │     │       DECODER            │
       │   per-step update        │     │   2-layer MLP            │
       │   h_fast_t = GRU(z, h)   │     │   D_fast → 64 → d_in     │
       └──────────┬───────────────┘     │   x̂_t = Dec(z_fast_t)    │
                  │                      └────────────┬─────────────┘
                  │ h_fast every step                  │ x̂
                  ▼                                    │
       ┌──────────────────────────┐                    │
       │      SLOW GRU            │                    │
       │   every k=4 steps        │                    │
       │   h_slow ← GRU(mean(     │                    │
       │      h_fast_window),      │                    │
       │      h_slow)              │                    │
       │   → z_slow ∈ ℝ^(T_s,D_s) │                    │
       └────┬─────────────────────┘                    │
            │                                          │
            │ z_slow (regime-bearing)                  │
            │                                          │
            ▼                                          │
┌────────────────────────────────────────────┐         │
│        DESCRIPTOR HEADS                     │         │
│   d_dyn = (slow, recur, lyap*, var) ∈ ℝ^4   │         │
│   computed from z_slow window                │         │
├──────────────┬─────────────┬────────────────┤         │
│  SLOWNESS    │ RECURRENCE  │  LYAP PROXY    │         │
│  ‖Δz_std‖²   │ σ((ε−d)/τ)  │  var(ln d_ij)  │         │
│              │ soft RQA    │                 │         │
└───────┬──────┴─────────────┴────────────────┘         │
        │                                                │
        │ d_dyn                                          │
        ▼                                                │
┌────────────────────────────┐                          │
│      REGIME ROUTER          │                          │
│   MLP(4 → 32 → 3) +         │                          │
│   softmax(./T)              │                          │
│   → r ∈ Δ²                  │                          │
│  (entropy_reg=0, sup_CE=5)  │                          │
└────────┬───────────────────┘                          │
         │                                              │
         │ r = (r_smooth, r_periodic, r_chaotic)        │
         │                                              │
         ▼                                              │
┌────────────────────────────────────────────────────┐  │
│         REGIME DISTRIBUTION Δ²                      │  │
│   (●S)  ──  smooth   ──  r_smooth     teal          │  │
│   (●P)  ──  periodic ──  r_periodic   yellow        │  │
│   (●C)  ──  chaotic  ──  r_chaotic    red           │  │
└──────────────────┬─────────────────────────────────┘  │
                   │                                     │
        ┌──────────┴────────┐                            │
        ▼                   ▼                            │
        ↓ gating weights into mixture dynamics           │
                                                         │
   z_slow ──────────┐                                   │
                    ▼                                    │
┌────────────────────────────────────────────────────────┐
│              REGIME MIXTURE DYNAMICS                    │
│  ẑ_{t+1} = Σ_k r_t^(k) · Φ_k(z_slow_t)                 │
├────────────────┬───────────────────┬───────────────────┤
│  Φ_SMOOTH      │   Φ_PERIODIC      │   Φ_CHAOTIC       │
│  init = 0.3    │   init = 0.6      │   init = 1.0      │
│  contractive   │   oscillatory     │   divergent       │
│  2-layer MLP   │   2-layer MLP     │   2-layer MLP     │
│  D_s→64→D_s    │   D_s→64→D_s      │   D_s→64→D_s      │
└────────┬───────┴────────┬──────────┴──────────┬────────┘
         │                │                      │
         └──────  Σ ◄─────┴──────────────────────┘
                  │
                  ▼
              ẑ_slow_{t+1}
                  │
                  └──────────────────────────────────────┐
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────┐
│              REGIME-CONDITIONAL LOSS                            │
│                                                                 │
│  L = L_recon                                                    │
│      + L_dyn (1-step ‖ẑ_{t+1} − z_{t+1}‖²)                    │
│      + (r_smooth + r_periodic) · L_slow      ← gated            │
│      + r_periodic · L_recurrence              ← gated            │
│      + r_chaotic  · L_lyap_proxy              ← gated            │
│      + β · L_entropy                          (β=0 Phase 2)     │
│      + α · L_regime_sup (CE r vs r*)          (α=5 Phase 2)     │
│                                                                 │
│  3-STAGE TRAINING SCHEDULE :                                    │
│   Stage 1 (30% epochs) : L_recon + L_dyn only                  │
│   Stage 2 (30% epochs) : + L_slow, L_recur, L_lyap (uniform)   │
│   Stage 3 (40% epochs) : + regime-cond weighting + L_regime_sup│
└────────────────────────────────────────────────────────────────┘
```

---

## Spécification dimensionnelle

| Composant | Input | Output | Params |
|---|---|---|---|
| PerceptualEncoder MLP (P1) | (T, d_in=3) | (T, D_fast=32) | ~2K |
| DINOv2-S frozen (P2) | (N, 224, 224, 3) | (N, 384) | 22M (frozen) |
| FastGRU | (T, D_fast) | (T, D_fast) | ~6K |
| SlowGRU | (T/k, D_fast) | (T/k, D_slow) | ~6K |
| DescriptorHeads | (T/k, D_slow) | (4,) | 0 (computed) |
| RegimeRouter | (4,) | (3,) | ~250 |
| 3× Φ_k | (T/k, D_slow) | (T/k, D_slow) | 3×4K |
| Decoder | (T, D_fast) | (T, d_in) | ~2K |
| **Total Phase 1** | | | **~30K** |
| **Total Phase 2** | | | **~134K** |

D_fast = 32 (P1) ou 64 (P2). D_slow = 32 (P1) ou 64 (P2). k=4 stride. T = sequence length.

---

## Données d'entraînement et expériences

### Phase 1 : raw observable trajectories
- **Synth zoo** : 8 systèmes × 3 régimes (Lorenz/Rössler/Hénon/VdP/Kuramoto/sin/damped/exp)
- **Window** : T=128 points
- **Training** : 60 samples/seed × 8 seeds × 50 epochs
- **Results** : RWM **3.3× better OOD** vs matched-capacity Flat-GRU (30K params)

### Phase 2 : real video features (DINOv2)
- **Procedural video** : 6 generators × 3 régimes
- **Frame resolution** : 224×224 (DINOv2 patch 14)
- **Feature cache** : `cache/regime_world_video/*.npy` (per-video features)
- **Window** : 64 frames, stride 16
- **Training** : 3 seeds × 40 epochs
- **Results router-fixed** : RWM **28× better OOD** vs matched Flat (134K each), router accuracy 0.84

---

## Loss hyperparameters

| Loss | Weight Phase 1 | Weight Phase 2 (final) |
|---|---|---|
| L_recon | 1.0 | 1.0 |
| L_dyn | 1.0 | 1.0 |
| L_slow | 1.0 | 1.0 |
| L_recurrence | 0.5 | 0.5 |
| L_lyap | 0.3 | 0.3 |
| **L_entropy (β)** | 0.1 | **0.0** (router fix) |
| **L_regime_sup (α)** | 1.0 | **5.0** (router fix) |

Router-fix discovered in §10.6 : entropy_reg at 0.1 was strangling router learning (33% random). Disabling + boosting sup CE → 84% router accuracy.

---

## Failure modes documentés

| # | Description | Status |
|---|---|---|
| F1 | Probe ceiling sur synth (AUROC saturé 1.0) | benchmark too easy |
| F2 | Unsupervised router collapse (33% random) | needs supervision |
| F3 | Supervised CE restores router (+30 pts, p=0.0007) | architectural fix |
| F4 | Single-regime training collapses router | data diversity required |
| F5 | Single-regime Φ overspecialization | Φ_k locked to training regime |
| F6 | RWM 4× WORSE than Flat under (F4) | architecture cost without benefit |
| F7 | Doubling Flat capacity HURTS OOD | overfitting without inductive bias |
| F8 | Router collapses on video DINOv2 features | hyperparameter issue (not architecture) |
| F9 | Seed-0 anomaly low MSE + low lstd | partial collapse but non-zero |

---

## Références

- Architecture inspirée de : Shazeer et al. 2017 (MoE), Hafner et al. (RSSM), Bardes 2024 (V-JEPA)
- Loss inspirations : Wiskott-Sejnowski 2002 (SFA), Schreiber 2000 (TE), Rosenstein 1993 (Lyap), Marwan (RQA)
- Code : `backend/regime_world/` (model.py, losses.py, trainer.py, synth.py, video_data.py)
- Tests : `tests/test_regime_world.py` (12 tests, all green)
- Experiments : `experiments/regime_world_*.py` (probe, OOD, mixed, video, router_fix)
- Paper : `paper/dynamoscope.tex` §10 (7 subsections, ~400 lines LaTeX)
