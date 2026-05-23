"""Experiment : distill classical Takens trajectory dans neural mini-RSSM.

Hypothèse : ClassicalAlignmentLoss permet à un neural encoder d'acquérir
dynamics-richness du classical (better DNA topology/spectral/structure)
sans perdre semantic invariance (recon loss préservée).

Setup :
- Train sets : bouncing_balls, rotating_shapes, color_morph (procedural)
- Held-out : double_pendulum_render (chaotic régime jamais vu pendant training)
- 3 conditions :
  - baseline       : recon + dynamics only (no Phase C, no alignment)
  - phase_c        : recon + dynamics + Phase C losses (SFA + causal + lyap)
  - distill        : recon + dynamics + classical alignment (motion delay, m=3)
- 5 seeds chaque, 15 epochs
- Eval : DNA composite sur held-out

Question : distill batte-t-il baseline et/ou phase_c sur DNA composite ?
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.datasets import (
    gen_bouncing_balls,
    gen_rotating_shapes,
    gen_color_morph,
    gen_double_pendulum_render,
)
from backend.training import MiniRSSM, TrainConfig
from backend.training.trainer import video_to_tensor
from backend.losses import (
    SFASlownessRegularizer,
    CausalSparsityLoss,
    LyapunovMatchingLoss,
    ClassicalAlignmentLoss,
    compute_classical_target,
)
from backend.reducer import reduce_3d
from backend.dna import compute_dna


N_SEEDS = 5
N_EPOCHS = 15
TARGET_SIZE = 64


def make_clips_with_classical_targets(seed: int) -> tuple[list, list]:
    """Génère train clips + leurs Takens classical targets (motion, m=3)."""
    np_clips = []
    np_clips.append(np.stack(gen_bouncing_balls(n_frames=32, n_balls=3, seed=seed)).astype(np.uint8))
    np_clips.append(np.stack(gen_rotating_shapes(n_frames=32)).astype(np.uint8))
    np_clips.append(np.stack(gen_color_morph(n_frames=32)).astype(np.uint8))

    tensor_clips = [video_to_tensor(c, TARGET_SIZE) for c in np_clips]
    targets = [compute_classical_target(c, observable="motion", m=3) for c in np_clips]
    return tensor_clips, targets


def make_held_out() -> np.ndarray:
    import cv2
    frames = gen_double_pendulum_render(n_frames=60)
    arr = np.stack(frames).astype(np.float32) / 255.0
    return np.stack([cv2.resize(f, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for f in arr])


def train_one(cfg_name: str, seed: int) -> dict:
    """Train mini-RSSM avec config. Returns metrics on held-out."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_clips, classical_targets = make_clips_with_classical_targets(seed)
    held_out = make_held_out()

    cfg = TrainConfig(
        n_epochs=N_EPOCHS,
        embed_dim=64,
        hidden_dim=128,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MiniRSSM(in_channels=3, embed_dim=64, hidden_dim=128).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Loss modules
    sfa = SFASlownessRegularizer(weight=0.5)
    causal = CausalSparsityLoss(weight=0.3)
    lyap = LyapunovMatchingLoss(target_lyapunov=0.0, weight=0.1)
    align = ClassicalAlignmentLoss(mode="distance_corr", weight=1.0)

    targets_dev = [t.to(device) for t in classical_targets]

    for epoch in range(N_EPOCHS):
        for clip, target in zip(train_clips, targets_dev):
            frames = clip.to(device)
            z, z_pred, recon = model(frames)
            l_recon = ((recon - frames) ** 2).mean()
            l_dyn = ((z_pred[:-1] - z[1:].detach()) ** 2).mean() if z.shape[0] >= 2 else torch.tensor(0.0, device=device)

            total = l_recon + l_dyn
            if cfg_name == "phase_c":
                total = total + sfa(z) + causal(z) + lyap(z)
            elif cfg_name == "distill":
                total = total + align(z, target)
            # baseline = recon + dyn only

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    # Eval on held-out
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(held_out).permute(0, 3, 1, 2).float().to(device)
        z = model.encoder(t).cpu().numpy()

    coords = reduce_3d(z, method="pca")
    dna = compute_dna(z, coords)
    return {
        "dna_score": dna["composite_score"],
        "dna_axes": dna["axes"],
        "dna_label": dna["label"],
    }


def main():
    print("=" * 75)
    print(f"DISTILLATION EXPERIMENT — {N_SEEDS} seeds × 3 configs × held-out double_pendulum")
    print("=" * 75)

    configs = ["baseline", "phase_c", "distill"]
    results = {c: [] for c in configs}
    t0 = time.time()
    n_done = 0
    n_total = N_SEEDS * 3
    for seed in range(N_SEEDS):
        for cfg in configs:
            n_done += 1
            ts = time.time()
            r = train_one(cfg, seed)
            elapsed = time.time() - ts
            results[cfg].append(r)
            print(f"  [{n_done}/{n_total}] seed={seed} cfg={cfg:<10} DNA={r['dna_score']:.2f}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"\n[time] total {total_elapsed:.1f}s")

    # Aggregate
    print("\n" + "=" * 75)
    print("AGGREGATE (mean ± std)")
    print("=" * 75)
    print(f"{'config':<12}{'DNA mean':>12}{'DNA std':>12}{'DNA range':>20}")
    print("-" * 60)
    summary = {}
    for cfg in configs:
        scores = [r["dna_score"] for r in results[cfg]]
        s_mean = float(np.mean(scores))
        s_std = float(np.std(scores))
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        summary[cfg] = {
            "mean": s_mean, "std": s_std, "min": s_min, "max": s_max,
            "n": len(scores), "scores": scores,
        }
        print(f"{cfg:<12}{s_mean:>12.2f}{s_std:>12.2f}{f'[{s_min:.2f}, {s_max:.2f}]':>20}")

    # Welch t-tests
    from math import erf, sqrt
    def welch(a, b):
        a, b = np.array(a), np.array(b)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = np.sqrt(va / na + vb / nb)
        if se < 1e-12: return float("nan"), 1.0
        t = (ma - mb) / se
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)

    print("\n" + "=" * 75)
    print("WELCH'S T-TEST (vs baseline)")
    print("=" * 75)
    base_scores = [r["dna_score"] for r in results["baseline"]]
    tests = {}
    for cfg in ["phase_c", "distill"]:
        cfg_scores = [r["dna_score"] for r in results[cfg]]
        t, p = welch(cfg_scores, base_scores)
        sig = "yes (*)" if p < 0.05 else "no"
        print(f"  {cfg:<12}vs baseline:  t={t:>+8.3f}  p={p:>.4f}  {sig}")
        tests[f"{cfg}_vs_baseline"] = {"t": t, "p": p, "significant": p < 0.05}

    # Per-axis decomposition for distill vs baseline
    print("\n" + "=" * 75)
    print("PER-AXIS DELTA (distill mean − baseline mean)")
    print("=" * 75)
    axes_keys = list(results["baseline"][0]["dna_axes"].keys())
    print(f"{'axis':<22}{'baseline':>12}{'distill':>12}{'delta':>14}")
    print("-" * 60)
    per_axis = {}
    for ax in axes_keys:
        b_vals = [r["dna_axes"][ax] for r in results["baseline"]]
        d_vals = [r["dna_axes"][ax] for r in results["distill"]]
        b_mean = float(np.mean(b_vals))
        d_mean = float(np.mean(d_vals))
        delta = d_mean - b_mean
        per_axis[ax] = {"baseline_mean": b_mean, "distill_mean": d_mean, "delta": delta}
        print(f"{ax:<22}{b_mean:>12.4f}{d_mean:>12.4f}{delta:>+14.4f}")

    # Save artifact
    Path("results").mkdir(exist_ok=True)
    out = {
        "summary": summary,
        "tests": tests,
        "per_axis": per_axis,
        "raw": results,
        "n_seeds": N_SEEDS,
        "n_epochs": N_EPOCHS,
    }
    def _default(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"non-serializable {type(o)}")
    json.dump(out, open("results/classical_distillation.json", "w"), indent=2, default=_default)
    print(f"\n[saved] results/classical_distillation.json")


if __name__ == "__main__":
    main()
