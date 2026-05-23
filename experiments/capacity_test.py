"""Capacity hypothesis test — disentangle representational vs capacity cause.

§7.7v channel_split (embed_dim=64, split 32/32) underperformed phase_c.
Per-axis : structure −0.34 (huge drop). Hypothesis (c) capacity :
phase_c needed full 64 dims; halving to 32 cost expressivity.

This experiment :
- Train at embed_dim=128 (split 64/64 OR full 128).
- channel_split_big now has 64 dims per loss (= phase_c original capacity)
- If channel_split_big beats phase_c_64 OR ties phase_c_128 → capacity was
  binding. (c) confirmed.
- If still underperforms → (a) representational is dominant cause.

Configs (same 5 seeds, 15 epochs, held-out double_pendulum) :
- baseline_64           : recon + dyn only, embed_dim=64
- phase_c_64            : full Phase C losses on embed_dim=64
- phase_c_128           : full Phase C losses on embed_dim=128
- channel_split_128     : 128 dims, split 64/64
- phase_c_plus_distill_128: full Phase C + distill on 128 dim z
"""
from __future__ import annotations
import json
import sys
import time
from math import erf, sqrt
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.datasets import (
    gen_bouncing_balls, gen_rotating_shapes, gen_color_morph, gen_double_pendulum_render,
)
from backend.training import MiniRSSM
from backend.training.trainer import video_to_tensor
from backend.losses import (
    SFASlownessRegularizer, CausalSparsityLoss, LyapunovMatchingLoss,
    ClassicalAlignmentLoss, compute_classical_target,
)
from backend.reducer import reduce_3d
from backend.dna import compute_dna


N_SEEDS = 5
N_EPOCHS = 15
TARGET_SIZE = 64


CONFIGS = {
    # name : (embed_dim, hidden_dim, mode)
    "baseline_64":            (64,  128, "baseline"),
    "phase_c_64":             (64,  128, "phase_c"),
    "phase_c_128":            (128, 256, "phase_c"),
    "channel_split_128":      (128, 256, "channel_split"),
    "phase_c_plus_distill_128": (128, 256, "phase_c_plus_distill"),
}


def make_clips_targets(seed):
    np_clips = [
        np.stack(gen_bouncing_balls(n_frames=32, n_balls=3, seed=seed)).astype(np.uint8),
        np.stack(gen_rotating_shapes(n_frames=32)).astype(np.uint8),
        np.stack(gen_color_morph(n_frames=32)).astype(np.uint8),
    ]
    clips = [video_to_tensor(c, TARGET_SIZE) for c in np_clips]
    targets = [compute_classical_target(c, "motion", m=3) for c in np_clips]
    return clips, targets


def make_held_out():
    import cv2
    frames = gen_double_pendulum_render(n_frames=60)
    arr = np.stack(frames).astype(np.float32) / 255.0
    return np.stack([cv2.resize(f, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for f in arr])


def train_one(cfg_name: str, seed: int) -> dict:
    embed_dim, hidden_dim, mode = CONFIGS[cfg_name]
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MiniRSSM(in_channels=3, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=3e-4)

    clips, targets = make_clips_targets(seed)
    targets_dev = [t.to(device) for t in targets]
    held = make_held_out()

    sfa = SFASlownessRegularizer(weight=0.5)
    causal = CausalSparsityLoss(weight=0.3)
    lyap = LyapunovMatchingLoss(target_lyapunov=0.0, weight=0.1)
    align = ClassicalAlignmentLoss(mode="distance_corr", weight=1.0)

    split = embed_dim // 2  # for channel_split mode

    for epoch in range(N_EPOCHS):
        for clip, target in zip(clips, targets_dev):
            frames = clip.to(device)
            z, z_pred, recon = model(frames)
            l_recon = ((recon - frames) ** 2).mean()
            l_dyn = (
                ((z_pred[:-1] - z[1:].detach()) ** 2).mean()
                if z.shape[0] >= 2 else torch.tensor(0.0, device=device)
            )
            total = l_recon + l_dyn

            if mode == "baseline":
                pass
            elif mode == "phase_c":
                total = total + sfa(z) + causal(z) + lyap(z)
            elif mode == "channel_split":
                z_stat = z[:, :split]
                z_geo = z[:, split:]
                total = total + sfa(z_stat) + causal(z_stat) + lyap(z_stat)
                total = total + align(z_geo, target)
            elif mode == "phase_c_plus_distill":
                total = total + sfa(z) + causal(z) + lyap(z) + align(z, target)

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(held).permute(0, 3, 1, 2).float().to(device)
        z = model.encoder(t).cpu().numpy()
    coords = reduce_3d(z, method="pca")
    dna = compute_dna(z, coords)
    return {
        "dna_score": dna["composite_score"],
        "dna_axes": dna["axes"],
        "embed_dim": embed_dim,
        "params": sum(p.numel() for p in params),
    }


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


def main():
    print("=" * 75)
    print(f"CAPACITY HYPOTHESIS TEST — {N_SEEDS} seeds × {len(CONFIGS)} configs × {N_EPOCHS} ep")
    print("=" * 75)

    results = {c: [] for c in CONFIGS}
    t0 = time.time()
    n_done = 0
    n_total = N_SEEDS * len(CONFIGS)
    for seed in range(N_SEEDS):
        for cfg in CONFIGS:
            n_done += 1
            ts = time.time()
            r = train_one(cfg, seed)
            elapsed = time.time() - ts
            results[cfg].append(r)
            print(f"  [{n_done}/{n_total}] seed={seed} cfg={cfg:<30} DNA={r['dna_score']:.2f}  ({elapsed:.1f}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    print("\n" + "=" * 75)
    print("AGGREGATE")
    print("=" * 75)
    summary = {}
    print(f"  {'config':<30}{'DNA':>10}{'std':>8}{'params':>12}")
    for cfg in CONFIGS:
        scores = [r["dna_score"] for r in results[cfg]]
        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))
        n_params = results[cfg][0]["params"]
        summary[cfg] = {"mean": s_mean, "std": s_std, "scores": scores, "n_params": n_params}
        print(f"  {cfg:<30}{s_mean:>10.2f}{s_std:>8.2f}{n_params:>12,}")

    print("\n" + "=" * 75)
    print("WELCH'S T-TEST")
    print("=" * 75)
    cs_128 = [r["dna_score"] for r in results["channel_split_128"]]
    pc_64 = [r["dna_score"] for r in results["phase_c_64"]]
    pc_128 = [r["dna_score"] for r in results["phase_c_128"]]
    pcd_128 = [r["dna_score"] for r in results["phase_c_plus_distill_128"]]
    tests = {}
    for label, (a, b) in [
        ("channel_split_128 vs phase_c_64",        (cs_128, pc_64)),
        ("channel_split_128 vs phase_c_128",       (cs_128, pc_128)),
        ("phase_c_128 vs phase_c_64",              (pc_128, pc_64)),
        ("phase_c_plus_distill_128 vs phase_c_128", (pcd_128, pc_128)),
    ]:
        t, p = welch(a, b)
        sig = "yes (*)" if p < 0.05 else "no"
        print(f"  {label:<46}t={t:>+6.2f}  p={p:.4f}  {sig}")
        tests[label] = {"t": t, "p": p}

    Path("results").mkdir(exist_ok=True)
    out = {
        "summary": summary,
        "tests": tests,
        "raw": results,
        "configs": {k: {"embed_dim": v[0], "hidden_dim": v[1], "mode": v[2]} for k, v in CONFIGS.items()},
        "n_seeds": N_SEEDS,
        "n_epochs": N_EPOCHS,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/capacity_test.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/capacity_test.json")

    # Final interpretation
    print("\n" + "=" * 75)
    print("INTERPRETATION")
    print("=" * 75)
    cs_mean = float(np.mean(cs_128))
    pc64_mean = float(np.mean(pc_64))
    pc128_mean = float(np.mean(pc_128))
    if cs_mean > pc64_mean and cs_mean > pc128_mean:
        print("  → channel_split_128 beats BOTH phase_c variants")
        print("  → CAPACITY hypothesis (c) confirmed : doubling dims resolves conflict")
    elif cs_mean > pc64_mean and cs_mean < pc128_mean:
        print("  → channel_split_128 only beats phase_c_64 (smaller model)")
        print("  → Mixed : capacity helps but phase_c at same total size still wins")
    else:
        print("  → channel_split_128 does NOT beat phase_c (any size)")
        print("  → REPRESENTATIONAL hypothesis (a) dominates : conflict is not capacity-limited")


if __name__ == "__main__":
    main()
