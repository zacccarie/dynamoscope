"""Channel separation test — résoudre conflit Phase C / Distill via décomposition latente.

§7.7 + PCGrad ont établi : Phase C et Distill sur MEME latent z conflictent
fondamentalement (représentationnel, pas optimization). Hypothèse Part 4
framework : decouple en assignant chaque loss à un sub-channel disjoint.

Architecture :
    z ∈ R^64 → z = (z_stat ∈ R^32, z_geo ∈ R^32)
    - z_stat reçoit Phase C losses (SFA, Causal, Lyap)
    - z_geo reçoit ClassicalAlignmentLoss
    - Reconstruction et dynamics losses sur z entier (cohérence)

Si DNA(channel_split) > DNA(phase_c) :
  → architecturalement, decoupling résout. Valide framework Part 4.
Sinon :
  → conflit ne réside pas dans partage de dimensions mais dans objectif global.

5 seeds × 4 configs × 15 epochs. Held-out double_pendulum_render.
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
    gen_bouncing_balls,
    gen_rotating_shapes,
    gen_color_morph,
    gen_double_pendulum_render,
)
from backend.training import MiniRSSM
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
EMBED_DIM = 64
SPLIT = 32  # z_stat = z[:, :SPLIT], z_geo = z[:, SPLIT:]


def make_clips_targets(seed: int):
    np_clips = []
    np_clips.append(np.stack(gen_bouncing_balls(n_frames=32, n_balls=3, seed=seed)).astype(np.uint8))
    np_clips.append(np.stack(gen_rotating_shapes(n_frames=32)).astype(np.uint8))
    np_clips.append(np.stack(gen_color_morph(n_frames=32)).astype(np.uint8))
    clips = [video_to_tensor(c, TARGET_SIZE) for c in np_clips]
    targets = [compute_classical_target(c, "motion", m=3) for c in np_clips]
    return clips, targets


def make_held_out():
    import cv2
    frames = gen_double_pendulum_render(n_frames=60)
    arr = np.stack(frames).astype(np.float32) / 255.0
    return np.stack([cv2.resize(f, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for f in arr])


def train_one(cfg: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MiniRSSM(in_channels=3, embed_dim=EMBED_DIM, hidden_dim=128).to(device)
    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=3e-4)

    clips, targets = make_clips_targets(seed)
    targets_dev = [t.to(device) for t in targets]
    held = make_held_out()

    sfa = SFASlownessRegularizer(weight=0.5)
    causal = CausalSparsityLoss(weight=0.3)
    lyap = LyapunovMatchingLoss(target_lyapunov=0.0, weight=0.1)
    align = ClassicalAlignmentLoss(mode="distance_corr", weight=1.0)

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

            if cfg == "baseline":
                pass

            elif cfg == "phase_c_full_z":
                total = total + sfa(z) + causal(z) + lyap(z)

            elif cfg == "distill_full_z":
                total = total + align(z, target)

            elif cfg == "channel_split":
                # z_stat = z[:, :SPLIT], z_geo = z[:, SPLIT:]
                z_stat = z[:, :SPLIT]
                z_geo = z[:, SPLIT:]
                total = total + sfa(z_stat) + causal(z_stat) + lyap(z_stat)
                total = total + align(z_geo, target)

            elif cfg == "phase_c_plus_distill_z":
                # control : both on full z, no channel split (= §7.7)
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
    print(f"CHANNEL SEPARATION TEST — {N_SEEDS} seeds × 5 configs × {N_EPOCHS} ep")
    print(f"  embed_dim={EMBED_DIM}, split at {SPLIT}: z_stat[:{SPLIT}] / z_geo[{SPLIT}:]")
    print("=" * 75)

    configs = [
        "baseline",
        "phase_c_full_z",
        "distill_full_z",
        "phase_c_plus_distill_z",  # control = §7.7 negative
        "channel_split",            # hypothesis
    ]
    results = {c: [] for c in configs}
    t0 = time.time()
    n_done = 0
    n_total = N_SEEDS * len(configs)
    for seed in range(N_SEEDS):
        for cfg in configs:
            n_done += 1
            ts = time.time()
            r = train_one(cfg, seed)
            elapsed = time.time() - ts
            results[cfg].append(r)
            print(f"  [{n_done}/{n_total}] seed={seed} cfg={cfg:<26} DNA={r['dna_score']:.2f}  ({elapsed:.1f}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    print("\n" + "=" * 75)
    print("AGGREGATE")
    print("=" * 75)
    summary = {}
    for cfg in configs:
        scores = [r["dna_score"] for r in results[cfg]]
        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))
        summary[cfg] = {"mean": s_mean, "std": s_std, "scores": scores, "n": len(scores)}
        print(f"  {cfg:<26}{s_mean:>10.2f} ± {s_std:.2f}")

    print("\n" + "=" * 75)
    print("WELCH'S T-TEST")
    print("=" * 75)
    base = [r["dna_score"] for r in results["baseline"]]
    pc = [r["dna_score"] for r in results["phase_c_full_z"]]
    sum_ = [r["dna_score"] for r in results["phase_c_plus_distill_z"]]
    cs = [r["dna_score"] for r in results["channel_split"]]
    tests = {}
    for label, (a, b) in [
        ("channel_split vs baseline",        (cs, base)),
        ("channel_split vs phase_c",         (cs, pc)),
        ("channel_split vs phase_c+distill_z", (cs, sum_)),
    ]:
        t, p = welch(a, b)
        sig = "yes (*)" if p < 0.05 else "no"
        print(f"  {label:<38}t={t:>+6.2f}  p={p:.4f}  {sig}")
        tests[label] = {"t": t, "p": p}

    # Per-axis comparison channel_split vs phase_c
    print("\n" + "=" * 75)
    print("PER-AXIS (channel_split mean − phase_c mean)")
    print("=" * 75)
    axes_keys = list(results["baseline"][0]["dna_axes"].keys())
    per_axis = {}
    print(f"{'axis':<22}{'phase_c':>12}{'cs':>12}{'delta':>14}")
    for ax in axes_keys:
        pc_vals = [r["dna_axes"][ax] for r in results["phase_c_full_z"]]
        cs_vals = [r["dna_axes"][ax] for r in results["channel_split"]]
        pc_m = float(np.mean(pc_vals))
        cs_m = float(np.mean(cs_vals))
        delta = cs_m - pc_m
        per_axis[ax] = {"phase_c": pc_m, "channel_split": cs_m, "delta": delta}
        print(f"  {ax:<22}{pc_m:>10.4f}{cs_m:>12.4f}{delta:>+14.4f}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "summary": summary,
        "tests": tests,
        "per_axis": per_axis,
        "raw": results,
        "split_dim": SPLIT,
        "embed_dim": EMBED_DIM,
        "n_seeds": N_SEEDS,
        "n_epochs": N_EPOCHS,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/channel_split.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/channel_split.json")


if __name__ == "__main__":
    main()
