"""PCGrad surgery test : résout-il conflit phase_c + distill ?

§7.7 finding : phase_c + distill (45.15) < phase_c seul (47.84).
Hypothèse : gradients conflictent (tug-of-war). PCGrad project orthogonal.
Test : entraîne avec PCGrad. Si DNA > phase_c seul, surgery résout.
Sinon : conflit fondamental représentationnel, pas optimisation.

Setup identique à classical_distillation.py : 5 seeds, 15 epochs,
held-out double_pendulum_render.

Configs comparés :
- baseline                : recon + dyn only
- phase_c                 : full Phase C losses
- phase_c_plus_distill    : sum naive (§7.7)
- phase_c_plus_distill_pc : combined via PCGrad
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
from backend.training import MiniRSSM, pcgrad_step
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


def make_clips_targets(seed: int):
    clips = []
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
    model = MiniRSSM(in_channels=3, embed_dim=64, hidden_dim=128).to(device)
    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=3e-4)

    clips, targets = make_clips_targets(seed)
    targets_dev = [t.to(device) for t in targets]
    held = make_held_out()

    sfa = SFASlownessRegularizer(weight=0.5)
    causal = CausalSparsityLoss(weight=0.3)
    lyap = LyapunovMatchingLoss(target_lyapunov=0.0, weight=0.1)
    align = ClassicalAlignmentLoss(mode="distance_corr", weight=1.0)

    total_conflicts = 0
    n_steps = 0

    for epoch in range(N_EPOCHS):
        for clip, target in zip(clips, targets_dev):
            frames = clip.to(device)
            z, z_pred, recon = model(frames)
            l_recon = ((recon - frames) ** 2).mean()
            l_dyn = (
                ((z_pred[:-1] - z[1:].detach()) ** 2).mean()
                if z.shape[0] >= 2 else torch.tensor(0.0, device=device)
            )

            if cfg == "baseline":
                total = l_recon + l_dyn
                optim.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()

            elif cfg == "phase_c":
                total = l_recon + l_dyn + sfa(z) + causal(z) + lyap(z)
                optim.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()

            elif cfg == "phase_c_plus_distill_sum":
                total = l_recon + l_dyn + sfa(z) + causal(z) + lyap(z) + align(z, target)
                optim.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()

            elif cfg == "phase_c_plus_distill_pcgrad":
                # 2 tasks : task_phase = recon + dyn + sfa + causal + lyap, task_distill = align
                # PCGrad projects task_distill gradient orthogonal to task_phase si conflict
                task_phase = l_recon + l_dyn + sfa(z) + causal(z) + lyap(z)
                task_distill = align(z, target)
                stats = pcgrad_step([task_phase, task_distill], params, optim, clip_norm=1.0)
                total_conflicts += stats["n_conflicts"]
                n_steps += 1

    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(held).permute(0, 3, 1, 2).float().to(device)
        z = model.encoder(t).cpu().numpy()
    coords = reduce_3d(z, method="pca")
    dna = compute_dna(z, coords)
    return {
        "dna_score": dna["composite_score"],
        "dna_axes": dna["axes"],
        "n_conflicts": total_conflicts,
        "n_steps": n_steps,
    }


def main():
    print("=" * 70)
    print(f"PCGRAD SURGERY TEST — {N_SEEDS} seeds × 4 configs × {N_EPOCHS} epochs")
    print("=" * 70)
    configs = [
        "baseline",
        "phase_c",
        "phase_c_plus_distill_sum",
        "phase_c_plus_distill_pcgrad",
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
            extra = ""
            if "pcgrad" in cfg and r["n_steps"] > 0:
                extra = f" conflicts={r['n_conflicts']}/{r['n_steps']*1} = {r['n_conflicts']/max(r['n_steps'],1):.2f}/step"
            print(f"  [{n_done}/{n_total}] seed={seed} cfg={cfg:<32} DNA={r['dna_score']:.2f}{extra}  ({elapsed:.1f}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE (DNA mean ± std)")
    print("=" * 70)
    summary = {}
    for cfg in configs:
        scores = [r["dna_score"] for r in results[cfg]]
        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))
        summary[cfg] = {"mean": s_mean, "std": s_std, "scores": scores, "n": len(scores)}
        print(f"  {cfg:<32}{s_mean:>10.2f} ± {s_std:.2f}")

    # Welch test
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

    print("\n" + "=" * 70)
    print("WELCH'S T-TEST")
    print("=" * 70)
    base_scores = [r["dna_score"] for r in results["baseline"]]
    pc_scores = [r["dna_score"] for r in results["phase_c"]]
    tests = {}
    print(f"\n  vs baseline:")
    for cfg in configs:
        if cfg == "baseline": continue
        s = [r["dna_score"] for r in results[cfg]]
        t, p = welch(s, base_scores)
        sig = "yes (*)" if p < 0.05 else "no"
        print(f"    {cfg:<32}t={t:>+6.2f}  p={p:.4f}  {sig}")
        tests[f"{cfg}_vs_baseline"] = {"t": t, "p": p}

    print(f"\n  vs phase_c:")
    for cfg in ("phase_c_plus_distill_sum", "phase_c_plus_distill_pcgrad"):
        s = [r["dna_score"] for r in results[cfg]]
        t, p = welch(s, pc_scores)
        sig = "yes (*)" if p < 0.05 else "no"
        print(f"    {cfg:<32}t={t:>+6.2f}  p={p:.4f}  {sig}")
        tests[f"{cfg}_vs_phase_c"] = {"t": t, "p": p}

    # Conflict rate analysis for PCGrad config
    pc_runs = results["phase_c_plus_distill_pcgrad"]
    avg_conflict_rate = float(np.mean([
        r["n_conflicts"] / max(r["n_steps"], 1) for r in pc_runs
    ]))
    print(f"\n  PCGrad mean conflict rate : {avg_conflict_rate:.3f} per step")
    print(f"    (0 = never conflicts, 2 = both tasks conflict each step)")

    Path("results").mkdir(exist_ok=True)
    out = {
        "summary": summary,
        "tests": tests,
        "pcgrad_conflict_rate": avg_conflict_rate,
        "raw": results,
        "n_seeds": N_SEEDS,
        "n_epochs": N_EPOCHS,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/pcgrad_distill.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/pcgrad_distill.json")


if __name__ == "__main__":
    main()
