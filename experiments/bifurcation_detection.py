"""Bifurcation detection — Van der Pol Hopf at μ=0.

Van der Pol :
    dx/dt = y
    dy/dt = μ(1 − x²) y − x

Hopf bifurcation at μ=0 :
  - μ < 0 : origin stable, trajectory spirals in → fixed point
  - μ > 0 : origin unstable, limit cycle emerges (radius ∝ √μ near μ=0)

Test : pour chaque μ ∈ {-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1.0} :
1. Generate VdP trajectory as video (render position on canvas)
2. Encode via 3 pipelines :
   - delay_motion_auto_m3 (classical)
   - mini-RSSM phase_c_64 trained on μ=0.5 only (neural)
   - delay_brightness_auto_m3 (alt classical)
3. Compute regime classifier verdict + DNA
4. Plot : verdict_kind, confidence, DNA axes vs μ
5. Bifurcation detection : changepoint au plus près de μ=0 ?

Held-out test : framework détecte transition même sur encoder neural
entraîné sur un seul régime ?
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

from backend.datasets import _write_video_from_frames
from backend.models.registry import get_model
from backend.reducer import reduce_3d
from backend.dna import compute_dna
from backend.dynamics import analyse_trajectory
from backend.regime_classifier import classify_from_analysis


def vdp_trajectory(mu: float, n_steps: int = 800, dt: float = 0.05, sub: int = 2) -> np.ndarray:
    """RK4 Van der Pol. Returns (n_steps, 2) (x, y) trajectory."""
    def f(s):
        x, y = s
        return np.array([y, mu * (1 - x * x) * y - x])

    def rk4(s, dt):
        k1 = f(s)
        k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2)
        k4 = f(s + dt * k3)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    s = np.array([0.5, 0.0])
    # transient
    for _ in range(400):
        s = rk4(s, dt)
    traj = []
    for _ in range(n_steps):
        for _ in range(sub):
            s = rk4(s, dt)
        traj.append(s.copy())
    return np.array(traj)


def render_vdp_video(mu: float, n_frames: int = 80, size: int = 64) -> np.ndarray:
    """Render VdP trajectory as video. Each frame shows position of ball
    plus recent trail. Tracks current attractor visually.

    Returns: (n_frames, H, W, 3) uint8 array.
    """
    traj = vdp_trajectory(mu, n_steps=n_frames, dt=0.04, sub=4)
    # Fit traj into [size//4, 3*size//4] bbox
    mx = max(2.5, float(np.abs(traj).max() * 1.1))
    cx, cy = size // 2, size // 2
    scale = (size * 0.4) / mx

    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    trail_len = 12
    for i in range(n_frames):
        f = np.zeros((size, size, 3), dtype=np.uint8)
        # background grid
        f[size // 2, :, 0] = 40
        f[:, size // 2, 1] = 40
        # trail (fading)
        start = max(0, i - trail_len)
        for k, j in enumerate(range(start, i + 1)):
            x_p = int(cx + traj[j, 0] * scale)
            y_p = int(cy + traj[j, 1] * scale)
            if 0 <= x_p < size and 0 <= y_p < size:
                alpha = (k + 1) / (trail_len + 1)
                f[y_p, x_p, 0] = max(f[y_p, x_p, 0], int(200 * alpha))
                f[y_p, x_p, 1] = max(f[y_p, x_p, 1], int(220 * alpha))
                f[y_p, x_p, 2] = max(f[y_p, x_p, 2], int(255 * alpha))
        # current head (3×3 brighter)
        x_p = int(cx + traj[i, 0] * scale)
        y_p = int(cy + traj[i, 1] * scale)
        if 0 <= x_p < size and 0 <= y_p < size:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x_p + dx, y_p + dy
                    if 0 <= xx < size and 0 <= yy < size:
                        f[yy, xx, :] = 255
        frames[i] = f
    return frames


def eval_at_mu(mu: float, models: list[str], n_frames: int) -> dict:
    """For each model, encode VdP(μ) frames + compute metrics."""
    frames = render_vdp_video(mu, n_frames=n_frames, size=64).astype(np.float32) / 255.0
    out = {"mu": float(mu)}

    # Ground truth from raw 2D trajectory (no encoding)
    raw_traj = vdp_trajectory(mu, n_steps=n_frames, dt=0.04, sub=4)
    # Stretch raw to 3D for compatibility
    raw_3d = np.concatenate([raw_traj, np.zeros((n_frames, 1))], axis=1)
    raw_analysis = analyse_trajectory(raw_3d)
    raw_verdict = classify_from_analysis(raw_analysis, embedding_dim=3)
    out["raw"] = {
        "verdict": raw_verdict.to_dict(),
        "lyapunov": raw_analysis["lyapunov"],
        "correlation_dim": raw_analysis["correlation_dim"],
        "max_diag_ratio": raw_analysis["max_diag_ratio"],
        "convergence_rate": raw_analysis["convergence_rate"],
    }

    for model_name in models:
        try:
            model = get_model(model_name)
            latents = model.produce_trajectory(frames)
            coords = reduce_3d(latents, method="pca")
            analysis = analyse_trajectory(coords)
            verdict = classify_from_analysis(analysis, embedding_dim=3)
            dna = compute_dna(latents, coords)
            out[model_name] = {
                "verdict": verdict.to_dict(),
                "dna_score": dna["composite_score"],
                "lyapunov": analysis["lyapunov"],
                "max_diag_ratio": analysis["max_diag_ratio"],
                "convergence_rate": analysis["convergence_rate"],
                "correlation_dim": analysis["correlation_dim"],
            }
        except Exception as e:
            out[model_name] = {"error": str(e)}
    return out


def detect_bifurcation(curve: list[float], mu_values: list[float]) -> dict:
    """Find argmax |Δ| (largest change) along curve sorted by μ.
    Returns predicted μ* + raw differences."""
    arr = np.array(curve)
    mus = np.array(mu_values)
    diffs = np.abs(np.diff(arr))
    if len(diffs) == 0:
        return {"detected_mu": None, "diffs": []}
    idx = int(np.argmax(diffs))
    mu_predicted = (mus[idx] + mus[idx + 1]) / 2
    return {
        "detected_mu": float(mu_predicted),
        "max_diff_idx": idx,
        "max_diff": float(diffs[idx]),
        "diffs": diffs.tolist(),
    }


def main():
    print("=" * 75)
    print("BIFURCATION DETECTION — Van der Pol Hopf (μ=0)")
    print("=" * 75)

    mu_values = [-0.5, -0.3, -0.1, -0.02, 0.0, 0.02, 0.1, 0.3, 0.5, 1.0]
    # Note : dinov2 needs 224×224 input (patch 14×14), incompatible with our
    # 64×64 render — would require separate larger video. Skip for now.
    models = [
        "delay_motion_auto_m3",
        "delay_brightness_auto_m3",
        "delay_entropy_auto_m3",
        "pca_obs_m3",
        "direct_bme",
    ]
    n_frames = 80

    rows = []
    t0 = time.time()
    for mu in mu_values:
        print(f"\n[mu={mu:+.2f}] generating + encoding...")
        ts = time.time()
        r = eval_at_mu(mu, models, n_frames)
        print(f"  raw verdict: {r['raw']['verdict']['kind']:<10}  "
              f"λ={r['raw']['lyapunov']:>+.3f}  D={r['raw']['correlation_dim']:.2f}  "
              f"conv={r['raw']['convergence_rate']:>+.3f}")
        for m in models:
            if "error" in r[m]:
                print(f"  {m}: ERROR {r[m]['error']}")
                continue
            v = r[m]['verdict']
            print(f"  {m:<28}: {v['kind']:<10}  DNA={r[m]['dna_score']:>6.2f}  "
                  f"λ={r[m]['lyapunov']:>+.3f}  conv={r[m]['convergence_rate']:>+.3f}")
        rows.append(r)
        print(f"  ({time.time() - ts:.1f}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Summary table : verdict kind per (model, mu)
    print("\n" + "=" * 75)
    print("VERDICT KIND PER (μ, MODEL)")
    print("=" * 75)
    header = f"{'μ':<8}{'raw':<14}" + "".join(f"{m[:14]:<16}" for m in models)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['mu']:+.2f}    {r['raw']['verdict']['kind']:<14}"
        for m in models:
            line += f"{r[m].get('verdict', {}).get('kind', 'err'):<16}"
        print(line)

    # Bifurcation detection per model : where does verdict_kind / DNA change most
    print("\n" + "=" * 75)
    print("DETECTED BIFURCATION μ* (max-Δ of convergence_rate per model)")
    print("=" * 75)
    print("Ground truth : μ* = 0")
    detections = {}
    for m in ["raw"] + models:
        conv_curve = [r[m].get("convergence_rate", 0.0) for r in rows]
        det = detect_bifurcation(conv_curve, mu_values)
        detections[m] = det
        err = abs(det["detected_mu"] - 0.0) if det["detected_mu"] is not None else float("inf")
        print(f"  {m:<28}μ_detected = {det['detected_mu']:>+.3f}  (|err| = {err:.3f})")

    # Save
    Path("results").mkdir(exist_ok=True)
    out = {
        "mu_values": mu_values,
        "rows": rows,
        "detections": detections,
        "ground_truth_mu_star": 0.0,
        "models": models,
        "n_frames": n_frames,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/bifurcation_detection.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/bifurcation_detection.json")


if __name__ == "__main__":
    main()
