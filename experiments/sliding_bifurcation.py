"""Sliding-window bifurcation detection — real-time variant.

Différence avec §8 : pas de clips pré-coupés par μ. Une SEULE vidéo
continue où μ ramps smoothly de -0.5 à +1.0 sur N frames. Test if
sliding-window analysis detects Hopf transition WHILE it happens.

Pipeline :
1. Generate VdP avec μ(t) = lerp(-0.5, +1.0, t/N), render 1 video
2. Sliding window width W (e.g. 30 frames), stride S (e.g. 5)
3. For each window : encode → analyse_trajectory → regime verdict
4. Plot timeline : convergence_rate, regime_kind, DNA per window
5. Detect transition frame via change-point algorithm :
   - CUSUM on convergence_rate
   - Argmax discrete-diff
6. Compare to ground-truth transition frame (μ(t)=0)

Lead-time : positive si détection happens BEFORE actual transition,
negative si après. Real-time systems veulent positive lead-time.
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

from backend.models.registry import get_model
from backend.reducer import reduce_3d
from backend.dynamics import analyse_trajectory
from backend.regime_classifier import classify_from_analysis


N_FRAMES = 400
WINDOW = 40
STRIDE = 5
MU_START = -0.5
MU_END = 1.5
SIZE = 64


def vdp_ramp_trajectory(n_frames: int, mu_start: float, mu_end: float,
                       dt: float = 0.04, sub: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """RK4 VdP avec μ qui ramp linéairement. Returns (xy, mu_per_frame)."""
    def f(s, mu):
        x, y = s
        return np.array([y, mu * (1 - x * x) * y - x])
    def rk4(s, mu, dt):
        k1 = f(s, mu)
        k2 = f(s + dt / 2 * k1, mu)
        k3 = f(s + dt / 2 * k2, mu)
        k4 = f(s + dt * k3, mu)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # transient with μ = mu_start
    s = np.array([0.5, 0.0])
    for _ in range(400):
        s = rk4(s, mu_start, dt)

    mus = np.linspace(mu_start, mu_end, n_frames)
    traj = []
    for i in range(n_frames):
        mu_i = mus[i]
        for _ in range(sub):
            s = rk4(s, mu_i, dt)
        traj.append(s.copy())
    return np.array(traj), mus


def render_ramp_video(xy: np.ndarray, size: int = 64, trail_len: int = 12,
                       fixed_mx: float | None = None) -> np.ndarray:
    """Render trajectory as video frames.

    Args:
        fixed_mx: si donné, scale fixe au lieu d'auto-scale. Important pour
            détection bifurcation : préserve la taille relative attracteur
            (radius √μ pour VdP) à travers la ramp.
    """
    n = len(xy)
    if fixed_mx is None:
        mx = max(2.5, float(np.abs(xy).max() * 1.05))
    else:
        mx = fixed_mx
    cx = cy = size // 2
    scale = (size * 0.4) / mx

    frames = np.zeros((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        f = np.zeros((size, size, 3), dtype=np.uint8)
        f[size // 2, :, 0] = 30
        f[:, size // 2, 1] = 30
        # trail
        start = max(0, i - trail_len)
        for k, j in enumerate(range(start, i + 1)):
            x_p = int(cx + xy[j, 0] * scale)
            y_p = int(cy + xy[j, 1] * scale)
            if 0 <= x_p < size and 0 <= y_p < size:
                alpha = (k + 1) / (trail_len + 1)
                f[y_p, x_p, 0] = max(f[y_p, x_p, 0], int(200 * alpha))
                f[y_p, x_p, 1] = max(f[y_p, x_p, 1], int(220 * alpha))
                f[y_p, x_p, 2] = max(f[y_p, x_p, 2], int(255 * alpha))
        x_p = int(cx + xy[i, 0] * scale)
        y_p = int(cy + xy[i, 1] * scale)
        if 0 <= x_p < size and 0 <= y_p < size:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x_p + dx, y_p + dy
                    if 0 <= xx < size and 0 <= yy < size:
                        f[yy, xx, :] = 255
        frames[i] = f
    return frames


def sliding_eval(frames: np.ndarray, model_name: str,
                 window: int, stride: int) -> list[dict]:
    """Apply model on sliding windows. Returns list of metric dicts."""
    model = get_model(model_name)
    n = frames.shape[0]
    out = []
    centers = []
    for start in range(0, n - window + 1, stride):
        end = start + window
        center = start + window // 2
        clip = frames[start:end].astype(np.float32) / 255.0
        try:
            latents = model.produce_trajectory(clip)
            if latents.shape[0] < 8:
                continue
            coords = reduce_3d(latents, method="pca")
            analysis = analyse_trajectory(coords)
            verdict = classify_from_analysis(analysis, embedding_dim=3)
            out.append({
                "center_frame": center,
                "start": start,
                "end": end,
                "verdict_kind": verdict.kind,
                "confidence": verdict.confidence,
                "lyapunov": analysis["lyapunov"],
                "correlation_dim": analysis["correlation_dim"],
                "max_diag_ratio": analysis["max_diag_ratio"],
                "convergence_rate": analysis["convergence_rate"],
            })
            centers.append(center)
        except Exception as e:
            out.append({"center_frame": center, "error": str(e)})
    return out


def detect_changepoint(curve: list[float], smooth: int = 3) -> int:
    """CUSUM-like : argmax of moving |diff|, smoothed. Returns index."""
    arr = np.array(curve)
    if len(arr) < 4:
        return 0
    # Smooth
    if smooth > 1 and len(arr) >= smooth:
        kernel = np.ones(smooth) / smooth
        arr = np.convolve(arr, kernel, mode="valid")
    diffs = np.abs(np.diff(arr))
    return int(np.argmax(diffs))


def main():
    print("=" * 75)
    print("SLIDING-WINDOW BIFURCATION DETECTION — Van der Pol ramp")
    print(f"  μ ramps from {MU_START} to {MU_END} across {N_FRAMES} frames")
    print(f"  window={WINDOW}, stride={STRIDE}")
    print("=" * 75)

    t0 = time.time()
    print("\n[gen] VdP ramp trajectory + render...")
    xy, mus_per_frame = vdp_ramp_trajectory(N_FRAMES, MU_START, MU_END)
    print(f"  done : {len(xy)} (x,y) points")
    # Fixed scale based on expected max radius at MU_END (limit cycle ≈ 2)
    frames = render_ramp_video(xy, size=SIZE, fixed_mx=2.5)
    print(f"  rendered : {frames.shape} (fixed scale mx=2.5)")
    # Ground-truth transition frame : where μ crosses 0
    gt_frame = int(np.argmin(np.abs(mus_per_frame)))
    print(f"  ground truth bifurcation frame : {gt_frame} (μ ≈ {mus_per_frame[gt_frame]:+.3f})")

    models = [
        "delay_motion_auto_m3",
        "delay_entropy_auto_m3",
        "pca_obs_m3",
        "direct_bme",
    ]

    print("\n[eval] sliding windows...")
    results = {}
    for m in models:
        ts = time.time()
        out = sliding_eval(frames, m, WINDOW, STRIDE)
        results[m] = out
        n_ok = sum(1 for r in out if "error" not in r)
        print(f"  {m:<28}{n_ok} windows  ({time.time() - ts:.1f}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # For each model : detect changepoint on convergence_rate, regime_kind transitions
    print("\n" + "=" * 75)
    print(f"DETECTED TRANSITION FRAME (ground truth = {gt_frame})")
    print("=" * 75)
    summary = {}
    print(f"{'model':<28}{'cp_frame':>12}{'cp_err':>10}{'lead_time':>12}")
    for m in models:
        valid = [r for r in results[m] if "error" not in r]
        if not valid:
            continue
        conv_curve = [r["convergence_rate"] for r in valid]
        centers = [r["center_frame"] for r in valid]
        cp_idx = detect_changepoint(conv_curve, smooth=3)
        cp_frame = centers[cp_idx]
        err = cp_frame - gt_frame
        lead_time = -err  # positive = detected before transition
        summary[m] = {
            "cp_frame": cp_frame,
            "cp_err_frames": err,
            "lead_time_frames": lead_time,
            "gt_frame": gt_frame,
        }
        print(f"  {m:<28}{cp_frame:>12}{err:>+10}{lead_time:>+12}")

    # Regime kind transitions per model
    print("\n" + "=" * 75)
    print("REGIME KIND TRANSITIONS (window center frame → kind)")
    print("=" * 75)
    for m in models:
        valid = [r for r in results[m] if "error" not in r]
        if not valid:
            continue
        kinds = [r["verdict_kind"] for r in valid]
        centers = [r["center_frame"] for r in valid]
        transitions = []
        for i in range(1, len(kinds)):
            if kinds[i] != kinds[i - 1]:
                transitions.append((centers[i - 1], kinds[i - 1], centers[i], kinds[i]))
        print(f"\n  {m}:")
        if not transitions:
            print("    no transitions detected — single regime throughout")
        for prev_c, prev_k, c, k in transitions:
            mu_c = mus_per_frame[c]
            print(f"    frame {prev_c:>3}→{c:>3} : {prev_k:<10}→{k:<10}  (μ≈{mu_c:+.2f})")

    Path("results").mkdir(exist_ok=True)
    out = {
        "n_frames": N_FRAMES,
        "window": WINDOW,
        "stride": STRIDE,
        "mu_start": MU_START,
        "mu_end": MU_END,
        "gt_frame": gt_frame,
        "gt_mu_at_gt_frame": float(mus_per_frame[gt_frame]),
        "mus_per_frame": mus_per_frame.tolist(),
        "results_per_model": results,
        "summary": summary,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/sliding_bifurcation.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/sliding_bifurcation.json")


if __name__ == "__main__":
    main()
