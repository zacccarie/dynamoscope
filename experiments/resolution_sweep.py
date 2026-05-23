"""Resolution sweep — validate §8.1 Heisenberg prediction μ_vis ∝ 1/resolution².

§8.1 claims : detection lag bounded by μ_vis where cycle radius √μ exceeds
pixel resolution. Render scale = (size * 0.4) / mx. For fixed mx=2.5,
scale ∝ size. Cycle radius √μ becomes one-pixel-distinct when
scale * √μ > 1 → √μ > 1/scale → μ_vis > 1/scale².

Predictions :
  size  scale_pix/unit  μ_vis_predicted
  64    10.24           0.0095   (with 1-pixel threshold)
  128   20.48           0.0024
  256   40.96           0.0006

BUT : we observed μ_vis ≈ 0.5 at 64×64, much higher than 0.0095.
True threshold is more like ~5 pixels (visible distinction needs SHAPE,
not just 1 pixel offset). With 5-pixel threshold :
  64    μ_vis = 25/scale² = 25/10.24² ≈ 0.24 (closer to observed 0.5)
  128   μ_vis ≈ 0.060
  256   μ_vis ≈ 0.015

We measure : at higher resolution, detection should happen at MUCH smaller μ.
If μ_vis scales as 1/scale² (i.e. /4 each doubling), prediction confirmed.
If μ_vis scales slower (e.g. 1/scale), other factors dominate (delay-embedding
sensitivity, regime classifier thresholds, etc).

Output : table size × {detected_frame, detected_μ, predicted_μ_vis}.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.sliding_bifurcation import (
    vdp_ramp_trajectory,
    render_ramp_video,
    sliding_eval,
    detect_changepoint,
)


SIZES = [64, 128, 256]
N_FRAMES = 400
WINDOW = 40
STRIDE = 5
MU_START = -0.5
MU_END = 1.5
FIXED_MX = 2.5  # attractor radius bound


def predicted_mu_vis(size: int, mx: float = FIXED_MX, pixel_threshold: float = 5.0) -> float:
    """Cycle radius √μ visible when scale × √μ > threshold → μ_vis > (threshold/scale)²."""
    scale = (size * 0.4) / mx  # pixels per unit
    return (pixel_threshold / scale) ** 2


def main():
    print("=" * 75)
    print(f"RESOLUTION SWEEP — validate μ_vis ∝ 1/resolution²")
    print(f"  sizes : {SIZES}")
    print(f"  μ ramps {MU_START} → {MU_END} across {N_FRAMES} frames")
    print(f"  Hopf bifurcation at frame {int(N_FRAMES * (0 - MU_START) / (MU_END - MU_START))}")
    print("=" * 75)

    # Generate one VdP trajectory, then render at multiple resolutions
    xy, mus_per_frame = vdp_ramp_trajectory(N_FRAMES, MU_START, MU_END)
    gt_frame = int(np.argmin(np.abs(mus_per_frame)))

    # Single classical detector (delay_entropy = best in §8.1)
    model = "delay_entropy_auto_m3"

    rows = []
    t0 = time.time()
    for size in SIZES:
        ts = time.time()
        frames = render_ramp_video(xy, size=size, fixed_mx=FIXED_MX)
        windows = sliding_eval(frames, model, WINDOW, STRIDE)
        valid = [r for r in windows if "error" not in r]
        conv_curve = [r["convergence_rate"] for r in valid]
        centers = [r["center_frame"] for r in valid]
        cp_idx = detect_changepoint(conv_curve, smooth=3)
        cp_frame = centers[cp_idx]
        cp_mu = float(mus_per_frame[cp_frame])
        lag_frames = cp_frame - gt_frame
        mu_vis_observed = cp_mu  # detected μ
        mu_vis_pred = predicted_mu_vis(size)

        rows.append({
            "size": size,
            "cp_frame": cp_frame,
            "lag_frames": lag_frames,
            "mu_vis_observed": mu_vis_observed,
            "mu_vis_predicted": mu_vis_pred,
            "ratio_obs_pred": mu_vis_observed / max(mu_vis_pred, 1e-9),
            "n_windows": len(valid),
            "compute_s": round(time.time() - ts, 1),
        })
        print(f"  size={size:>4}  cp_frame={cp_frame:>4}  μ_obs={mu_vis_observed:+.4f}  "
              f"μ_pred={mu_vis_pred:.4f}  ratio={rows[-1]['ratio_obs_pred']:.2f}  "
              f"lag={lag_frames:>+4}f  ({rows[-1]['compute_s']}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Scaling analysis
    print("\n" + "=" * 75)
    print("SCALING ANALYSIS")
    print("=" * 75)
    print(f"{'size':>6}{'μ_vis observed':>18}{'×inv from prev':>20}{'expected (÷4)':>18}")
    for i, r in enumerate(rows):
        if i == 0:
            ratio = "—"
            expected = "—"
        else:
            ratio = f"{rows[i - 1]['mu_vis_observed'] / max(r['mu_vis_observed'], 1e-9):.2f}"
            expected = "4.0"
        print(f"{r['size']:>6}{r['mu_vis_observed']:>+18.4f}{ratio:>20}{expected:>18}")

    # Verdict
    if len(rows) >= 2:
        ratios = []
        for i in range(1, len(rows)):
            r = rows[i - 1]["mu_vis_observed"] / max(rows[i]["mu_vis_observed"], 1e-9)
            ratios.append(r)
        mean_ratio = float(np.mean(ratios))
        print(f"\nMean μ_vis-scaling-ratio per doubling : {mean_ratio:.2f}")
        if 3.0 < mean_ratio < 5.0:
            verdict = "CONFIRMED : μ_vis ∝ 1/resolution² (Heisenberg prediction holds)"
        elif 1.5 < mean_ratio < 2.5:
            verdict = "PARTIAL : μ_vis ∝ 1/resolution (linear, not quadratic)"
        elif mean_ratio < 1.5:
            verdict = "FALSIFIED : detection limit dominated by non-resolution factors"
        else:
            verdict = "STRONGER THAN PREDICTED : super-quadratic scaling"
        print(f"\nVerdict : {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "rows": rows,
        "model": model,
        "gt_frame": gt_frame,
        "configs": {"n_frames": N_FRAMES, "window": WINDOW, "stride": STRIDE,
                    "mu_start": MU_START, "mu_end": MU_END, "fixed_mx": FIXED_MX},
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/resolution_sweep.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/resolution_sweep.json")


if __name__ == "__main__":
    main()
