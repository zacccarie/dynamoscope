"""Window-size sweep — validate §8.1 revised hypothesis : lag is temporal-bound.

§8.1 (revised after resolution sweep falsification) claims detection lag
bound by window size + classifier thresholds, NOT pixel resolution.
Test directly : vary WINDOW ∈ {15, 30, 60, 120}, keep all else equal,
measure detected μ_vis.

Predictions :
- Smaller window → less temporal smoothing → earlier detection → smaller μ_vis
- If lag ∝ window : confirms temporal-bound hypothesis
- If lag stable across window : classifier thresholds dominate
- Lower bound : ~5 frames (RQA needs minimum trajectory length)

Output : table window × {detected_frame, μ_vis, lag}.
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


WINDOWS = [15, 30, 60, 120]
SIZE = 128  # mid resolution
N_FRAMES = 400
STRIDE = 5
MU_START = -0.5
MU_END = 1.5


def main():
    print("=" * 75)
    print(f"WINDOW SWEEP — test temporal hypothesis (§8.1 revised)")
    print(f"  windows : {WINDOWS}")
    print(f"  size={SIZE}, μ ramps {MU_START}→{MU_END} across {N_FRAMES} frames")
    print("=" * 75)

    xy, mus_per_frame = vdp_ramp_trajectory(N_FRAMES, MU_START, MU_END)
    gt_frame = int(np.argmin(np.abs(mus_per_frame)))
    print(f"  ground truth bifurcation frame : {gt_frame} (μ = {mus_per_frame[gt_frame]:+.3f})")

    frames = render_ramp_video(xy, size=SIZE, fixed_mx=2.5)
    print(f"  rendered : {frames.shape}")

    model = "delay_entropy_auto_m3"
    rows = []
    t0 = time.time()
    for win in WINDOWS:
        ts = time.time()
        windows = sliding_eval(frames, model, win, STRIDE)
        valid = [r for r in windows if "error" not in r]
        if len(valid) < 5:
            print(f"  window={win:>4}  too few valid windows ({len(valid)}), skip")
            continue
        conv_curve = [r["convergence_rate"] for r in valid]
        centers = [r["center_frame"] for r in valid]
        cp_idx = detect_changepoint(conv_curve, smooth=3)
        cp_frame = centers[cp_idx]
        cp_mu = float(mus_per_frame[cp_frame])
        lag = cp_frame - gt_frame
        rows.append({
            "window": win,
            "cp_frame": cp_frame,
            "mu_vis": cp_mu,
            "lag_frames": lag,
            "n_windows": len(valid),
            "compute_s": round(time.time() - ts, 1),
        })
        print(f"  window={win:>4}  cp_frame={cp_frame:>4}  μ_vis={cp_mu:+.3f}  "
              f"lag={lag:>+4}f  ({rows[-1]['compute_s']}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Scaling analysis : does lag scale with window?
    print("\n" + "=" * 75)
    print("SCALING ANALYSIS")
    print("=" * 75)
    print(f"{'window':>8}{'lag (frames)':>15}{'×inv from prev':>20}{'expected (÷2)':>18}")
    for i, r in enumerate(rows):
        if i == 0 or rows[i - 1]["lag_frames"] <= 0:
            ratio_str = "—"
        else:
            ratio = rows[i - 1]["lag_frames"] / max(r["lag_frames"], 1)
            ratio_str = f"{ratio:.2f}"
        expected = "2.0" if i > 0 else "—"
        print(f"{r['window']:>8}{r['lag_frames']:>+15}{ratio_str:>20}{expected:>18}")

    # Verdict
    if len(rows) >= 2:
        # Test linear scaling: lag(small) / lag(large) ≈ 1, lag(large) / lag(small) ≈ 1 if independent
        # If lag ∝ window : ratio of lag should equal ratio of window
        ratios = []
        for i in range(1, len(rows)):
            lag_ratio = rows[i - 1]["lag_frames"] / max(rows[i]["lag_frames"], 1)
            win_ratio = rows[i - 1]["window"] / rows[i]["window"]
            ratios.append((lag_ratio, win_ratio))
        mean_lag_ratio = float(np.mean([r[0] for r in ratios]))
        mean_win_ratio = float(np.mean([r[1] for r in ratios]))
        print(f"\nMean lag-ratio per window-halving : {mean_lag_ratio:.2f}")
        print(f"Mean window-ratio per step          : {mean_win_ratio:.2f}")
        if mean_lag_ratio > 1.5:
            verdict = "CONFIRMED : lag scales with window size (temporal-bound)"
        elif 0.7 < mean_lag_ratio < 1.5:
            verdict = "FLAT : lag invariant to window — classifier thresholds dominate"
        else:
            verdict = "INVERSE : larger window gives SMALLER lag — unexpected"
        print(f"\nVerdict : {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "rows": rows,
        "model": model,
        "gt_frame": gt_frame,
        "configs": {"size": SIZE, "n_frames": N_FRAMES, "stride": STRIDE,
                    "mu_start": MU_START, "mu_end": MU_END},
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/window_sweep.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/window_sweep.json")


if __name__ == "__main__":
    main()
