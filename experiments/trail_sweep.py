"""Trail-width sweep — test §8.1 synthesis claim.

After 3 falsified hypotheses (resolution, window, metric), §8.1 synthesis
claims the lag floor comes from interaction trail-width × observable
computation × classifier thresholds. Direct test : vary trail_len.

Trail_len = number of past positions drawn fading. Trail width in pixels
≈ trail_len × cycle_speed × dt × scale. If trail crosses itself rapidly
when cycle radius is small, observable signal is dominated by trail
rather than cycle structure.

Predictions :
- Short trail (1-3 frames) : less self-overlap → smaller lag
- Long trail (12-24 frames) : self-overlap masks cycle structure → larger lag
- If lag inversely correlates with trail length → claim confirmed

Sizes : single resolution 128, window 40, stride 5.
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


SIZE = 128
WINDOW = 40
STRIDE = 5
N_FRAMES = 400
MU_START = -0.5
MU_END = 1.5
TRAIL_LENGTHS = [1, 3, 6, 12, 24]


def main():
    print("=" * 75)
    print(f"TRAIL WIDTH SWEEP — test §8.1 synthesis claim")
    print(f"  trail_len : {TRAIL_LENGTHS}")
    print(f"  size={SIZE}, window={WINDOW}")
    print("=" * 75)

    xy, mus_per_frame = vdp_ramp_trajectory(N_FRAMES, MU_START, MU_END)
    gt_frame = int(np.argmin(np.abs(mus_per_frame)))
    print(f"  ground truth μ=0 at frame {gt_frame}\n")

    model = "delay_entropy_auto_m3"
    rows = []
    t0 = time.time()

    for tl in TRAIL_LENGTHS:
        ts = time.time()
        frames = render_ramp_video(xy, size=SIZE, fixed_mx=2.5, trail_len=tl)
        windows = sliding_eval(frames, model, WINDOW, STRIDE)
        valid = [r for r in windows if "error" not in r]
        if len(valid) < 5:
            continue
        conv_curve = [r["convergence_rate"] for r in valid]
        centers = [r["center_frame"] for r in valid]
        cp_idx = detect_changepoint(conv_curve, smooth=3)
        cp_frame = centers[cp_idx]
        cp_mu = float(mus_per_frame[cp_frame])
        lag = cp_frame - gt_frame

        rows.append({
            "trail_len": tl,
            "cp_frame": cp_frame,
            "mu_vis": cp_mu,
            "lag_frames": lag,
            "compute_s": round(time.time() - ts, 1),
        })
        print(f"  trail_len={tl:>3}  cp_frame={cp_frame:>4}  μ_vis={cp_mu:+.3f}  "
              f"lag={lag:>+4}f  ({rows[-1]['compute_s']}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Analysis : does lag scale with trail length?
    print("\n" + "=" * 75)
    print("SCALING ANALYSIS")
    print("=" * 75)
    print(f"{'trail_len':>10}{'lag':>10}{'μ_vis':>10}")
    for r in rows:
        print(f"{r['trail_len']:>10}{r['lag_frames']:>+10}{r['mu_vis']:>+10.3f}")

    if len(rows) >= 2:
        # Correlation lag vs trail_len
        tls = np.array([r["trail_len"] for r in rows])
        lags = np.array([r["lag_frames"] for r in rows])
        # Pearson
        tls_c = tls - tls.mean()
        lags_c = lags - lags.mean()
        num = (tls_c * lags_c).sum()
        den = (tls_c.std() * lags_c.std() * len(tls))
        corr = float(num / max(den, 1e-9))
        print(f"\nPearson correlation(trail_len, lag) : {corr:+.3f}")
        if corr > 0.7:
            verdict = "CONFIRMED : longer trail → larger lag (trail masks cycle)"
        elif corr > 0.3:
            verdict = "PARTIAL : weak positive correlation, trail contributes"
        elif corr > -0.3:
            verdict = "NEUTRAL : trail length has no clear effect"
        else:
            verdict = "INVERSE : longer trail → smaller lag (unexpected)"
        print(f"\nVerdict : {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = {"rows": rows, "model": model, "gt_frame": gt_frame}
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/trail_sweep.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/trail_sweep.json")


if __name__ == "__main__":
    main()
