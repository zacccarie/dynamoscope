"""Metric response sweep — test §8.1 residual hypothesis.

§8.1 (final form) : after Heisenberg + window hypotheses falsified,
the remaining explanation is that detection lag is bound by the
convergence_rate metric's own response curve to μ.

Test : run sliding analysis once, then run change-point detection on
4 different metrics : convergence_rate, max_diag_ratio (Lmax/N),
RQA DET, lyapunov. Compare detected μ_vis per metric.

If a sharper-response metric detects at μ << 0.5 → hypothesis 3
confirmed (metric-bound). If all metrics detect near μ=0.5 → deeper
issue (e.g. video rendering insensitive).
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

METRICS = ["convergence_rate", "max_diag_ratio", "rqa_det_proxy", "lyapunov", "correlation_dim"]
# rqa_det_proxy from analysis dict not exposed directly in sliding_eval ;
# we'll add a quick map via re-analysis.


def main():
    print("=" * 75)
    print("METRIC RESPONSE SWEEP — test §8.1 hypothesis 3")
    print(f"  size={SIZE}, window={WINDOW}, stride={STRIDE}")
    print("=" * 75)

    xy, mus_per_frame = vdp_ramp_trajectory(N_FRAMES, MU_START, MU_END)
    gt_frame = int(np.argmin(np.abs(mus_per_frame)))
    print(f"  ground truth μ=0 at frame {gt_frame}")

    frames = render_ramp_video(xy, size=SIZE, fixed_mx=2.5)

    model = "delay_entropy_auto_m3"
    windows = sliding_eval(frames, model, WINDOW, STRIDE)
    valid = [r for r in windows if "error" not in r]
    centers = [r["center_frame"] for r in valid]
    print(f"  {len(valid)} valid windows")

    # Collect each metric series
    series = {}
    for m in ["convergence_rate", "max_diag_ratio", "lyapunov", "correlation_dim"]:
        series[m] = [r[m] for r in valid]
    # RQA DET not in sliding_eval output ; re-compute via direct call
    # Actually we can use max_diag_ratio as proxy for periodicity sharpness
    # (high in cycle regime, low in chaos/noise). For DET, would need a
    # second pass — skip to keep this experiment focused.

    rows = []
    t0 = time.time()
    print()
    for metric, curve in series.items():
        cp_idx = detect_changepoint(curve, smooth=3)
        cp_frame = centers[cp_idx]
        cp_mu = float(mus_per_frame[cp_frame])
        lag = cp_frame - gt_frame
        # Compute monotonicity of curve : do values change in expected direction?
        arr = np.array(curve)
        # Sign of value at high-μ window vs low-μ window
        v_low = float(np.mean(arr[:5]))
        v_high = float(np.mean(arr[-5:]))
        delta_total = v_high - v_low
        rows.append({
            "metric": metric,
            "cp_frame": cp_frame,
            "mu_vis": cp_mu,
            "lag_frames": lag,
            "v_low_mu": v_low,
            "v_high_mu": v_high,
            "delta_total": delta_total,
        })
        print(f"  {metric:<22}cp_frame={cp_frame:>4}  μ_vis={cp_mu:+.3f}  "
              f"lag={lag:>+4}f  Δtotal={delta_total:+.3f}")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Find best metric
    sorted_rows = sorted(rows, key=lambda r: abs(r["lag_frames"]))
    print("\n" + "=" * 75)
    print("RANKED BY |LAG| (smaller = better detector)")
    print("=" * 75)
    for i, r in enumerate(sorted_rows):
        marker = " ← best" if i == 0 else ""
        print(f"  {i + 1}. {r['metric']:<22}lag={r['lag_frames']:>+4}f  μ_vis={r['mu_vis']:+.3f}{marker}")

    # Verdict
    best_lag = sorted_rows[0]["lag_frames"]
    if abs(best_lag) < 30:
        verdict = "CONFIRMED : sharper metric resolves detection (hypothesis 3 holds)"
    elif abs(best_lag) < 70:
        verdict = "PARTIAL : different metric helps but still substantial lag"
    else:
        verdict = "NEGATIVE : all metrics show similar lag — bottleneck is video signal itself"
    print(f"\nVerdict : {verdict}")
    print(f"Best metric : {sorted_rows[0]['metric']}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "rows": rows,
        "model": model,
        "gt_frame": gt_frame,
        "configs": {"size": SIZE, "window": WINDOW, "stride": STRIDE,
                    "n_frames": N_FRAMES, "mu_start": MU_START, "mu_end": MU_END},
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/metric_sweep.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/metric_sweep.json")


if __name__ == "__main__":
    main()
