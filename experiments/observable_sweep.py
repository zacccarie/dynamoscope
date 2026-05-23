"""Observable sweep — confirm remaining hypothesis (i) from §8.1.

After 4 falsified hypotheses (resolution, window, metric, trail), §8.1
identifies observable choice as the remaining source of lag variance.
This experiment tests directly : same render, same window, same metric,
sweep through all 12 observables as delay-embedding input.

If lags span a wide range across observables → hypothesis (i) confirmed.
If lags cluster around same value → even observable choice doesn't matter,
and a deeper factor is binding.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.observables import OBSERVABLE_KEYS
from backend.models.observable_producers import DelayEmbedProducer
from backend.reducer import reduce_3d
from backend.dynamics import analyse_trajectory
from experiments.sliding_bifurcation import (
    vdp_ramp_trajectory,
    render_ramp_video,
    detect_changepoint,
)


SIZE = 128
WINDOW = 40
STRIDE = 5
N_FRAMES = 400
MU_START = -0.5
MU_END = 1.5


def sliding_eval_producer(frames, producer, window, stride):
    """Like sliding_eval but takes pre-instantiated producer."""
    n = frames.shape[0]
    out = []
    for start in range(0, n - window + 1, stride):
        end = start + window
        center = start + window // 2
        clip = frames[start:end].astype(np.float32) / 255.0
        try:
            latents = producer.produce_trajectory(clip)
            if latents.shape[0] < 8:
                continue
            coords = reduce_3d(latents, method="pca")
            analysis = analyse_trajectory(coords)
            out.append({
                "center_frame": center,
                "convergence_rate": analysis["convergence_rate"],
                "lyapunov": analysis["lyapunov"],
                "max_diag_ratio": analysis["max_diag_ratio"],
            })
        except Exception:
            continue
    return out


def main():
    print("=" * 75)
    print(f"OBSERVABLE SWEEP — test §8.1 remaining hypothesis (i)")
    print(f"  size={SIZE}, window={WINDOW}, stride={STRIDE}")
    print(f"  μ ramps {MU_START}→{MU_END} across {N_FRAMES} frames")
    print("=" * 75)

    xy, mus_per_frame = vdp_ramp_trajectory(N_FRAMES, MU_START, MU_END)
    gt_frame = int(np.argmin(np.abs(mus_per_frame)))
    print(f"  ground truth μ=0 at frame {gt_frame}\n")

    frames = render_ramp_video(xy, size=SIZE, fixed_mx=2.5)

    rows = []
    t0 = time.time()
    for obs in OBSERVABLE_KEYS:
        producer = DelayEmbedProducer(obs, tau=None, m=3)
        ts = time.time()
        windows = sliding_eval_producer(frames, producer, WINDOW, STRIDE)
        if len(windows) < 5:
            print(f"  {obs:<14}too few windows ({len(windows)})")
            continue
        conv_curve = [w["convergence_rate"] for w in windows]
        centers = [w["center_frame"] for w in windows]
        cp_idx = detect_changepoint(conv_curve, smooth=3)
        cp_frame = centers[cp_idx]
        cp_mu = float(mus_per_frame[cp_frame])
        lag = cp_frame - gt_frame
        rows.append({
            "observable": obs,
            "cp_frame": cp_frame,
            "mu_vis": cp_mu,
            "lag_frames": lag,
            "tau_effective": producer.effective_tau,
            "compute_s": round(time.time() - ts, 1),
        })
        print(f"  {obs:<14}cp={cp_frame:>4}  μ_vis={cp_mu:+.3f}  lag={lag:>+4}f  "
              f"τ={producer.effective_tau}  ({rows[-1]['compute_s']}s)")

    print(f"\n[time] total {time.time() - t0:.1f}s")

    # Ranked
    sorted_rows = sorted(rows, key=lambda r: abs(r["lag_frames"]))
    print("\n" + "=" * 75)
    print("RANKED BY |LAG| (smallest lag = best detector)")
    print("=" * 75)
    for i, r in enumerate(sorted_rows):
        marker = " ←" if i == 0 else ("  best 3" if i < 3 else "")
        print(f"  {i + 1:>2}. {r['observable']:<14}lag={r['lag_frames']:>+4}f  "
              f"μ_vis={r['mu_vis']:+.3f}  τ={r['tau_effective']}{marker}")

    # Variance
    lags = [r["lag_frames"] for r in rows]
    spread = max(lags) - min(lags)
    print(f"\nLag spread across observables : {spread} frames "
          f"(min={min(lags)}, max={max(lags)}, mean={np.mean(lags):.1f})")
    if spread > 50:
        verdict = "CONFIRMED : observable choice is dominant source of lag variance"
    elif spread > 20:
        verdict = "PARTIAL : observable matters but other factors persist"
    else:
        verdict = "WEAK : even observable choice has small effect — deeper factor binding"
    print(f"\nVerdict : {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = {"rows": rows, "spread": spread, "gt_frame": gt_frame,
           "configs": {"size": SIZE, "window": WINDOW, "stride": STRIDE}}
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/observable_sweep.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/observable_sweep.json")


if __name__ == "__main__":
    main()
