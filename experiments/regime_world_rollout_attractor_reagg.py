"""Re-aggregate regime_world_rollout_attractor.json en filtrant les seeds
NaN/diverged. Issue identifiée : 1 seed sur 8 a divergé en training pour
RWM-stab+ms → poids NaN → toutes métriques NaN → arr.mean propage NaN
sur l'agrégat.

Fix : nanmean + reporter seed_health (n_finite / n_total).
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "regime_world_rollout_attractor.json"
DST = ROOT / "results" / "regime_world_rollout_attractor_clean.json"

REGIMES = ["smooth", "periodic", "chaotic"]
MODELS = ["rwm_orig", "rwm_stab_ms", "flat56", "flat56_ms"]


def is_finite_vec(v):
    arr = np.asarray(v, dtype=float)
    return bool(np.isfinite(arr).all()) and bool((np.abs(arr) < 1e15).all())


def main():
    data = json.load(open(SRC))
    K_max = data["K_max"]
    runs = data["all_runs"]
    print(f"Total seeds : {len(runs)}", flush=True)

    new_summary = {}
    for model in MODELS:
        mse_per_regime = {}
        attr_per_regime = {}
        seed_health = {r: 0 for r in REGIMES}
        for regime in REGIMES:
            mse_stack = []
            attr_stack = []
            for run in runs:
                if model not in run: continue
                res = run[model]
                if regime in res["mse_per_regime"]:
                    v = res["mse_per_regime"][regime]
                    if is_finite_vec(v):
                        mse_stack.append(v)
                        seed_health[regime] += 1
                if regime in res["attractor_per_regime"]:
                    a = res["attractor_per_regime"][regime]
                    # filter NaN attractor fields
                    clean = {}
                    valid = True
                    for k, vv in a.items():
                        val = vv[0] if isinstance(vv, list) else vv
                        if val is None or not np.isfinite(val):
                            valid = False
                            break
                        clean[k] = val
                    if valid:
                        attr_stack.append(clean)
            if mse_stack:
                arr = np.array(mse_stack)
                mse_per_regime[regime] = {
                    "mean": arr.mean(axis=0).tolist(),
                    "std": arr.std(axis=0).tolist(),
                    "n_seeds_used": int(arr.shape[0]),
                    "n_seeds_total": len(runs),
                }
            else:
                mse_per_regime[regime] = {"mean": [float("nan")] * K_max,
                                            "std": [float("nan")] * K_max,
                                            "n_seeds_used": 0,
                                            "n_seeds_total": len(runs)}
            attr_summ = {}
            if attr_stack:
                all_keys = attr_stack[0].keys()
                for k in all_keys:
                    vals = [s[k] for s in attr_stack if k in s]
                    if vals:
                        attr_summ[k] = (float(np.mean(vals)), float(np.std(vals)))
                attr_summ["_n_seeds_used"] = len(attr_stack)
            attr_per_regime[regime] = attr_summ
        new_summary[model] = {"mse": mse_per_regime, "attractor": attr_per_regime,
                                "seed_health_mse": seed_health}

    # Print table
    print("\n" + "=" * 96, flush=True)
    print("MSE per K per regime (CLEAN, NaN seeds filtered)", flush=True)
    print("=" * 96, flush=True)
    for regime in REGIMES:
        print(f"\n--- {regime.upper()} ---", flush=True)
        print(f"  {'K':>3}{'RWM-orig':>18}{'RWM-stab+ms':>18}"
              f"{'Flat-56':>18}{'Flat-56+ms':>18}", flush=True)
        for k in [0, 3, 7, 15, 23]:
            row = [f"  {k+1:>3}"]
            for key in MODELS:
                mse = new_summary[key]["mse"][regime]
                m = mse["mean"][k] if k < len(mse["mean"]) else float("nan")
                s = mse["std"][k] if k < len(mse["std"]) else float("nan")
                n = mse["n_seeds_used"]
                row.append(f"{m:.3e}±{s:.0e}({n})".rjust(18))
            print("".join(row), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("Attractor preservation (CLEAN)", flush=True)
    print("=" * 96, flush=True)
    for regime in REGIMES:
        print(f"\n--- {regime.upper()} ---", flush=True)
        for key in MODELS:
            a = new_summary[key]["attractor"].get(regime, {})
            n = a.get("_n_seeds_used", 0)
            def f(k):
                v = a.get(k, None)
                if v is None: return "n/a"
                return f"{v[0]:.3f}"
            print(f"  {key:<14} N={n}/{len(runs)}  "
                  f"bnd={f('bounded_frac')}  spec={f('spectral_match_mean')}  "
                  f"acf={f('autocorr_match_mean')}  "
                  f"DET={f('recurrence_DET_match_mean')}  "
                  f"lyap={f('lyap_match_mean')}", flush=True)

    # Capacity verdict
    K = K_max - 1
    def mean_K(model):
        vals = []
        for r in REGIMES:
            v = new_summary[model]["mse"][r]["mean"]
            if K < len(v) and np.isfinite(v[K]):
                vals.append(v[K])
        return float(np.mean(vals)) if vals else float("nan")
    rwm_K = mean_K("rwm_stab_ms")
    flat_K = mean_K("flat56")
    flat_ms_K = mean_K("flat56_ms")
    print("\n" + "=" * 96, flush=True)
    print("CAPACITY-FAIRNESS VERDICT (CLEAN)", flush=True)
    print("=" * 96, flush=True)
    print(f"  K={K+1} mean MSE across regimes :", flush=True)
    print(f"    RWM-stab+ms : {rwm_K:.4f}", flush=True)
    print(f"    Flat-56     : {flat_K:.4f}", flush=True)
    print(f"    Flat-56+ms  : {flat_ms_K:.4f}", flush=True)
    print(f"  RWM-stab+ms / Flat-56    : {rwm_K/max(flat_K, 1e-9):.3f}", flush=True)
    print(f"  RWM-stab+ms / Flat-56+ms : {rwm_K/max(flat_ms_K, 1e-9):.3f}   "
          f"← FAIR", flush=True)
    print(f"  Flat-56+ms / Flat-56     : {flat_ms_K/max(flat_K, 1e-9):.3f}   "
          f"← ms loss effect", flush=True)

    capacity_verdict = {
        "rwm_stab_ms_K_last": rwm_K, "flat56_K_last": flat_K,
        "flat56_ms_K_last": flat_ms_K,
        "ratio_vs_flat": rwm_K / max(flat_K, 1e-9),
        "ratio_vs_flat_ms": rwm_K / max(flat_ms_K, 1e-9),
        "multistep_effect_on_flat": flat_ms_K / max(flat_K, 1e-9),
    }

    out = {
        "n_seeds": len(runs), "K_max": K_max,
        "summary": new_summary,
        "capacity_verdict": capacity_verdict,
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): _d(v) for k, v in o.items()}
        if isinstance(o, tuple): return [_d(x) for x in o]
        raise TypeError(f"{type(o)}")
    json.dump(out, open(DST, "w"), indent=2, default=_d)
    print(f"\n[saved] {DST}", flush=True)


if __name__ == "__main__":
    main()
