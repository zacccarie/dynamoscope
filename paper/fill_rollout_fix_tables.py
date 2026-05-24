"""Remplit les placeholders MSE_*_*_K et ATTR_*_*_* dans dynamoscope.tex
à partir de results/regime_world_rollout_attractor.json.

Usage : .venv/bin/python paper/fill_rollout_fix_tables.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "regime_world_rollout_attractor_clean.json"
TEX = ROOT / "paper" / "dynamoscope.tex"

REGIMES = {"smooth": "s", "periodic": "p", "chaotic": "c"}
MODELS = {"rwm_orig": "orig", "rwm_stab_ms": "stabms",
          "flat56": "flat", "flat56_ms": "flatms"}
K_REPORTED = [0, 3, 7, 15, 23]   # K = 1, 4, 8, 16, 24
# Use 2-digit zero-padding (K01, K04, K08, K16, K24) so no placeholder
# is a prefix of another — avoids partial substitution.


def fmt_mse(mean: float, std: float) -> str:
    """Format MSE with abbreviated scientific if huge, decimal if small."""
    if mean > 1e3 or std > 1e3:
        return f"${mean:.1e}$"
    if mean > 1.0:
        return f"${mean:.2f}$"
    return f"${mean:.3f}$"


def fmt_attr(v: float) -> str:
    return f"${v:.3f}$"


def fmt_bool(v: float) -> str:
    if v >= 0.99: return "$1.00$"
    return f"${v:.2f}$"


def main():
    data = json.load(open(RESULTS))
    summary = data["summary"]
    tex = TEX.read_text()
    replaced = 0

    # MSE table
    for regime, r_abbr in REGIMES.items():
        for model, m_abbr in MODELS.items():
            mse_data = summary[model]["mse"][regime]
            mean = mse_data["mean"]
            std = mse_data["std"]
            for K_idx in K_REPORTED:
                k_disp = K_idx + 1
                ph = f"MSE-{r_abbr}-{m_abbr}-K{k_disp:02d}"
                val = fmt_mse(mean[K_idx], std[K_idx])
                if ph in tex:
                    tex = tex.replace(ph, val)
                    replaced += 1
                else:
                    print(f"  missing placeholder: {ph}")

    # Attractor table
    for regime, r_abbr in REGIMES.items():
        for model, m_abbr in MODELS.items():
            attr = summary[model]["attractor"].get(regime, {})
            mapping = {
                "bnd":  ("bounded_frac", fmt_bool),
                "spec": ("spectral_match_mean", fmt_attr),
                "acf":  ("autocorr_match_mean", fmt_attr),
                "det":  ("recurrence_DET_match_mean", fmt_attr),
                "lyap": ("lyap_match_mean", fmt_attr),
            }
            for short, (key, fmt) in mapping.items():
                ph = f"ATTR-{r_abbr}-{m_abbr}-{short}"
                v = attr.get(key, None)
                # Clean JSON stores (mean, std) tuples as 2-element lists
                if isinstance(v, list):
                    v = v[0]
                if v is None or (isinstance(v, float) and v != v):
                    val = "n/a"
                else:
                    val = fmt(v)
                if ph in tex:
                    tex = tex.replace(ph, val)
                    replaced += 1
                else:
                    print(f"  missing placeholder: {ph}")

    # Fair ratio in narrative
    cv = data["capacity_verdict"]
    ratio_fair = cv["ratio_vs_flat_ms"]
    # Ratio < 1 means RWM-stab+ms better; report as 1/ratio× better
    if ratio_fair < 1:
        ratio_disp = f"{1.0 / ratio_fair:.2f}"
    else:
        ratio_disp = f"{ratio_fair:.2f}"
    fair_ph = "RATIO\\_FAIR\\_PLACEHOLDER"
    if fair_ph in tex:
        tex = tex.replace(fair_ph, ratio_disp)
        replaced += 1
    else:
        print(f"  missing narrative ratio placeholder")

    TEX.write_text(tex)
    print(f"\n[done] {replaced} substitutions written to {TEX}")
    print(f"\nKey numbers :")
    print(f"  RWM-stab+ms vs Flat-56    : {cv['ratio_vs_flat']:.3f}")
    print(f"  RWM-stab+ms vs Flat-56+ms : {cv['ratio_vs_flat_ms']:.3f}  ← FAIR")
    print(f"  Multi-step alone effect   : {cv['multistep_effect_on_flat']:.3f}")


if __name__ == "__main__":
    main()
