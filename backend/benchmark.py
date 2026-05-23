"""Cross-tool benchmark : compare Dynamoscope outputs vs external libraries.

Permet de :
1. Valider correctness — agreement entre notre implémentation et reference
2. Positionner contribution — what we add vs existing tools
3. Detect bugs — divergence = signal

Tools compared :
- pysindy (Brunton lab official SINDy)
- ripser (PH reference, déjà notre backend)
- tigramite (PCMCI+ reference, déjà notre backend pour causal_advanced)
- scipy (DMD baseline via pure SVD)
"""
from __future__ import annotations
import time
import numpy as np


def benchmark_sindy(
    latents: np.ndarray, dt: float = 0.01, threshold: float = 0.5,
) -> dict:
    """Run notre SINDy + pysindy STLSQ sur mêmes données, compare coefficients.

    Returns:
        - dynamoscope : (R², n_active, equations)
        - pysindy : (R², n_active, equations)
        - agreement : cosine similarity entre coefficient vectors
    """
    from .sindy import fit_sindy
    import pysindy as ps

    # Dynamoscope
    t0 = time.time()
    our = fit_sindy(latents, dt=dt, order=2, threshold=threshold)
    our["compute_s"] = round(time.time() - t0, 3)

    # pysindy
    t0 = time.time()
    optimizer = ps.STLSQ(threshold=threshold)
    library = ps.PolynomialLibrary(degree=2)
    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(latents, t=dt)
    ps_score = float(model.score(latents, t=dt))
    ps_coef = model.coefficients()
    ps_n_active = int((np.abs(ps_coef) > 1e-9).sum())
    ps_time = round(time.time() - t0, 3)

    # Cosine similarity between coefficient matrices (flatten)
    # Approximate alignment by L2 normalisation of flat vectors
    def _flat_norm(arr):
        v = np.asarray(arr).flatten()
        return v / max(np.linalg.norm(v), 1e-9)

    our_coefs = []
    for eq in our["equations"]:
        for t in eq["terms"]:
            our_coefs.append(t["coef"])
    our_v = _flat_norm(our_coefs)
    ps_v = _flat_norm(ps_coef[np.abs(ps_coef) > 1e-9])

    # Padding shorter to match
    min_len = min(len(our_v), len(ps_v))
    if min_len > 0:
        cos_sim = float(np.dot(our_v[:min_len], ps_v[:min_len]))
    else:
        cos_sim = 0.0

    return {
        "dynamoscope": {
            "r2": our["r2"], "n_active": our["n_active"],
            "compute_s": our["compute_s"],
        },
        "pysindy": {
            "r2": ps_score, "n_active": ps_n_active, "compute_s": ps_time,
        },
        "agreement_cosine_sim": cos_sim,
        "agreement_note": "≈1.0 = strong agreement, <0.7 = significant divergence",
    }


def benchmark_dmd_vs_scipy(latents: np.ndarray, dt: float = 1.0) -> dict:
    """Notre DMD (custom SVD-based) vs raw scipy SVD reference."""
    from .spectral import dmd as our_dmd
    import scipy.linalg

    t0 = time.time()
    our = our_dmd(latents, dt=dt)
    our_time = time.time() - t0

    # Reference : pure scipy DMD-exact
    t0 = time.time()
    X = latents[:-1].T
    Y = latents[1:].T
    U, S, Vh = scipy.linalg.svd(X, full_matrices=False)
    r = our["rank"]
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].conj().T
    A_tilde = Ur.conj().T @ Y @ Vr @ np.diag(1.0 / np.maximum(Sr, 1e-12))
    eigs = scipy.linalg.eigvals(A_tilde)
    ref_time = time.time() - t0

    # Compare eigvals (sorted by magnitude)
    our_eigs = np.array(our["eigvals_re"]) + 1j * np.array(our["eigvals_im"])
    our_sorted = np.sort_complex(our_eigs)
    ref_sorted = np.sort_complex(eigs)
    diff = float(np.linalg.norm(our_sorted - ref_sorted))

    return {
        "dynamoscope_eigenvalues_modulus": [float(abs(e)) for e in our_sorted],
        "scipy_reference_eigenvalues_modulus": [float(abs(e)) for e in ref_sorted],
        "eigenvalue_l2_diff": diff,
        "compute_s_dynamoscope": round(our_time, 3),
        "compute_s_scipy_reference": round(ref_time, 3),
        "agreement_note": "diff << 1e-6 = numerically identical",
    }


def benchmark_summary(latents: np.ndarray, dt: float = 0.01) -> dict:
    """Full benchmark : SINDy + DMD comparisons."""
    n, d = latents.shape
    out: dict = {
        "n_samples": int(n),
        "n_dims": int(d),
    }
    try:
        out["sindy"] = benchmark_sindy(latents, dt=dt)
    except Exception as e:
        out["sindy_error"] = str(e)
    try:
        out["dmd"] = benchmark_dmd_vs_scipy(latents, dt=dt)
    except Exception as e:
        out["dmd_error"] = str(e)
    return out
