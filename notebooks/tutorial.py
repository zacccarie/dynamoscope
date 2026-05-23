# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dynamoscope · tutorial
#
# Walkthrough du pipeline. Exécutable cellule par cellule dans Jupyter/VS Code.
# Format jupytext `py:percent` : `# %%` = cellule code, `# %% [markdown]` = cellule markdown.
#
# Convert to notebook :
# ```bash
# jupytext --to ipynb tutorial.py
# ```
#
# ## Objectifs
# 1. Encoder une trajectoire (Lorenz synthétique)
# 2. Projeter en 3D via UMAP/PCA
# 3. Lancer toutes les analyses
# 4. Comparer 3 encoders sur même vidéo
# 5. Visualiser via matplotlib

# %% [markdown]
# ## 1. Setup

# %%
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.style.use("dark_background")

# %% [markdown]
# ## 2. Système synthétique : Lorenz
#
# Génère trajectoire chaos canonique. σ=10, ρ=28, β=8/3 → exposant Lyapunov ≈ 0.91.

# %%
from backend.systems import lorenz

lor = lorenz(n=2000)
coords = np.array(lor["coords"])  # normalisé [-1,1]³
raw = np.array(lor["raw_coords"])  # brut, échelle physique

print(f"shape: {coords.shape}, dt={lor['dt']}, vars={lor['var_names']}")
print(f"raw range: x∈[{raw[:,0].min():.2f}, {raw[:,0].max():.2f}], "
      f"y∈[{raw[:,1].min():.2f}, {raw[:,1].max():.2f}], "
      f"z∈[{raw[:,2].min():.2f}, {raw[:,2].max():.2f}]")

# %%
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
t = np.linspace(0, 1, len(coords))
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=t, cmap="cool", s=2)
ax.set_title("Lorenz attractor · UMAP coords (= raw normalisé ici)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Analyses dynamiques
#
# Métriques classiques systèmes dynamiques sur trajectoire :
# - **Lyapunov** : taux divergence trajectoires proches → quantifie chaos
# - **Correlation dim** : effective dim de l'attracteur (Lorenz ≈ 2.06)
# - **RQA** : recurrence statistics (RR, DET, LAM)

# %%
from backend.dynamics import analyse_trajectory
dyn = analyse_trajectory(coords)
print(f"Lyapunov λ = {dyn['lyapunov']:.4f}")
print(f"Correlation dim = {dyn['correlation_dim']:.3f}")
print(f"ε threshold = {dyn['epsilon']:.4f}")
print(f"RQA : RR={dyn['rqa']['RR']:.3f}, DET={dyn['rqa']['DET']:.3f}, LAM={dyn['rqa']['LAM']:.3f}")

# %% [markdown]
# ### Recurrence plot

# %%
R = np.array(dyn["recurrence"])
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(R, cmap="cool", origin="lower", aspect="equal")
ax.set_title(f"Recurrence plot · ε={dyn['epsilon']:.3f}")
ax.set_xlabel("frame i"); ax.set_ylabel("frame j")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. SINDy : retrouver équations Lorenz
#
# Hypothèse : `dz/dt = f(z)` où f est sparse dans bibliothèque polynomiale.
# Validation directe : si SINDy retrouve `-10x+10y`, `28x-y-xz`, `xy-8/3z`, le pipeline fonctionne.

# %%
from backend.sindy import fit_sindy
out = fit_sindy(raw, dt=lor["dt"], order=2, threshold=0.5, var_names=["x", "y", "z"])
print(f"R² = {out['r2']:.4f}  ({out['n_active']}/{out['n_terms_lib']} terms actifs)\n")
for eq in out["equations"]:
    parts = " + ".join(f"{t['coef']:+.3f}·{t['term']}" for t in eq["terms"])
    print(f"  {eq['lhs']} = {parts}")

# %% [markdown]
# Ground truth : `dx = -10x + 10y`, `dy = 28x - y - xz`, `dz = xy - 8/3·z`
# Comparer aux coefficients retrouvés ci-dessus.

# %% [markdown]
# ## 5. Topology : persistent homology

# %%
from backend.topology import persistent_homology
ph = persistent_homology(coords[:400], max_dim=1)
print(f"H0 (composantes) : {ph['diagrams'][0]['count']} pairs")
print(f"H1 (cycles)      : {ph['diagrams'][1]['count']} pairs")
print(f"Persistence entropy : {ph['persistence_entropy']}")

# Persistence diagram
h0 = np.array(ph["diagrams"][0]["pairs"])
h1 = np.array(ph["diagrams"][1]["pairs"])
fig, ax = plt.subplots(figsize=(6, 6))
if len(h0):
    ax.scatter(h0[:, 0], h0[:, 1], c="cyan", s=20, label=f"H0 ({len(h0)})", alpha=0.6)
if len(h1):
    ax.scatter(h1[:, 0], h1[:, 1], c="magenta", s=40, label=f"H1 ({len(h1)})", alpha=0.8)
lims = [0, max(h0[:, 1].max() if len(h0) else 0, h1[:, 1].max() if len(h1) else 1)]
ax.plot(lims, lims, "--", color="gray", alpha=0.4)
ax.set_xlabel("birth"); ax.set_ylabel("death")
ax.legend(); ax.set_title("Persistence diagram")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. DNA composite score

# %%
from backend.dna import compute_dna
dna = compute_dna(raw, coords)
print(f"\n  COMPOSITE = {dna['composite_score']:.1f} / 100")
print(f"  LABEL     = {dna['label']}\n")
for k, v in dna["axes"].items():
    bar = "█" * int(v * 30)
    print(f"  {k:14s} {v:.3f}  {bar}")

# %% [markdown]
# ## 7. Comparaison Lorenz vs Rössler
#
# Deux systèmes chaotiques 3D, mais structures topologiques différentes :
# - Lorenz = 2 ailes papillon (H1 ≈ 2)
# - Rössler = 1 single funnel (H1 ≈ 1)

# %%
from backend.systems import rossler
ros = rossler(n=2000)
coords_r = np.array(ros["coords"])

dna_l = compute_dna(np.array(lor["raw_coords"]), coords)
dna_r = compute_dna(np.array(ros["raw_coords"]), coords_r)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})
for ax, c, name, score in [
    (axes[0], coords, f"Lorenz · DNA={dna_l['composite_score']:.1f}", dna_l),
    (axes[1], coords_r, f"Rössler · DNA={dna_r['composite_score']:.1f}", dna_r),
]:
    t = np.linspace(0, 1, len(c))
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=t, cmap="cool", s=2)
    ax.set_title(name)
plt.tight_layout()
plt.show()

print("\nAxis comparison :")
print(f"{'axis':<18}{'Lorenz':>10}{'Rössler':>12}")
for k in dna_l["axes"]:
    print(f"  {k:<16}{dna_l['axes'][k]:>10.3f}{dna_r['axes'][k]:>12.3f}")

# %% [markdown]
# ## 8. CLI batch usage
#
# Pour processing de plusieurs vidéos :
#
# ```bash
# python -m backend.cli batch ./videos --out-dir ./results \
#     --encoder dinov2_vits14 --analyses all
# ```
#
# Charger les résultats :

# %%
# import json
# from pathlib import Path
# results = []
# for f in Path("./results").glob("*.analysis.json"):
#     d = json.load(open(f))
#     results.append({
#         "source": d["source"],
#         "dna_score": d["analyses"]["dna"]["composite_score"],
#         "dna_label": d["analyses"]["dna"]["label"],
#         "lyapunov": d["analyses"]["dynamics"]["lyapunov"],
#         "n_clusters": d["analyses"]["clusters"]["n_clusters"],
#     })
# import pandas as pd
# df = pd.DataFrame(results).sort_values("dna_score", ascending=False)
# print(df)

# %% [markdown]
# ## 9. Suite : Web UI interactive
#
# Pour exploration interactive avec :
# - 3D trajectoire Three.js
# - Sliders params UMAP en temps réel
# - 12 onglets analyses
# - Live webcam mode
# - CLIP text query
# - Multi-video overlay
#
# ```bash
# python -m uvicorn backend.main:app --port 8770
# ```
#
# Open http://127.0.0.1:8770
