# Notebooks

Tutorial walkthrough du pipeline.

## tutorial.py

Format jupytext `py:percent` — plain Python avec cell markers `# %%`. Git-friendly.

### Ouvrir dans VS Code

Installer extension Jupyter, ouvrir `tutorial.py`. VS Code détecte automatiquement les cellules et propose Run/Run All.

### Convertir en .ipynb

```bash
pip install jupytext
jupytext --to ipynb tutorial.py
# crée tutorial.ipynb
jupyter notebook tutorial.ipynb
```

### Run as plain Python

```bash
.venv/bin/python notebooks/tutorial.py
```
(saute les `# %%` markers, exécute tout linéairement)

## Sections du tutorial

1. Setup imports
2. Système synthétique Lorenz
3. Analyses dynamiques (Lyapunov, RQA)
4. SINDy équations recovery
5. Persistent homology
6. DNA composite score
7. Comparaison Lorenz vs Rössler
8. CLI batch usage
9. Suite Web UI

Tutorial = ~250 lignes, ~9 cellules, ~30s execution full run.
