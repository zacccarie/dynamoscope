# Dynamoscope paper — arxiv submission

LaTeX source : `dynamoscope.tex` (~2300 lines, 13 sections, 36 references).

## Compile locally

Requires `pdflatex` (mactex, texlive-base, or miktex) :

```bash
brew install --cask mactex-no-gui   # macOS
# or
sudo apt install texlive-base texlive-latex-extra   # Ubuntu

cd paper
pdflatex dynamoscope.tex
pdflatex dynamoscope.tex   # second pass for cross-references
open dynamoscope.pdf
```

Expected output : ~30-40 page PDF, no broken refs, all tables render.

## Build arxiv submission tarball

```bash
./paper/build_arxiv.sh
```

Output : `paper/arxiv/dynamoscope_arxiv.tar.gz` (~35 KB).

## arXiv submission steps

1. Compile locally first to verify zero broken refs / undefined cites
2. Create account at https://arxiv.org/user
3. Start new submission at https://arxiv.org/submit
4. Categories :
   - **Primary** : `cs.LG` (Machine Learning)
   - **Cross-list** : `cs.CV` (Computer Vision), `nlin.CD` (Chaotic Dynamics)
5. Upload `paper/arxiv/dynamoscope_arxiv.tar.gz` as source
6. arXiv will compile pdflatex on their end ; review proof PDF
7. License : CC BY 4.0 recommended (open science)
8. After acceptance, edit `dynamoscope.tex` abstract to reference arxiv ID

## Paper structure (13 sections)

1. Introduction
2. Related Work
3. Methods
4. Multi-encoder integration
5. Diagnostic axes (8 axes DNA composite)
6. Differentiable Diagnostic Losses (Phase C)
7. Empirical Validation (7 subsections including negative-result cascade §7.7)
8. Application : Bifurcation Detection (4 falsifications + observable sweep)
9. Discussion (was honest limitations)
10. RegimeWorldModel Phase 1 + Phase 2 (7 subsections, MoE w/ regime conditioning)
11. What This Paper Does Not Claim (explicit boundary)
12. Limitations Summary (12 numbered items)
13. Conclusion

## Honest framing notes

The paper deliberately documents :
- 11 failure modes across §7.7, §8.1, §10
- 4 falsified hypotheses for classical-distillation
- 4 falsified hypotheses for bifurcation-detection
- Explicit `§11 What This Paper Does Not Claim` section
- 12-item `§12 Limitations Summary`

Strongest contributions :
- Methodology rigor (multi-seed, bootstrap CI, Welch t-tests)
- Honest negative results (entire §7.7 + §8.1 are negative cascades)
- Classical-observable baseline strength on dynamics-rich metrics
- Architectural validation at matched capacity for RegimeWorldModel

Weakest exposure :
- 1-step latent MSE = model's own training objective (partial circularity)
- Procedural-only video (no real action)
- 3 seeds on largest experiment
- Supervised regime labels required

## Code + data availability

All experiments reproducible from `experiments/` directory.
JSON results in `results/`. Tests in `tests/` (118/118 passing).

Repository : https://github.com/zacccarie/dynamoscope (public)
