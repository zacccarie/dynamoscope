# Paper

`dynamoscope.tex` — arXiv-formatted paper explaining the project.

## Compile

```bash
cd paper
pdflatex dynamoscope.tex
pdflatex dynamoscope.tex   # 2nd pass for refs/toc
```

Produces `dynamoscope.pdf`.

## Structure

- Abstract
- Introduction (motivation, contributions)
- Related work (dynamical systems, TDA, self-supervised vision, world models, causality)
- Methods (pipeline, encoders, reduction, all 16 analysis modules)
- Results (Lorenz validation, encoder comparison, UMAP caveats, auto-tune, counterfactual cones, cross-modal)
- Discussion (encoder as lens, projection gaps as scale detectors, multi-representation triangulation, limitations)
- Conclusion
- 35 references

~8 pages compiled.

## arXiv submission

Category candidates : cs.CV (primary), cs.LG, math.DS (dynamical systems), nlin.CD (chaotic dynamics).

Pre-submission checklist :
- [ ] PDF compiles clean
- [ ] All citations resolved
- [ ] Figures (if added) under 10MB total
- [ ] Source code repo link inserted
- [ ] License declared
