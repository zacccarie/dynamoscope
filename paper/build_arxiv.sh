#!/usr/bin/env bash
# Build arxiv submission tarball.
# Usage : ./paper/build_arxiv.sh
#
# Output : paper/arxiv/dynamoscope_arxiv.tar.gz
#
# arxiv prefers : flat tex source + .bbl (pre-compiled bib) + figures.
# If no figures, the single .tex suffices.

set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="arxiv"
TARGET="dynamoscope_arxiv.tar.gz"

mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR/$TARGET"

# Copy LaTeX source
cp dynamoscope.tex "$BUILD_DIR/"

# If a .bbl exists (from prior compile), include it to avoid arxiv
# needing to rerun bibtex on a non-standard bib style.
if [ -f dynamoscope.bbl ]; then
  cp dynamoscope.bbl "$BUILD_DIR/"
fi

# Add ancillary README declaring contents
cat > "$BUILD_DIR/README" <<'EOF'
Dynamoscope arxiv submission
============================

Files in this submission :
- dynamoscope.tex : main LaTeX source (self-contained, uses inline
  thebibliography environment; no separate .bib file needed)

Compile locally with :
  pdflatex dynamoscope.tex
  pdflatex dynamoscope.tex   # second pass for cross-references

Code, data, all experiment scripts and JSON result files are
available at the repository accompanying this submission. See the
abstract for the repository link in the published version.
EOF

# Make the tarball (arxiv accepts .tar.gz)
cd "$BUILD_DIR"
tar czf "$TARGET" dynamoscope.tex README $([ -f dynamoscope.bbl ] && echo dynamoscope.bbl || echo "")
cd ..

echo "Built $BUILD_DIR/$TARGET"
ls -lh "$BUILD_DIR/$TARGET"

cat <<'EOF'

Next steps for arxiv submission :

1. Compile locally first to verify no broken cross-references :
     cd paper && pdflatex dynamoscope.tex && pdflatex dynamoscope.tex
   (Requires mactex or texlive-base installed.)

2. Go to https://arxiv.org/submit and create a new submission.

3. Recommended categories :
   - Primary : cs.LG (Machine Learning)
   - Cross-list : cs.CV (Computer Vision), nlin.CD (Chaotic Dynamics)

4. Upload arxiv/dynamoscope_arxiv.tar.gz as the source.

5. arxiv will compile pdflatex on its end. Review the proof PDF
   carefully before final submission.

6. License : recommended CC BY 4.0 for openness.

7. After submission, edit the dynamoscope.tex abstract to include the
   final arxiv ID once assigned (for next-version updates).
EOF
