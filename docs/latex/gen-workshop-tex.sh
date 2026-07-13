#!/bin/sh
# Regenerate docs/ocannl_workshop_article_human.tex (and the PDF under
# docs/html/pdfs/) from docs/ocannl_workshop_article_human.md.
#
# The .tex is a pure function of the .md: edit the Markdown, run this
# script, commit both. Requires pandoc; the PDF step additionally
# requires a TeX installation with latexmk (skipped if absent).
#
# Usage: docs/latex/gen-workshop-tex.sh   (from the repository root)
set -eu

cd "$(dirname "$0")/../.."

MD=docs/ocannl_workshop_article_human.md
TEX=docs/ocannl_workshop_article_human.tex
PDF_DIR=docs/html/pdfs

pandoc -s --listings --shift-heading-level-by=-1 \
  --from=markdown+tex_math_dollars+tex_math_single_backslash \
  --lua-filter=docs/latex/workshop-abstract.lua \
  -H docs/latex/workshop-preamble.tex \
  --metadata author="Łukasz Stafiniak" \
  -V fontsize=10pt -V papersize=letter -V geometry=margin=1in \
  "$MD" -o "$TEX"

# authblk's \affil must directly follow \author, but pandoc's template
# offers no hook there (include-before lands after \maketitle), so
# insert it as a post-pass.
perl -pi -e 's/^(\\author\{Łukasz Stafiniak\})$/$1\n\\affil{Independent researcher, sponsored by Ahrefs}/' "$TEX"
grep -q '\\affil{' "$TEX" || { echo "affiliation insertion failed" >&2; exit 1; }
echo "Wrote $TEX"

if command -v latexmk >/dev/null 2>&1; then
  builddir=$(mktemp -d)
  trap 'rm -rf "$builddir"' EXIT
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -output-directory="$builddir" "$TEX" >"$builddir/latexmk.log" 2>&1 || {
    echo "latexmk failed; see $builddir/latexmk.log" >&2
    trap - EXIT
    exit 1
  }
  mkdir -p "$PDF_DIR"
  cp "$builddir/$(basename "$TEX" .tex).pdf" "$PDF_DIR/"
  echo "Wrote $PDF_DIR/$(basename "$TEX" .tex).pdf"
else
  echo "latexmk not found; skipped PDF generation" >&2
fi
