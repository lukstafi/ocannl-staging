#!/usr/bin/env bash
# Promote dune test outputs and normalize line endings in the promoted goldens.
#
# `dune promote` on Windows copies CRLF test output into `.expected` files
# (test exe stdout is text-mode); goldens are LF in the repo (.gitattributes).
# This wrapper promotes and then strips trailing CRs from the promoted files,
# replacing the manual `sed -i 's/\r$//'` ritual. It also works from worktrees
# nested inside the repo, where plain `dune promote` resolves the PARENT
# checkout: `dune promotion apply` accepts the `--root .` override.
#
# Usage: tools/promote.sh [FILES...]
#   Run from anywhere; extra arguments are passed through to dune (e.g. paths
#   of specific files to promote).

set -eu
cd "$(dirname "$0")/.."

command -v dune >/dev/null 2>&1 || . tools/opam-env.sh

dune promotion apply --root . "$@"

# Strip trailing CRs from any promoted golden that now differs from the index.
# perl -i, not sed -i: BSD sed (macOS) requires a backup-suffix argument for
# -i, so GNU-style `sed -i` errors there; perl is portable (and already a
# dependency of tools/test-run.sh).
git diff --name-only -z -- '*.expected' 'test/ppx/*_expected.ml' \
  | while IFS= read -r -d '' f; do
      [ -f "$f" ] && perl -i -pe 's/\r$//' "$f"
    done
