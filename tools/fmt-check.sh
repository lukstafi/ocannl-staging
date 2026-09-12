#!/usr/bin/env bash

# Run the formatter gate while also rejecting the invalid-odoc warnings that
# ocamlformat reports without changing its successful exit status.

set -uo pipefail

if [ "$#" -eq 0 ]; then
  set -- opam exec -- dune build @fmt
fi

fmt_log=$(mktemp "${TMPDIR:-/tmp}/ocannl-fmt-check.XXXXXX") || exit 2
trap 'rm -f "$fmt_log"' EXIT

"$@" >"$fmt_log" 2>&1
fmt_status=$?
if ! cat "$fmt_log"; then
  echo "fmt-check: could not replay formatter output" >&2
  exit 2
fi

# A real formatter failure owns the verdict, including statuses other than 1.
if [ "$fmt_status" -ne 0 ]; then
  exit "$fmt_status"
fi

if grep -Fq "Warning: Invalid documentation comment:" "$fmt_log"; then
  echo "fmt-check: invalid documentation comments are forbidden" >&2
  exit 1
fi
