#!/usr/bin/env bash
# Intersect Verdict.skipped announcements from complete per-backend test logs.
#
# A claim absent from one COMPLETE backend log was evaluated there; a claim
# present in every complete log was not.  The caller owns completeness -- the
# sweep passes only successful forced full-suite units, never incremental logs.
#
# Usage:
#   tools/aggregate-skips.sh \
#     --known cc --known multidev_cc --known metal --known cuda --known hip \
#     --run cc /path/to/cc.log --run metal /path/to/metal.log
#
# Exit 1 means every known backend was represented and at least one claim was
# skipped by all of them.  Partial coverage is a loud report but exits 0 because
# an absent backend may have evaluated the claim.  Malformed input exits 2.

set -uo pipefail

known=()
run_backends=()
run_logs=()

die() {
  echo "aggregate-skips: $*" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case $1 in
    --known)
      [ $# -ge 2 ] || die "--known needs a backend"
      known+=("$2")
      shift 2
      ;;
    --run)
      [ $# -ge 3 ] || die "--run needs a backend and log"
      run_backends+=("$2")
      run_logs+=("$3")
      shift 3
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ ${#known[@]} -gt 0 ] || die "no known backends supplied"

contains() {
  local wanted=$1 item
  shift
  for item in "$@"; do [ "$item" = "$wanted" ] && return 0; done
  return 1
}

for ((i = 0; i < ${#known[@]}; i++)); do
  for ((j = i + 1; j < ${#known[@]}; j++)); do
    [ "${known[$i]}" != "${known[$j]}" ] || die "duplicate known backend '${known[$i]}'"
  done
done

for ((i = 0; i < ${#run_backends[@]}; i++)); do
  backend=${run_backends[$i]}
  log=${run_logs[$i]}
  contains "$backend" "${known[@]}" || die "run names unknown backend '$backend'"
  [ -r "$log" ] || die "cannot read $backend log $log"
  for ((j = i + 1; j < ${#run_backends[@]}; j++)); do
    [ "$backend" != "${run_backends[$j]}" ] || die "duplicate run backend '$backend'"
  done
done

join_by_comma() {
  local out= item
  for item in "$@"; do
    [ -z "$out" ] || out="$out, "
    out="$out$item"
  done
  printf '%s' "$out"
}

missing=()
for backend in "${known[@]}"; do
  contains "$backend" "${run_backends[@]}" || missing+=("$backend")
done

echo "completed backends: $(join_by_comma "${run_backends[@]}")"
if [ ${#missing[@]} -eq 0 ]; then
  echo "missing backends: <none>"
else
  echo "missing backends: $(join_by_comma "${missing[@]}")"
fi

if [ ${#run_backends[@]} -lt 2 ]; then
  echo "status: insufficient (${#run_backends[@]} of ${#known[@]} known backends completed; need at least 2)"
  echo "result: NOT AGGREGATED"
  exit 0
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-skip-coverage.XXXXXX") ||
  die "cannot create temporary directory"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

extract_claims() {
  awk '
    index($0, "SKIPPED on ") == 1 {
      marker = " (vacuous): "
      at = index($0, marker)
      if (at > 0) print substr($0, at + length(marker))
    }
  ' "$1" | LC_ALL=C sort -u
}

extract_claims "${run_logs[0]}" >"$tmp/common"
for ((i = 1; i < ${#run_logs[@]}; i++)); do
  extract_claims "${run_logs[$i]}" >"$tmp/next"
  LC_ALL=C comm -12 "$tmp/common" "$tmp/next" >"$tmp/intersection"
  mv "$tmp/intersection" "$tmp/common"
done

common_count=$(wc -l <"$tmp/common" | tr -d ' ')
if [ ${#missing[@]} -eq 0 ]; then
  echo "status: complete (${#run_backends[@]} of ${#known[@]} known backends completed)"
  if [ "$common_count" -eq 0 ]; then
    echo "result: PASS -- no claim was skipped on every known backend"
    exit 0
  fi
  echo "result: FAIL -- $common_count claim(s) skipped on every known backend"
  while IFS= read -r claim; do
    printf 'FAIL: skipped on every known backend: %s\n' "$claim"
  done <"$tmp/common"
  exit 1
fi

echo "status: partial (${#run_backends[@]} of ${#known[@]} known backends completed)"
if [ "$common_count" -eq 0 ]; then
  echo "result: CLEAR across completed backends -- absent backends remain unknown"
else
  echo "result: POTENTIAL -- $common_count claim(s) skipped on every completed backend; absent backends remain unknown"
  while IFS= read -r claim; do
    printf 'POTENTIAL: skipped on every completed backend: %s\n' "$claim"
  done <"$tmp/common"
fi
