#!/bin/bash
# gh-ocannl-740 before/after cells: what the pad-waste economy saves a search. One driver, one box,
# every leg the closing comment quotes, so the comment names this invocation.
#
# Usage: gh740_cells.sh <out-dir> <backend> <arm-order> <site>...
#   <arm-order>  a word over {A,B}: A lifts the economy (autotune_pad_waste_max=inf, the menus as
#                they were before it), B runs at the default threshold; e.g. ABBA, so each arm is
#                timed early and late and drift shows as disagreement between its two legs.
#   <site>       name=b,m,n,k as bench_pad_waste takes it.
# Each leg is one bench_pad_waste process over every site, a cell of the shared runner
# (benchmarks/cells.py `cell`, gh-ocannl-1607): <out-dir>/<backend>-<n>-<arm>/ holds its stdout,
# its stderr (the autotune_log per-candidate lines among them) and its cell.json record. Then an
# `exit: N` line; the `padwaste:` lines are echoed. Every search starts from an empty schedule cache
# (the bench tunes with the disk cache off).
#
# Hermetic through the runner: no OCANNL_*, BENCH_* or OpenMP variable and no config file reaches a
# leg, the treatment is on argv, and the leg's resolved configuration is read back (an arm whose
# threshold did not take fails it). No fixture is read. Exits nonzero when any leg did; 125, at
# once, when a leg's process group outlived its cleanup.
set -u
[ $# -ge 4 ] || { sed -n '5,9p' "$0"; exit 2; }
OUT=$1 BACKEND=$2 ORDER=$3; shift 3
case $ORDER in *[!AB]*|'') echo "arm order must be a word over A,B: $ORDER" >&2; exit 2;; esac
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$OUT" || exit 1
OUT=$(CDPATH= cd -- "$OUT" && pwd)
cd "$ROOT" || exit 1
dune build benchmarks/runners/ocannl/bench_pad_waste.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
EXE=$ROOT/_build/default/benchmarks/runners/ocannl/bench_pad_waste.exe
rc=0 n=0
for arm in $(echo "$ORDER" | sed 's/./& /g'); do
  n=$((n + 1))
  case $arm in A) thr=inf ;; B) thr=default ;; esac
  leg="$OUT/$BACKEND-$n-$arm"
  if [ "$thr" = default ]; then
    python3 "$ROOT/benchmarks/cells.py" cell --backend "$BACKEND" --dir "$leg" -- \
      "$EXE" "$@" --ocannl_autotune_log=true
  else
    python3 "$ROOT/benchmarks/cells.py" cell --backend "$BACKEND" --dir "$leg" \
      --expect autotune_pad_waste_max="$thr" -- \
      "$EXE" "$@" --ocannl_autotune_pad_waste_max="$thr" --ocannl_autotune_log=true
  fi
  r=$?
  echo "leg $n arm $arm ($thr): exit: $r"
  [ "$r" -eq 125 ] && { echo "leg $n: a process outlived its cleanup; stopping"; exit 125; }
  grep '^padwaste:' "$leg/stdout"
  [ "$r" -ne 0 ] && rc=$r
done
exit $rc
