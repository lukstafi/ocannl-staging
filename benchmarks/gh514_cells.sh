#!/bin/bash
# gh-514 phase 6 evaluation cells, one box per invocation.
# Usage: gh514_cells.sh <backend> <precision> <repo-root> [pin-prefix...]
set -u
BK=$1; PREC=$2; ROOT=$3; shift 3
PIN=("$@")
cd "$ROOT"
opam exec -- dune build tools/fit_envelope.exe benchmarks/runners/ocannl/bench_mlp.exe \
  benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || \
  dune build tools/fit_envelope.exe benchmarks/runners/ocannl/bench_mlp.exe \
    benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null
cd benchmarks
EXE=../_build/default/benchmarks/runners/ocannl/bench_mlp.exe
GPT=../_build/default/benchmarks/runners/ocannl/bench_gpt.exe
FIT=../_build/default/tools/fit_envelope.exe
OUT="$HOME/gh514-eval-results-$BK"
rm -rf "$OUT"; mkdir -p "$OUT"

mlp() {
  local name=$1 tune=$2; shift 2
  echo "=== cell $name start $(date +%T) ==="
  BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_TUNE=$tune BENCH_TUNE_REPORT=1 \
    BENCH_PRECISION=$PREC \
    ${PIN[@]+"${PIN[@]}"} "$EXE" --ocannl_backend="$BK" --ocannl_autotune_log=true \
    --ocannl_autotune_cache_dir="$(mktemp -d)" "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "=== cell $name exit $? $(date +%T) ==="
}
gpt() {
  local name=$1; shift
  echo "=== cell $name start $(date +%T) ==="
  BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_PRECISION=$PREC \
    ${PIN[@]+"${PIN[@]}"} "$GPT" --ocannl_backend="$BK" --ocannl_autotune_log=true "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "=== cell $name exit $? $(date +%T) ==="
}

# A: tuned baseline + calibration ledger.
mlp A 1 --ocannl_autotune_calibration_file="$OUT/calib.tsv"
# Fit the envelope from A's rows; later cells pin the fitted peaks.
"$FIT" "$OUT/calib.tsv" > "$OUT/fit.txt" 2>/dev/null
PF=$(grep '^model_peak_flops=' "$OUT/fit.txt" | cut -d= -f2)
PB=$(grep '^model_peak_memory_bandwidth=' "$OUT/fit.txt" | cut -d= -f2)
PEAKS=()
[ -n "${PF:-}" ] && PEAKS+=("--ocannl_model_peak_flops=$PF")
[ -n "${PB:-}" ] && PEAKS+=("--ocannl_model_peak_memory_bandwidth=$PB")
echo "fitted peaks: ${PEAKS[*]:-none}"
# B: tuned, measured-incumbent bound pruning off vs on, fitted envelope.
mlp B-off 1 ${PEAKS[@]+"${PEAKS[@]}"}
mlp B-on 1 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_bound_pruning=true
# C: the flip chain at budget 5, legacy cost ordering vs enablement ordering, two replicates;
# then enablement + bound pruning.
for r in 1 2; do
  mlp C-cost-$r 1 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=cost
  mlp C-enab-$r 1 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=enablement
done
mlp C-enab-bp 1 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=enablement \
  --ocannl_autotune_bound_pruning=true
# D: the untuned regime — default pipeline vs model_default, +placements, +lattice.
mlp D-default 0
mlp D-model 0 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true
mlp D-model-plc 0 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true --ocannl_model_default_placements=5
mlp D-model-lat 0 ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true --ocannl_model_default_geometry_lattice=true
# D on gpt2_mini (forward-only): the bigger sites for the family/lattice ledger.
gpt Dg-default
gpt Dg-model ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true
gpt Dg-model-lat ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true --ocannl_model_default_geometry_lattice=true
echo "ALL CELLS DONE ($BK)"
