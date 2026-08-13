#!/bin/bash
# gh-514 phase 6 evaluation cells, one box per invocation.
# Usage: gh514_cells.sh <backend> <precision> <repo-root> [pin-prefix...]
#
# <precision> applies to the tuned mlp cells (A/B/C) and the gpt cells; the untuned mlp D cells
# run at the same precision EXCEPT under f16, where they drop to f32 — only the f16 training
# step is loss-scale-gated (bf16 keeps Plain_step), and Bench_harness.compile_train_step's
# gated arms bypass the model_default gate, so an f16 D comparison would measure identical
# executions (see report-gh514-eval.md, the harness-gap note). Control cells pin their
# experimental gates (and the tuned cells their search enables) to the treatment's values
# explicitly: command-line settings out-rank every other config source, so an ambient profile or
# config file cannot contaminate the matrix.
set -u
BK=$1; PREC=$2; ROOT=$3; shift 3
PIN=("$@")
DPREC=$PREC
[ "$PREC" = f16 ] && DPREC=f32
FAILED=""
cd "$ROOT" || exit 1
opam exec -- dune build tools/fit_envelope.exe benchmarks/runners/ocannl/bench_mlp.exe \
  benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || \
  dune build tools/fit_envelope.exe benchmarks/runners/ocannl/bench_mlp.exe \
    benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || {
  echo "BUILD FAILED"
  exit 1
}
cd benchmarks || exit 1
EXE=../_build/default/benchmarks/runners/ocannl/bench_mlp.exe
GPT=../_build/default/benchmarks/runners/ocannl/bench_gpt.exe
FIT=../_build/default/tools/fit_envelope.exe
OUT="$HOME/gh514-eval-results-$BK"
rm -rf "$OUT"; mkdir -p "$OUT"

mlp() {
  local name=$1 tune=$2 prec=$3 st; shift 3
  echo "=== cell $name start $(date +%T) ==="
  BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_TUNE=$tune BENCH_TUNE_REPORT=1 \
    BENCH_PRECISION=$prec \
    ${PIN[@]+"${PIN[@]}"} "$EXE" --ocannl_backend="$BK" --ocannl_autotune_log=true \
    --ocannl_autotune_cache_dir="$(mktemp -d)" "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  st=$?
  echo "=== cell $name exit $st $(date +%T) ==="
  [ "$st" -eq 0 ] || FAILED="$FAILED $name"
}
gpt() {
  local name=$1 st; shift
  echo "=== cell $name start $(date +%T) ==="
  BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=0 BENCH_PRECISION=$PREC \
    ${PIN[@]+"${PIN[@]}"} "$GPT" --ocannl_backend="$BK" --ocannl_autotune_log=true "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  st=$?
  echo "=== cell $name exit $st $(date +%T) ==="
  [ "$st" -eq 0 ] || FAILED="$FAILED $name"
}

# A: tuned baseline + calibration ledger (pruning pinned off — it is the B cells' treatment).
mlp A 1 "$PREC" --ocannl_autotune_search=true --ocannl_tune_inline_flips=0 \
  --ocannl_autotune_bound_pruning=false \
  --ocannl_autotune_calibration_file="$OUT/calib.tsv"
# Fit the envelope from A's rows; later cells pin the fitted peaks.
if "$FIT" "$OUT/calib.tsv" > "$OUT/fit.txt" 2>/dev/null; then :; else FAILED="$FAILED fit"; fi
PF=$(grep '^model_peak_flops=' "$OUT/fit.txt" | cut -d= -f2)
PB=$(grep '^model_peak_memory_bandwidth=' "$OUT/fit.txt" | cut -d= -f2)
PEAKS=()
[ -n "${PF:-}" ] && PEAKS+=("--ocannl_model_peak_flops=$PF")
[ -n "${PB:-}" ] && PEAKS+=("--ocannl_model_peak_memory_bandwidth=$PB")
echo "fitted peaks: ${PEAKS[*]:-none}"
# B: tuned, measured-incumbent bound pruning off vs on, fitted envelope.
mlp B-off 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
  --ocannl_tune_inline_flips=0 --ocannl_autotune_bound_pruning=false
mlp B-on 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
  --ocannl_tune_inline_flips=0 --ocannl_autotune_bound_pruning=true
# C: the flip chain at budget 5, legacy cost ordering vs enablement ordering, two replicates;
# then enablement + bound pruning.
for r in 1 2; do
  mlp C-cost-$r 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
    --ocannl_autotune_bound_pruning=false \
    --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=cost
  mlp C-enab-$r 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
    --ocannl_autotune_bound_pruning=false \
    --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=enablement
done
mlp C-enab-bp 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
  --ocannl_tune_inline_flips=5 \
  --ocannl_tune_flip_ordering=enablement --ocannl_autotune_bound_pruning=true
# D: the untuned regime — default pipeline vs model_default, +placements, +lattice. At the
# requested precision except f16 -> f32 (see the header note); each arm pins the gates the
# treatment does not enable.
mlp D-default 0 "$DPREC" --ocannl_model_default_schedule=false
mlp D-model 0 "$DPREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=false
mlp D-model-plc 0 "$DPREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=5 --ocannl_model_default_geometry_lattice=false
mlp D-model-lat 0 "$DPREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=true
# D on gpt2_mini (forward-only, requested precision): the bigger sites for the family/lattice
# ledger.
gpt Dg-default --ocannl_model_default_schedule=false
gpt Dg-model ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=false
gpt Dg-model-lat ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=true
if [ -n "$FAILED" ]; then
  echo "CELLS FAILED:$FAILED ($BK)"
  exit 1
fi
echo "ALL CELLS DONE ($BK)"
