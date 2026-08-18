#!/bin/bash
# gh-ocannl-612 VERIFICATION session -- the cell sequence behind benchmarks/report-gh612-hip-verified.md.
#
# Every cell here ships the DEFAULT-PLACEMENT arm, via gh-ocannl-638's `tune_ship_arm=a`. The
# original session (report-gh612-hip.md) profiled arm A in every cell while the search shipped
# whichever arm was faster, so in three of four cells the profiled routine was never executed
# against anything -- the limitation that report states in its verdict. Forcing the arm makes the
# routine whose kernels are profiled the same routine whose losses the run reports, so the parity
# gate covers it and the pass-2 step p50 is a timing OF it.
#
# Why a script rather than a Reproduction block of hand-typed commands: a cell's treatment is a set
# of flags, and EVERY subcommand of that cell -- search, snap, replay, profile -- must repeat it
# exactly. A `replay` that omits `--ocannl_tune_ship_arm=a` still replays from the cache, still
# emits a step_ms record and still passes the driver's two-cache-hit gate, while shipping arm B: the
# pass-2 p50 would be an arm B timing under an arm A label, which is the exact confusion this
# session exists to remove. So the treatment is declared once, per cell, and every subcommand
# derives from it.
#
# Usage:  gh612v_session.sh <block>...
#   trees      resolve and print the three trees with their commits (measures nothing)
#   gh574      the gh-574 pair, 3 reps, arm order alternated: base574A vs feat574A
#   gh573      the gh-573 pair, 3 reps, alternated:           cap8A vs capoffA
#   caps       the cap-default pair, reps 4-6, alternated:    cap8A vs cap4A
#   replays    pass-2 replay (the protocol's step timings) of every cell of every block above
#   structure  snap + profile + finger on rep 1 of the four structural cells, then the diffs
#   gate       the parity gate over exactly this session's cells, and the artifact gates
#
# Environment: BASE FEAT MASTER are the three built trees (see the report's Provenance for the
# commits); OUT_ROOT defaults to /tmp/gh612v. Blocks are independent and resumable -- a rerun of a
# block re-searches its cells from cold, which is the point (each rep is an independent search).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HERE/gh612_cells.sh
export OUT_ROOT=${OUT_ROOT:-/tmp/gh612v}
BASE=${BASE:-/home/lukstafi/wt-gh612v-base}
FEAT=${FEAT:-/home/lukstafi/wt-gh612v-feat}
MASTER=${MASTER:-/home/lukstafi/wt-gh612v-master}

# The cell table. A label that is not listed is an error rather than a default: with `MASTER` as a
# fallback, a typo would measure the wrong tree under a name the report trusts.
cell_tree() {
  case $1 in
    base574A) printf '%s\n' "$BASE" ;;
    feat574A) printf '%s\n' "$FEAT" ;;
    cap8A|capoffA|cap4A) printf '%s\n' "$MASTER" ;;
    *) echo "gh612v_session: unknown cell '$1'" >&2; return 2 ;;
  esac
}
# Arm A is forced in EVERY cell, including the two that shipped it anyway in the original session:
# the point is that the treatment is uniform, so a cross-cell comparison is not conditioned on which
# arm each search happened to prefer.
cell_flags() {
  case $1 in
    base574A|feat574A|cap8A) printf '%s\n' "--ocannl_tune_ship_arm=a" ;;
    capoffA) printf '%s\n' "--ocannl_tune_ship_arm=a --ocannl_virtualize_max_inline_fanin=-1" ;;
    cap4A)   printf '%s\n' "--ocannl_tune_ship_arm=a --ocannl_virtualize_max_inline_fanin=4" ;;
    *) echo "gh612v_session: unknown cell '$1'" >&2; return 2 ;;
  esac
}
# Unquoted on purpose: the flag string is word-split into separate arguments. It is built here, not
# taken from the caller.
run_cell() {  # <subcommand> <label> <rep> [extra driver args before the flags]
  local sub=$1 label=$2 rep=$3; shift 3
  local tree flags
  tree=$(cell_tree "$label") || return 2
  flags=$(cell_flags "$label") || return 2
  echo "--- $sub $label r$rep [$flags]"
  "$D" "$sub" "$tree" "$label" "$rep" "$@" $flags
}

CELLS_GH574="base574A feat574A"
CELLS_GH573="cap8A capoffA"
CELLS_CAPS="cap8A cap4A"

# Alternate the order across reps (gh-481): without it one arm sits permanently earlier in the
# session than the other and thermal/driver drift is indistinguishable from the treatment.
balanced_block() {  # <first-rep> <reps> <label>...
  local first=$1 reps=$2; shift 2
  local labels=("$@") r i ordered
  for r in $(seq "$first" $((first + reps - 1))); do
    if [ $(( (r - first) % 2 )) -eq 0 ]; then ordered=("${labels[@]}")
    else ordered=(); for ((i=${#labels[@]}-1; i>=0; i--)); do ordered+=("${labels[$i]}"); done; fi
    echo "=== rep $r, order: ${ordered[*]} ==="
    for l in "${ordered[@]}"; do
      # Abort the block rather than report a half-balanced series: a missing cell is not a smaller
      # experiment, it is an unbalanced one.
      run_cell search "$l" "$r" || { echo "gh612v_session: $l r$r failed; aborting this block" >&2; return 1; }
    done
  done
}

block_trees() {
  local t
  for t in "$BASE" "$FEAT" "$MASTER"; do
    [ -d "$t" ] || { echo "gh612v_session: missing tree $t" >&2; return 2; }
    printf '%-40s %s %s\n' "$t" "$(git -C "$t" rev-parse --short=8 HEAD)" \
      "$([ -z "$(git -C "$t" status --porcelain)" ] && echo clean || echo DIRTY)"
  done
}

block_replays() {  # every searched cell of every block, pass 2
  local l r rc=0
  for l in base574A feat574A cap8A capoffA; do
    for r in 1 2 3; do run_cell replay "$l" "$r" || rc=1; done
  done
  for l in cap8A cap4A; do
    for r in 4 5 6; do run_cell replay "$l" "$r" || rc=1; done
  done
  return $rc
}

block_structure() {  # rep 1 of the four structural cells: emitted source, per-kernel profile, fingerprints
  local l
  for l in base574A feat574A cap8A capoffA; do
    run_cell snap "$l" 1 && run_cell profile "$l" 1 3 && "$D" finger "$l" 1 \
      || { echo "gh612v_session: structure failed for $l" >&2; return 1; }
  done
  "$D" profiles base574A/r1 feat574A/r1 cap8A/r1 capoffA/r1 || return 1
  # The mechanism diffs, on artifacts that are now output-verified on both sides.
  "$D" diff base574A 1 feat574A 1 || return 1   # gh-574: the fused lm_head+row-max kernel disappears
  "$D" diff capoffA 1 cap8A 1     || return 1   # gh-573: the ffn_b2 triangle disappears
  "$D" diff feat574A 1 capoffA 1  || return 1   # negative control: the guard is silent without a cap
}

# The premise of the whole session, and the one thing none of the driver's gates can see: `parity`
# compares loss vectors, `replays` checks timings and cache hits, and BOTH are satisfied by a
# perfectly good arm B session. A stale pre-backport binary in one worktree, a dropped flag in one
# subcommand, or a future driver that forgets to forward it, would then produce a green gate over
# exactly the arm-B artifacts this session exists to replace -- the original limitation, recreated
# and certified. So assert it directly, over BOTH passes: the searches say which arm was kept, and
# the pass-2 replays say which arm was kept AGAIN in the process whose timings are quoted.
assert_shipped_arm_A() {  # <cell>...
  local bad=0 n=0 cell kind f arm
  for cell in "$@"; do
    for kind in search.out replay2.out; do
      f="$OUT_ROOT/${cell%/*}/${cell#*/}/$kind"
      [ -s "$f" ] || { echo "gh612v_session: missing record $f" >&2; bad=1; continue; }
      # The LAST JSON line, matching what `parity` reads: a cell's stdout can carry earlier records.
      arm=$(grep -h '^{' "$f" | tail -1 | grep -o '"shipped":"[AB?]"' | grep -o '[AB?]')
      n=$((n + 1))
      [ "${arm:-?}" = "A" ] || {
        echo "gh612v_session: $cell $kind shipped arm ${arm:-?}, not A -- the profiled arm is NOT the" >&2
        echo "  executed one, which is the exact gap this session closes. Refusing to certify it." >&2
        bad=1; }
    done
  done
  [ "$n" -gt 0 ] || { echo "gh612v_session: no records to check" >&2; return 1; }
  [ "$bad" -eq 0 ] || return 1
  echo "arm-A premise: $n records (search + pass-2) all shipped arm A"
}

block_gate() {
  local want=""
  local l r
  for l in base574A feat574A cap8A capoffA; do for r in 1 2 3; do want="$want $l/r$r"; done; done
  for l in cap8A cap4A; do for r in 4 5 6; do want="$want $l/r$r"; done; done
  # FIRST, because it is the premise the other two gates presuppose and cannot test: a green parity
  # gate over arm B artifacts is a correct answer to the wrong question.
  assert_shipped_arm_A $(echo "$want") || return 1
  # EXPECT_CELLS pins the exact set, so a stale OUT_ROOT cannot substitute one cell for another, and
  # it requires a pass-2 loss vector per cell -- a timed artifact that was never output-verified is
  # exactly what this session exists to eliminate.
  EXPECT_CELLS="${want# }" "$D" parity || return 1
  "$D" replays $(echo "$want") || return 1
}

[ $# -ge 1 ] || { sed -n '/^# Usage:/,/^#   gate/p' "$0"; exit 2; }
rc=0
for block in "$@"; do
  echo "########## block: $block ($(date -u +%H:%M:%SZ))"
  case $block in
    trees)     block_trees || rc=1 ;;
    gh574)     balanced_block 1 3 $CELLS_GH574 || rc=1 ;;
    gh573)     balanced_block 1 3 $CELLS_GH573 || rc=1 ;;
    caps)      balanced_block 4 3 $CELLS_CAPS  || rc=1 ;;
    replays)   block_replays   || rc=1 ;;
    structure) block_structure || rc=1 ;;
    gate)      block_gate      || rc=1 ;;
    *) echo "gh612v_session: unknown block '$block'" >&2; rc=2 ;;
  esac
  [ $rc -eq 0 ] || { echo "gh612v_session: block '$block' failed; stopping" >&2; exit $rc; }
done
exit $rc
