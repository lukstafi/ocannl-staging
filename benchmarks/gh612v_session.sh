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
# The exact bytes of the backported files, as measured. All three trees carry the same five copied
# files, and `lib/train.ml` here is the selector as it landed in ffc428d2 -- BEFORE the two
# review-round fixes on the PR branch, which touch failure paths (a raising `on_ship`; which arm's
# failure propagates when both arms fail) that no cell in this session took: every cell shipped a
# successful arm A. Recorded so the claim "verbatim backport" is checkable rather than asserted.
SHA_TRAIN=${SHA_TRAIN:-5d98ac818c547662608d7ef379c8ea6ecb0e5bc91be663a26c1b818c52195fcd}
SHA_GPT=${SHA_GPT:-de3cb8b4d56cac22885f014f6532a170a4b69c5ab1561f7bda7061289b548a1e}
SHA_HARNESS=${SHA_HARNESS:-f51ef550c714642594e7eb633745a60ee4bd1e6c3c94d716702aa666aaa7d2c1}
SHA_CONV=${SHA_CONV:-77dea492448dd31781ab28e34a2a29c5266af07548fb523a3d8d015253dfe0f3}
SHA_MLP=${SHA_MLP:-328c1dcc51a016eef36ad6142c9003315e8e62111c502d77da68b534eb2df44b}

# The commit each tree is pinned to in the report's Provenance. A tree carries that commit plus ONE
# commit, the gh-ocannl-638 backport -- so the pin is checked as an ancestor with a bounded distance,
# not as HEAD.
tree_pin() {
  case $1 in
    "$BASE") printf '%s\n' "${BASE_COMMIT:-6d14f401}" ;;
    "$FEAT") printf '%s\n' "${FEAT_COMMIT:-76f50dcd}" ;;
    "$MASTER") printf '%s\n' "${MASTER_COMMIT:-5d0c86d8}" ;;
    *) return 1 ;;
  esac
}

# What `gh612_cells.sh` checks per cell is the local configuration and the fixture -- the inputs. It
# does NOT check the tree, and it cannot: it takes the checkout it is handed. So a tree at the wrong
# commit, with edited sources, or with a stale binary built before the backport, is measured under
# the trusted `base574A` / `feat574A` / cap label and every gate downstream passes -- parity compares
# loss vectors, replays checks timings and cache hits, and even the arm-A premise check reads what
# the binary REPORTED. A pre-backport binary ignores the flag, ships arm B and reports "B", which
# fails loudly; a HALF-backported one (sources patched, binary stale) is the quiet case, and the
# mtime check below is what catches it.
#
# Validated here rather than in the optional `trees` block, because the measurement blocks are
# independently runnable and a printed DIRTY protects nothing. Memoized: once per tree per run.
validate_tree() {
  local t=$1 pin base extra src
  case " ${TREES_VALIDATED:-} " in *" $t "*) return 0 ;; esac
  [ -d "$t/.git" ] || [ -f "$t/.git" ] || { echo "gh612v_session: not a git checkout: $t" >&2; return 1; }
  pin=$(tree_pin "$t") || { echo "gh612v_session: tree $t is not one of BASE/FEAT/MASTER" >&2; return 1; }
  base=$(git -C "$t" rev-parse --verify --quiet "$pin^{commit}") || {
    echo "gh612v_session: $t does not contain the pinned commit $pin" >&2; return 1; }
  git -C "$t" merge-base --is-ancestor "$base" HEAD 2>/dev/null || {
    echo "gh612v_session: $t is not descended from its pinned commit $pin" >&2; return 1; }
  extra=$(git -C "$t" rev-list --count "$base..HEAD" 2>/dev/null) || extra=99
  [ "${extra:-99}" -le 1 ] || {
    echo "gh612v_session: $t carries $extra commits on top of $pin; expected at most the backport" >&2
    echo "  -- an unaccounted commit changes what the label 'the report's tree' means." >&2; return 1; }
  [ -z "$(git -C "$t" status --porcelain)" ] || {
    echo "gh612v_session: $t has uncommitted changes -- refusing to measure it" >&2; return 1; }
  # EXHAUSTIVE, then exact. The digests below pin the files the backport COPIES, but a commit that
  # also touched `arrayjit/lib/schedule.ml` would still satisfy them -- so first bound the whole
  # changed-path set to the backport's own, and only then check contents within it. Enumerated
  # rather than sampled: the previous two rounds of this check each answered "what about file X?"
  # with another named file, and a set equality answers every X at once.
  local changed expected
  expected=$(printf '%s\n' arrayjit/lib/utils.ml benchmarks/runners/ocannl/bench_conv.ml \
    benchmarks/runners/ocannl/bench_gpt.ml benchmarks/runners/ocannl/bench_harness.ml \
    benchmarks/runners/ocannl/bench_mlp.ml lib/train.ml ocannl_config.reference | sort)
  changed=$(git -C "$t" diff --name-only "$base" HEAD 2>/dev/null | sort)
  [ "$changed" = "$expected" ] || {
    echo "gh612v_session: $t's commit on top of $pin touches files the backport does not:" >&2
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$changed") | sed 's/^/    /' >&2
    echo "  Anything outside this set can change lowering, scheduling or the runners, and the cells" >&2
    echo "  would still be labelled as the report's tree." >&2; return 1; }

  # CONTENT, not shape. "One clean commit on top of the pinned base, and train.ml mentions
  # tune_ship_arm" accepts a commit that also changes lowering, scheduling or the runners -- and the
  # measurements would still be certified under the report's label. What the report actually claims
  # is that five files were copied VERBATIM, so that is what is checked: their exact digests, which
  # are identical in all three trees and equal to the selector as it landed in ffc428d2. A tree
  # carrying a different revision of those files is not the tree these numbers were measured on, and
  # says so rather than being silently measured. (The tree HEADs themselves -- ca4db3bf, e6b7b415,
  # 4d1ebb11 -- are local commits that no reproduction can match, which is why the pin is content.)
  local want got
  for src in "$SHA_TRAIN lib/train.ml" "$SHA_GPT benchmarks/runners/ocannl/bench_gpt.ml" \
             "$SHA_HARNESS benchmarks/runners/ocannl/bench_harness.ml" \
             "$SHA_CONV benchmarks/runners/ocannl/bench_conv.ml" \
             "$SHA_MLP benchmarks/runners/ocannl/bench_mlp.ml"; do
    want=${src%% *}; src=${src#* }
    got=$(sha256sum "$t/$src" 2>/dev/null | cut -d' ' -f1)
    [ "$got" = "$want" ] || {
      echo "gh612v_session: $t/$src is not the backported revision this session measured" >&2
      echo "    got  ${got:-<unreadable>}" >&2; echo "    want $want" >&2; return 1; }
  done
  # utils.ml is the ONE backported file whose base content legitimately differs per tree (the config
  # machinery moved between these commits), so it is checked by what the backport adds rather than by
  # a digest: the key registered AND classified, which is what the runs depend on. Its remaining
  # freedom is bounded by the path-set check above plus this one; `ocannl_config.reference` is
  # documentation that nothing in a cell reads, so path-set membership is all it needs.
  grep -q '"tune_ship_arm"' "$t/arrayjit/lib/utils.ml" 2>/dev/null || {
    echo "gh612v_session: $t/arrayjit/lib/utils.ml does not register tune_ship_arm" >&2; return 1; }
  [ "$(grep -c '"tune_ship_arm"' "$t/arrayjit/lib/utils.ml")" -ge 2 ] || {
    echo "gh612v_session: $t/arrayjit/lib/utils.ml registers tune_ship_arm but does not classify it" >&2
    return 1; }
  local exe="$t/_build/default/benchmarks/runners/ocannl/bench_gpt.exe"
  [ -x "$exe" ] || { echo "gh612v_session: $t has no built bench_gpt.exe -- build it first" >&2; return 1; }
  # The stale-binary case, stated as an mtime comparison because that is what is checkable here.
  for src in lib/train.ml arrayjit/lib/utils.ml benchmarks/runners/ocannl/bench_gpt.ml \
             benchmarks/runners/ocannl/bench_harness.ml; do
    [ ! "$t/$src" -nt "$exe" ] || {
      echo "gh612v_session: $t/$src is newer than the built bench_gpt.exe -- rebuild before" >&2
      echo "  measuring; a stale binary would ignore the arm selector." >&2; return 1; }
  done
  TREES_VALIDATED="${TREES_VALIDATED:-} $t"
  echo "tree ok: $t at $(git -C "$t" rev-parse --short=8 HEAD) (pinned base $pin, clean, selector present, binary current)"
}

# Unquoted on purpose: the flag string is word-split into separate arguments. It is built here, not
# taken from the caller.
run_cell() {  # <subcommand> <label> <rep> [extra driver args before the flags]
  local sub=$1 label=$2 rep=$3; shift 3
  local tree flags
  tree=$(cell_tree "$label") || return 2
  flags=$(cell_flags "$label") || return 2
  validate_tree "$tree" || return 1
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
  local t rc=0
  for t in "$BASE" "$FEAT" "$MASTER"; do
    # The same validation every cell runs, so `trees` is a dry run of the gate rather than a
    # cosmetic listing that a measurement block could then contradict.
    validate_tree "$t" || rc=1
  done
  return $rc
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
