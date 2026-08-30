#!/usr/bin/env bash

# Integration coverage for tools/sweep.sh's execution-accounting contract. The
# fake opam keeps the test small; real git/worktree operations exercise the
# history migration and the reused-worktree path that made cached GPU passes
# ambiguous in the first place.

set -euo pipefail

sweep=$1
aggregate=$2
verdict_probe=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-sweep-test.XXXXXX")
holder_pid=
wait_prefix=
cleanup() {
  [ -n "$wait_prefix" ] && touch "$wait_prefix.release"
  if [ -n "$holder_pid" ]; then
    kill "$holder_pid" 2>/dev/null || true
    wait "$holder_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

origin=$tmp/origin.git
main=$tmp/main
state=$tmp/state
fake_bin=$tmp/bin
calls=$tmp/opam.calls
mkdir -p "$state/logs" "$fake_bin"

git init -q --bare "$origin"
git init -q -b master "$main"
git -C "$main" config user.name sweep-test
git -C "$main" config user.email sweep-test@example.invalid
printf 'fixture\n' >"$main/fixture"
git -C "$main" add fixture
git -C "$main" commit -qm fixture
git -C "$main" remote add origin "$origin"
git -C "$main" push -q -u origin master

cat >"$fake_bin/opam" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$SWEEP_TEST_CALLS"
# Stands in for what a test run writes to the unit's log. The common output
# drives failure-fingerprint coverage; the per-backend outputs let the skip
# aggregation controls distinguish an intersection from a union without GPUs.
[ -n "${SWEEP_TEST_OPAM_OUT:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT"
case ${OCANNL_BACKEND:-} in
  cc) [ -n "${SWEEP_TEST_OPAM_OUT_CC:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT_CC" ;;
  multidev_cc)
    [ -n "${SWEEP_TEST_OPAM_OUT_MULTIDEV_CC:-}" ] &&
      printf '%s\n' "$SWEEP_TEST_OPAM_OUT_MULTIDEV_CC"
    ;;
esac
if [ -n "${SWEEP_TEST_WAIT_PREFIX:-}" ]; then
  : >"$SWEEP_TEST_WAIT_PREFIX.ready"
  waited=0
  while [ ! -e "$SWEEP_TEST_WAIT_PREFIX.release" ]; do
    sleep 0.05
    waited=$((waited + 1))
    [ "$waited" -lt 200 ] || exit 99
  done
fi
exit "${SWEEP_TEST_OPAM_RC:-0}"
EOF
chmod +x "$fake_bin/opam"

# Exercise the exact previous schema: old evidence is retained, but marked
# unknown rather than being upgraded retroactively to executed coverage.
printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n' >"$state/history.tsv"
printf '20260820T000000Z\tlocal\tcc\tdeadbee\tpass\t1\t<all>\t0\t-\n' >>"$state/history.tsv"

run_sweep_args() {
  HOME=$tmp/home \
  PATH=$fake_bin:$PATH \
  SWEEP_TEST_CALLS=$calls \
  SWEEP_TEST_WAIT_PREFIX=${SWEEP_TEST_WAIT_PREFIX:-} \
  SWEEP_TEST_OPAM_RC=${SWEEP_TEST_OPAM_RC:-0} \
  SWEEP_TEST_OPAM_OUT=${SWEEP_TEST_OPAM_OUT:-} \
  SWEEP_TEST_OPAM_OUT_CC=${SWEEP_TEST_OPAM_OUT_CC:-} \
  SWEEP_TEST_OPAM_OUT_MULTIDEV_CC=${SWEEP_TEST_OPAM_OUT_MULTIDEV_CC:-} \
  OCANNL_TOOL_SWEEP_REPO=$main \
  OCANNL_TOOL_SWEEP_STATE=$state \
    "$sweep" "$@"
}

run_sweep_backend() {
  local backend=$1
  shift
  run_sweep_args --only "$backend" "$@"
}

run_sweep() { run_sweep_backend cc "$@"; }

incremental=$(run_sweep)
forced=$(run_sweep --force)
slow_forced=$(run_sweep --slow --force)

grep -q 'local/cc: incremental-pass .*execution=incremental' <<<"$incremental"
grep -q 'local/cc: pass .*execution=forced' <<<"$forced"

expected_header='when	machine	backend	ref	outcome	seconds	target	slow	log	execution'
[ "$(head -1 "$state/history.tsv")" = "$expected_header" ]
[ "$(awk -F '\t' 'NR == 2 { print $5 ":" $10 }' "$state/history.tsv")" = 'legacy-pass:unknown' ]
[ "$(awk -F '\t' 'NR == 3 { print $5 ":" $10 }' "$state/history.tsv")" = 'incremental-pass:incremental' ]
[ "$(awk -F '\t' 'NR == 4 { print $5 ":" $10 }' "$state/history.tsv")" = 'pass:forced' ]
[ "$(awk -F '\t' 'NR == 5 { print $5 ":" $8 ":" $10 }' "$state/history.tsv")" = 'pass:1:forced' ]

# A full-suite unit builds @runtest and @train together in one dune call
# (test/training/dune says why the tier exists); only a narrow --target run
# still spells `dune runtest <target>`.
[ "$(sed -n '1p' "$calls")" = 'exec -- dune build @runtest @train' ]
[ "$(sed -n '2p' "$calls")" = 'exec -- dune clean' ]
[ "$(sed -n '3p' "$calls")" = 'exec -- dune build --force @runtest @train' ]
[ "$(sed -n '4p' "$calls")" = 'exec -- dune clean' ]
[ "$(sed -n '5p' "$calls")" = 'exec -- dune build --force @runtest @train' ]
[ "$(sed -n '6p' "$calls")" = 'exec -- dune build --force @slow' ]

# Two complete forced units expose only the INTERSECTION of their skip sets.
# The three absent backends keep this a potential finding rather than a failure:
# one of them may have evaluated the common claim. A per-backend-only marker
# must not leak into the report merely because it occurred somewhere. A skip
# whose gate belongs to the environment occurs in both logs too, but its
# explicit scope keeps it out of this backend-coverage question.
common=$'SKIPPED on fixture (vacuous): common unevaluated claim\nOCANNL_VERDICT_SKIP\tbackend\tfixture.exe\tcommon unevaluated claim'
cc_only=$'SKIPPED on fixture (vacuous): cc-only unevaluated claim\nOCANNL_VERDICT_SKIP\tbackend\tfixture.exe\tcc-only unevaluated claim'
multidev_only=$'SKIPPED on fixture (vacuous): multidev-only unevaluated claim\nOCANNL_VERDICT_SKIP\tbackend\tfixture.exe\tmultidev-only unevaluated claim'
environment=$'SKIPPED on fixture gate (vacuous): environment-gated claim\nOCANNL_VERDICT_SKIP\tenvironment\tfixture.exe\tenvironment-gated claim'
coverage=$(SWEEP_TEST_OPAM_OUT_CC="$common
$cc_only
$environment" \
  SWEEP_TEST_OPAM_OUT_MULTIDEV_CC="$common
$multidev_only
$environment" \
  run_sweep_args --force --only cc --only multidev_cc)
coverage_report=$(sed -n 's/^skip coverage: .* -- //p' <<<"$coverage" | tail -1)
[ -f "$coverage_report" ]
grep -q '^status: partial (2 of 5 known backends completed)$' "$coverage_report"
grep -q '^missing backends: metal, cuda, hip$' "$coverage_report"
grep -q '^POTENTIAL: skipped on every completed backend: fixture.exe: common unevaluated claim$' \
  "$coverage_report"
! grep -q 'cc-only unevaluated claim' "$coverage_report"
! grep -q 'multidev-only unevaluated claim' "$coverage_report"
! grep -q 'environment-gated claim' "$coverage_report"

# The pure aggregator's complete-backend control is the escalation seam the
# real sweep reaches only when both remote GPU boxes and all local units pass.
# All five logs sharing a claim is exit 1 and a FAIL line; removing it from one
# log proves the same complete census passes rather than treating a union as an
# intersection.
aggregate_args=()
for backend in cc multidev_cc metal cuda hip; do
  log=$tmp/$backend.log
  "$verdict_probe" "$backend" >"$log" 2>&1
  aggregate_args+=(--known "$backend")
  aggregate_args+=(--run "$backend" "$log")
done
set +e
complete_fail=$("$aggregate" "${aggregate_args[@]}" 2>&1)
complete_fail_rc=$?
set -e
[ "$complete_fail_rc" -eq 1 ]
grep -q '^status: complete (5 of 5 known backends completed)$' <<<"$complete_fail"
grep -q '^FAIL: skipped on every known backend: verdict_skip_probe.exe: common unevaluated claim$' \
  <<<"$complete_fail"
! grep -q 'common environment-gated claim' <<<"$complete_fail"

printf 'this backend evaluated the common claim\n' >"$tmp/hip.log"
complete_pass=$("$aggregate" "${aggregate_args[@]}")
grep -q '^result: PASS -- no claim was skipped on every known backend$' <<<"$complete_pass"

# Equal human labels in two DIFFERENT executables are different test legs. Copy
# the real probe under another basename so this control reaches the production
# identity emission rather than restating its record format in the fixture.
other_probe=$tmp/other_skip_probe.exe
cp "$verdict_probe" "$other_probe"
"$verdict_probe" cc >"$tmp/identity-cc.log" 2>&1
"$other_probe" metal >"$tmp/identity-metal.log" 2>&1
identity_clear=$("$aggregate" \
  --known cc --known metal \
  --run cc "$tmp/identity-cc.log" --run metal "$tmp/identity-metal.log")
grep -q '^result: PASS -- no claim was skipped on every known backend$' <<<"$identity_clear"

# Zero successful units is routine when every selected backend is unavailable
# or red. This runs under macOS's stock Bash 3.2 in the local suite and pins the
# nounset-safe branch before any empty-array expansion.
empty=$("$aggregate" --known cc --known metal)
grep -q '^completed backends: <none>$' <<<"$empty"
grep -q '^result: NOT AGGREGATED$' <<<"$empty"

# Evidence-processing errors are harness failures (exit 2), never an empty set
# that can read as CLEAR/PASS. Fault-inject sort, the last command of the
# extraction pipeline, so pipefail must reach the explicit error conversion.
fail_bin=$tmp/fail-bin
mkdir -p "$fail_bin"
cat >"$fail_bin/sort" <<'EOF'
#!/bin/sh
exit 7
EOF
chmod +x "$fail_bin/sort"
set +e
extract_error=$(PATH=$fail_bin:$PATH "$aggregate" \
  --known cc --known metal \
  --run cc "$tmp/identity-cc.log" --run metal "$tmp/identity-metal.log" 2>&1)
extract_error_rc=$?
set -e
[ "$extract_error_rc" -eq 2 ]
grep -q '^aggregate-skips: cannot extract compatible skip records from ' <<<"$extract_error"

# A supported `sweep.sh --ref` may target a commit from before Verdict emitted
# machine records. Its legacy human line is evidence of a skip, not evidence of
# execution; a human/machine count mismatch must make the whole log incompatible.
printf 'SKIPPED on cc (vacuous): common unevaluated claim\n' >"$tmp/legacy-cc.log"
set +e
legacy_error=$("$aggregate" \
  --known cc --known metal \
  --run cc "$tmp/legacy-cc.log" --run metal "$tmp/identity-metal.log" 2>&1)
legacy_error_rc=$?
set -e
[ "$legacy_error_rc" -eq 2 ]
grep -q '^aggregate-skips: cannot extract compatible skip records from ' <<<"$legacy_error"

# A successful analysis whose destination stops accepting bytes is still a
# harness failure. A read-only descriptor makes the first report write fail
# deterministically without relying on a device Dune's sandbox may deny; the
# explicit success exit below it must not erase that error.
: >"$tmp/read-only-report"
exec 8<"$tmp/read-only-report"
set +e
"$aggregate" "${aggregate_args[@]}" >&8 2>"$tmp/report-write.err"
report_write_rc=$?
set -e
exec 8<&-
[ "$report_write_rc" -eq 2 ]
grep -q '^aggregate-skips: cannot write report$' "$tmp/report-write.err"

# Hold one run after it owns the worktree lock, then replace its history with
# the old schema. A competing launch must refuse at the lock without migrating
# that file; this deterministically pins migration behind serialization.
wait_prefix=$tmp/migration-lock
SWEEP_TEST_WAIT_PREFIX=$wait_prefix run_sweep >"$tmp/holder.out" 2>"$tmp/holder.err" &
holder_pid=$!
for _ in {1..200}; do
  [ -e "$wait_prefix.ready" ] && break
  sleep 0.05
done
[ -e "$wait_prefix.ready" ]
printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n' >"$state/history.tsv"
printf '20260820T000000Z\tlocal\tcc\tdeadbee\tpass\t1\t<all>\t0\t-\n' >>"$state/history.tsv"
set +e
SWEEP_TEST_WAIT_PREFIX= run_sweep >"$tmp/competitor.out" 2>"$tmp/competitor.err"
competitor_rc=$?
set -e
[ "$competitor_rc" -eq 2 ]
[ "$(head -1 "$state/history.tsv")" = "$(printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog')" ]
touch "$wait_prefix.release"
wait "$holder_pid"
holder_pid=

# A red GPU unit must carry its RTC context (gh-ocannl-784): the flags the
# kernels were compiled under and which toolkit did it, beside the failure rather
# than nowhere. `metal` is the case this can pin without hardware and without a
# reachable remote -- it is a LOCAL unit, and its context block is built only from
# `command -v` guards plus the pure option-builder alias, so on a machine with no
# macOS tooling it still prints the production property sequence and proves the
# block was emitted, reached the log, and was carried into the fingerprint. The
# cuda and hip arms differ from it only in which discovery commands they guard.
#
# A red unit, not a green one: this is diagnosis, and emitting it on a pass would
# run dune a second time on every sweep.
# The failure text the fake dune writes. Its second and third lines are the shapes
# `cuda_to_ptx` appends to nvrtc's message when a compile fails -- the one
# CUDA vector and `compile_metal_source` appends to a Metal failure -- as opposed
# to the pure policy vectors the context block prints. `fingerprint` is
# backend-blind, so the local metal unit pins both lines' extraction; producing
# them is separately covered on their hardware boxes.
nvrtc_failure='Fatal error: exception nvrtc_compile_program k.cu: nvrtc: error: no
nvrtc options: -I/usr/local/cuda/include --use_fast_math
metal options: language-version=3.1 math-mode=safe math-functions=fast'
SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$nvrtc_failure \
  run_sweep_backend metal >"$tmp/metal.out" 2>&1
grep -q 'local/metal: fail' "$tmp/metal.out"
# The recorded outcome is the column the collection must not be able to corrupt.
# Folded into the unit, the diagnostic shares the unit's CAP, and a red suite that
# ran the cap down would be filed as `timeout` -- coverage lost -- instead of as
# the failure it was (Codex P2 on PR #510). The structural guarantee is that
# collection happens strictly after `record`; this pins the column it protects.
[ "$(awk -F '\t' '$3 == "metal" { print $5 }' "$state/history.tsv" | tail -1)" = fail ]
metal_log=$(awk -F '\t' '$3 == "metal" { print $9 }' "$state/history.tsv" | tail -1)
[ -n "$metal_log" ] && [ -f "$metal_log" ]
grep -q '^=== rtc-context (metal) ===$' "$metal_log"
grep -q '^=== end rtc-context ===$' "$metal_log"
grep -q 'rtc option policy from arrayjit/test/runtest-test_metal_compile_options' "$metal_log"
# The fingerprint is what a caller diffs against yesterday's, so the block has to
# reach it and not merely the log.
grep -q '^=== rtc-context (metal) ===$' "${metal_log%.log}.fingerprint"
# And so does the effective vector of a failed compile, which reaches the log as
# an ordinary line of the exception message: it begins neither at an error site
# nor at `Error`/`Fatal error`/`Exception`, so before its own selector existed it
# stopped at the log and never reached the file callers diff.
grep -q '^nvrtc options: -I/usr/local/cuda/include --use_fast_math$' \
  "${metal_log%.log}.fingerprint"
grep -q '^metal options: language-version=3.1 math-mode=safe math-functions=fast$' \
  "${metal_log%.log}.fingerprint"

# And a GREEN unit must NOT pay for it -- the same backend, so the only thing
# that differs is the outcome. The log path is derived from the sweep's timestamp
# and may well be the one above, rewritten: that is fine and is itself part of the
# check, since the assertions on the failing run have already read it.
run_sweep_backend metal >"$tmp/metal_pass.out" 2>&1
grep -q 'local/metal: incremental-pass' "$tmp/metal_pass.out"
metal_pass_log=$(awk -F '\t' '$3 == "metal" { print $9 }' "$state/history.tsv" | tail -1)
[ -f "$metal_pass_log" ]
! grep -q 'rtc-context' "$metal_pass_log"

printf 'sweep execution accounting, RTC context and skip aggregation: PASS\n'
