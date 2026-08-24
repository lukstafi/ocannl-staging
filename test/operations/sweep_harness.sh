#!/usr/bin/env bash

# Integration coverage for tools/sweep.sh's execution-accounting contract. The
# fake opam keeps the test small; real git/worktree operations exercise the
# history migration and the reused-worktree path that made cached GPU passes
# ambiguous in the first place.

set -euo pipefail

sweep=$1
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
if [ -n "${SWEEP_TEST_WAIT_PREFIX:-}" ]; then
  : >"$SWEEP_TEST_WAIT_PREFIX.ready"
  waited=0
  while [ ! -e "$SWEEP_TEST_WAIT_PREFIX.release" ]; do
    sleep 0.05
    waited=$((waited + 1))
    [ "$waited" -lt 200 ] || exit 99
  done
fi
exit 0
EOF
chmod +x "$fake_bin/opam"

# Exercise the exact previous schema: old evidence is retained, but marked
# unknown rather than being upgraded retroactively to executed coverage.
printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n' >"$state/history.tsv"
printf '20260820T000000Z\tlocal\tcc\tdeadbee\tpass\t1\t<all>\t0\t-\n' >>"$state/history.tsv"

run_sweep() {
  HOME=$tmp/home \
  PATH=$fake_bin:$PATH \
  SWEEP_TEST_CALLS=$calls \
  SWEEP_TEST_WAIT_PREFIX=${SWEEP_TEST_WAIT_PREFIX:-} \
  OCANNL_TOOL_SWEEP_REPO=$main \
  OCANNL_TOOL_SWEEP_STATE=$state \
    "$sweep" --only cc "$@"
}

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

[ "$(sed -n '1p' "$calls")" = 'exec -- dune runtest' ]
[ "$(sed -n '2p' "$calls")" = 'exec -- dune clean' ]
[ "$(sed -n '3p' "$calls")" = 'exec -- dune runtest --force' ]
[ "$(sed -n '4p' "$calls")" = 'exec -- dune clean' ]
[ "$(sed -n '5p' "$calls")" = 'exec -- dune runtest --force' ]
[ "$(sed -n '6p' "$calls")" = 'exec -- dune build --force @slow' ]

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

printf 'sweep execution accounting: PASS\n'
