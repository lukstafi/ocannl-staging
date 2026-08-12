#!/usr/bin/env bash
# Cross-machine test sweep: runs the suite once per (machine, backend) pair on
# whichever machines are reachable, and records a compact result row plus a
# failure fingerprint for each.
#
# This exists because GitHub CI covers exactly one backend: test/config's
# ocannl_config pins `backend=cc`, and the runners have no GPU. Metal, CUDA and
# HIP have no automated coverage at all without this.
#
# The GPU boxes are usually powered off, so unreachable is the NORMAL case, not
# an error -- a skipped machine is recorded as `skip` and the caller is expected
# to notice when a backend has been skipped for too long.
#
# Deliberately does NOT exit non-zero on test failures: the point is to record
# every unit's outcome, including the ones after a failing one. Only a usable
# harness failure (no local repo, etc.) aborts.
#
# Usage:
#   tools/sweep.sh                     # cc + metal locally, cuda/hip if up
#   tools/sweep.sh --slow              # also `dune build @slow`
#   tools/sweep.sh --only metal        # one backend (repeatable)
#   tools/sweep.sh --target test/einsum  # narrower dune target, for smoke-testing
#   tools/sweep.sh --ref origin/master   # what to test (default: origin/master)

set -uo pipefail

STATE=${OCANNL_SWEEP_STATE:-$HOME/.ocannl-sweep}
HISTORY=$STATE/history.tsv
LOGS=$STATE/logs
MAIN=${OCANNL_SWEEP_REPO:-$HOME/ocannl-staging}
REF=origin/master
TARGET=
SLOW=0
ONLY=()
# Per-unit wall-clock cap. macOS has no timeout(1), hence the perl alarm.
CAP=${OCANNL_SWEEP_CAP:-5400}

while [ $# -gt 0 ]; do
  case $1 in
    --slow) SLOW=1 ;;
    --only) ONLY+=("$2"); shift ;;
    --target) TARGET=$2; shift ;;
    --ref) REF=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# (machine, backend, ssh-host) -- ssh-host empty means run locally.
# The WSL sides of the GPU boxes, not the native-Windows ones: plain Linux
# toolchain, and Windows portability is covered by the scheduled CI job.
UNITS=(
  "local:cc:"
  "local:metal:"
  "rog:cuda:rog-nv-wsl"
  "minix:hip:minix-amd-wsl"
)

# Errexit is off so that a FAILING TEST does not abort the remaining units --
# that is the whole point. It must not extend to the harness: anything that
# would make an outcome unrecordable, or make a recorded outcome describe a tree
# that was not the one under test, has to be loud. A sweep that silently reports
# coverage it did not perform is worse than one that does not run.
die() { echo "sweep: $*" >&2; exit 2; }

mkdir -p "$LOGS" || die "cannot create $LOGS"
[ -f "$HISTORY" ] ||
  printf 'when\tmachine\tbackend\tref\toutcome\tseconds\tlog\n' >"$HISTORY" ||
  die "cannot write $HISTORY"
# Probe once up front, so a read-only or full state filesystem is reported here
# with a clear message rather than as a run whose rows silently went nowhere.
printf '' >>"$HISTORY" || die "cannot append to $HISTORY"

[ -d "$MAIN/.git" ] || die "no repo at $MAIN (set OCANNL_SWEEP_REPO)"

# An --only typo must not look like a clean sweep: without this, `--only cudaa`
# selects nothing, records nothing, and exits 0 having tested nothing.
known_backends=$(for u in "${UNITS[@]}"; do printf '%s\n' "$u" | cut -d: -f2; done)
if [ ${#ONLY[@]} -gt 0 ]; then
  for b in "${ONLY[@]}"; do
    printf '%s\n' "$known_backends" | grep -qx "$b" ||
      die "unknown backend '$b'; known: $(printf '%s' "$known_backends" | tr '\n' ' ')"
  done
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
# Resolve the ref to a commit ONCE, here, and pin every machine to that commit.
# Letting each box resolve `origin/master` itself would have them testing
# different commits whenever a merge lands mid-sweep, which is exactly the
# ambiguity a sweep exists to remove. It does mean --ref must name something
# reachable from origin/master, since that is all the remotes fetch.
#
# The fetch is checked: an unchecked one that fails transiently still leaves
# `origin/master` resolvable at whatever it pointed to last time, so the sweep
# would pin every machine to a stale commit and record green coverage for a tree
# nobody asked about -- the exact silent non-coverage this script exists to make
# impossible.
git -C "$MAIN" fetch -q origin master || die "cannot fetch origin master in $MAIN"
full_sha=$(git -C "$MAIN" rev-parse "$REF" 2>/dev/null) ||
  die "cannot resolve $REF in $MAIN"
run_sha=$(git -C "$MAIN" rev-parse --short "$full_sha")

wanted() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local b
  for b in "${ONLY[@]}"; do [ "$b" = "$1" ] && return 0; done
  return 1
}

# The dune invocation, shared by the local and remote paths so the two cannot
# drift. Unpiped inside the shell that runs it: piping dune to anything reports
# the pipe's status, not dune's, and a promotion diff then reads as green.
dune_cmd() {
  local backend=$1 wt=$2
  # Double quotes, not single: on the remote path $wt is the literal string
  # "$HOME/..." and the remote shell has to expand it.
  local -a cmd=("cd \"$wt\" &&" "OCANNL_BACKEND=$backend opam exec -- dune runtest ${TARGET}")
  [ "$SLOW" = 1 ] && cmd+=("&& OCANNL_BACKEND=$backend opam exec -- dune build @slow")
  printf '%s ' "${cmd[@]}"
}

# Fatal on write failure: the history file is deliberately the only verdict, so
# a row that did not land is indistinguishable downstream from a unit that never
# ran. Better to abort mid-sweep, loudly, than to hand the consumer a partial
# history it will read as coverage.
record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stamp" "$1" "$2" "$run_sha" "$3" "$4" "${5:--}" >>"$HISTORY" ||
    die "cannot record $1/$2 outcome in $HISTORY"
}

# A compact, diffable summary of what went wrong, so a caller can tell a NEW
# failure from a standing one. Metal's operations suite carries known-red tests,
# and a sweep that shouts on every red is a sweep nobody reads.
fingerprint() {
  {
    grep -hoE '^File "[^"]+", line [0-9]+' "$1"
    grep -hoE '^(Error|Fatal error|Exception)[^,]*' "$1"
  } 2>/dev/null | sort -u | head -60
}

echo "sweep $stamp  ref=$REF ($run_sha)  slow=$SLOW  target=${TARGET:-<all>}"
echo

for unit in "${UNITS[@]}"; do
  IFS=: read -r machine backend host <<<"$unit"
  wanted "$backend" || continue

  log=$LOGS/$stamp-$machine-$backend.log
  started=$(date +%s)

  if [ -n "$host" ]; then
    if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" true >/dev/null 2>&1; then
      echo "  $machine/$backend: skip (unreachable)"
      record "$machine" "$backend" skip 0
      continue
    fi
    wt="\$HOME/ocannl-staging-worktrees/sweep"
    remote="cd ~/ocannl-staging && git fetch -q origin master && git worktree prune &&
      { git -C $wt rev-parse --git-dir >/dev/null 2>&1 ||
        git worktree add -q --detach $wt $full_sha; } &&
      git -C $wt checkout -q --detach $full_sha && $(dune_cmd "$backend" "$wt")"
    # rog needs the CUDA and WSL lib dirs on PATH; harmless elsewhere.
    remote="export PATH=/usr/local/cuda/bin:/usr/lib/wsl/lib:\$PATH; $remote"
    perl -e 'alarm shift; exec @ARGV' "$CAP" \
      ssh -o BatchMode=yes "$host" "$remote" >"$log" 2>&1
    rc=$?
  else
    wt=$HOME/ocannl-staging-worktrees/sweep
    # Checked, and fatal for this UNIT rather than the run. The worktree is
    # reused, so a checkout that fails -- a conflicting edit, a half-removed
    # worktree -- leaves the previous revision's tree on disk; running the suite
    # against it would record a pass under $run_sha for a commit that was never
    # tested. The remote path needs no equivalent because its prep is already
    # chained with && ahead of dune, so a prep failure cannot reach the tests.
    #
    # One error here is sticky and worth recognising: if the directory survives
    # while its administrative entry does not (`git worktree prune` reaps the
    # entry whenever the path is temporarily absent or replaced), every
    # subsequent run reports `fatal: '<path>' already exists`. Recover by hand --
    # move `_build` aside, remove the directory, `git worktree add --detach` it
    # again, move `_build` back. Deliberately not automated: deleting a
    # multi-gigabyte build tree unattended is worse than a loud repeated error.
    if ! { git -C "$MAIN" worktree prune &&
           { git -C "$wt" rev-parse --git-dir >/dev/null 2>&1 ||
             git -C "$MAIN" worktree add -q --detach "$wt" "$full_sha"; } &&
           git -C "$wt" checkout -q --detach "$full_sha"; } >"$log" 2>&1; then
      echo "  $machine/$backend: error (cannot pin $wt to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      fingerprint "$log" >"${log%.log}.fingerprint"
      continue
    fi
    perl -e 'alarm shift; exec @ARGV' "$CAP" \
      /bin/sh -c "$(dune_cmd "$backend" "$wt")" >"$log" 2>&1
    rc=$?
  fi

  elapsed=$(( $(date +%s) - started ))
  # perl's alarm kills with SIGALRM (142); distinguish it from a test failure,
  # since a hang and a red test call for different responses.
  case $rc in
    0) outcome=pass ;;
    142) outcome=timeout ;;
    *) outcome=fail ;;
  esac
  echo "  $machine/$backend: $outcome (${elapsed}s)"
  record "$machine" "$backend" "$outcome" "$elapsed" "$log"
  if [ "$outcome" != pass ]; then
    fingerprint "$log" >"${log%.log}.fingerprint"
  fi
done

echo
echo "history: $HISTORY"
echo "logs:    $LOGS/$stamp-*"
