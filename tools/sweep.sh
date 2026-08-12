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

# The scope columns are load-bearing rather than bookkeeping. The consumer ages
# the most recent `pass` per backend, so without them a narrow smoke run --
# `--target test/einsum` while debugging this script, say -- refreshes that age
# exactly as a full suite would, certifying coverage that never ran. `slow` is
# separate for the same reason: a weekday sweep must not make Sunday's slow
# coverage look current.
header_line() { printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n'; }

mkdir -p "$LOGS" || die "cannot create $LOGS"
if [ -f "$HISTORY" ]; then
  # A file written by an older schema would be silently mis-columned by the
  # consumer, which is worse than refusing to append to it.
  [ "$(head -1 "$HISTORY")" = "$(header_line)" ] ||
    die "$HISTORY has a different schema; archive it and let this run start a new one"
else
  header_line >"$HISTORY" || die "cannot write $HISTORY"
fi
# Probe once up front, so a read-only or full state filesystem is reported here
# with a clear message rather than as a run whose rows silently went nowhere.
printf '' >>"$HISTORY" || die "cannot append to $HISTORY"

# Ask git rather than inspecting `.git`'s file type: in a linked worktree -- a
# layout this project uses constantly -- `.git` is a regular file, and a -d test
# rejects a repository every later `git -C` call would have handled fine.
git -C "$MAIN" rev-parse --git-dir >/dev/null 2>&1 ||
  die "no git repository at $MAIN (set OCANNL_SWEEP_REPO)"

# An --only typo must not look like a clean sweep: without this, `--only cudaa`
# selects nothing, records nothing, and exits 0 having tested nothing.
known_backends=$(for u in "${UNITS[@]}"; do printf '%s\n' "$u" | cut -d: -f2; done)
if [ ${#ONLY[@]} -gt 0 ]; then
  for b in "${ONLY[@]}"; do
    printf '%s\n' "$known_backends" | grep -qx "$b" ||
      die "unknown backend '$b'; known: $(printf '%s' "$known_backends" | tr '\n' ' ')"
  done
fi

# One sweep at a time. Every local unit reuses a single fixed worktree, so an
# overlapping invocation -- a manual run started while the scheduled one is
# going, which is exactly how these collide -- would reset and clean that tree
# under a running dune. The earlier run's row would then describe a mixture of
# revisions: the precise failure this script exists to make impossible. A
# colliding sweep therefore refuses to start rather than queueing, so the miss is
# loud and the routine reports it, instead of two runs quietly corrupting each
# other. Per-invocation worktrees would also fix it, at the cost of the _build
# reuse that makes a daily cadence affordable.
#
# The lock is an flock on an inherited descriptor rather than a directory plus a
# pid file, so the KERNEL owns its lifetime. That removes the whole class of
# problems a hand-rolled lock has: nothing to reclaim after a crash, no window
# between creating the lock and publishing ownership, and no dependence on
# `kill -0`, which answers only "does SOME process hold this pid" -- a recycled
# pid belonging to an unrelated long-lived process would otherwise refuse every
# sweep for as long as that process lived.
#
# perl takes the lock and exits, but a lock belongs to the open file DESCRIPTION,
# which this shell still holds through fd 9; it is released when the last holder
# closes it. Children inherit fd 9 deliberately: if the sweep is killed outright
# mid-build, the orphaned dune keeps the lock, which is right -- the worktree
# really is still in use -- and the lock clears by itself when that orphan exits.
#
# It lives beside the WORKTREE, not under $STATE. OCANNL_SWEEP_STATE is a
# supported override -- it is how this script gets tested against a throwaway
# history -- and keying the lock there would split the lock namespace while
# leaving the worktree shared, so a run with its own state directory would walk
# straight past the lock and reset the tree under the scheduled sweep. A lock
# belongs in the same namespace as the resource it protects.
LOCAL_WT=$HOME/ocannl-staging-worktrees/sweep
LOCK=$LOCAL_WT.lock
# The parent will not exist on a machine that has never run a sweep, and
# bootstrapping is the one path a developer machine never exercises: without this
# the open below fails, and the failure would surface as a confusing complaint
# about another sweep rather than as a missing directory.
mkdir -p "$(dirname "$LOCK")" || die "cannot create $(dirname "$LOCK")"
exec 9>"$LOCK" || die "cannot open $LOCK"
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&9 ||
  die "another sweep is running; refusing to share the worktree"

# Signals aimed at THIS pid rather than at the process group -- `kill $pid` from a
# supervisor, or the scheduler cancelling the task -- never reach the capped
# supervisor on their own: bash defers a trap while it waits on a foreground
# command, so the tests would run to completion (or to the 90-minute cap) holding
# the lock, and every sweep queued behind them would be refused. Group signals
# reach the supervisor directly and it forwards them itself; this is the other
# half. Each unit therefore runs asynchronously and the trap relays to it.
#
# TERM rather than the signal received: bash sets SIGINT to ignored for
# asynchronous children, so relaying INT could be a no-op, while the supervisor
# installs its own TERM handler unconditionally.
UNIT_PID=
relay() {
  if [ -n "$UNIT_PID" ]; then
    kill -TERM "$UNIT_PID" 2>/dev/null
    wait "$UNIT_PID" 2>/dev/null
  fi
  exit "$1"
}
trap 'relay 130' INT
trap 'relay 143' TERM

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
#
# runtest and @slow are run UNCONDITIONALLY, their statuses combined, rather than
# chained with &&. Metal's regular operations suite is known-red, so chaining
# would mean the Sunday slow sweep never runs a single slow test on the one
# backend whose slow tests are least covered elsewhere -- while its history row
# reported only the already-known regular failure.
#
# The cap wraps the WHOLE unit -- capped() locally, one `timeout` remotely -- not
# each dune call. Per-call caps would let a --slow unit run for twice the budget
# the script advertises.
#
# A timeout in either suite still outranks a failure in the other when the two
# statuses combine. Otherwise, on a backend whose regular suite is known-red, a
# slow suite that hung would be filed forever as the already-known regular
# failure, and the distinct `timeout` verdict -- the one that says coverage was
# lost rather than merely red -- would never appear.
#
# Everything below is written with printf and single quotes so that `$?` and the
# arithmetic survive into the shell that finally runs them. The result is spliced
# into the remote string via command substitution, which bash does not rescan.
test_cmd() {
  local backend=$1 wt=$2
  # 127, not a generic failure: a worktree that is not there means nothing ran,
  # which the outcome mapping treats as non-coverage rather than a red suite.
  printf 'cd "%s" || exit 127; ' "$wt"
  printf 'OCANNL_BACKEND=%s opam exec -- dune runtest %s; rc1=$?; ' "$backend" "$TARGET"
  if [ "$SLOW" = 1 ]; then
    printf 'OCANNL_BACKEND=%s opam exec -- dune build @slow; rc2=$?; ' "$backend"
  else
    printf 'rc2=0; '
  fi
  printf 'for r in $rc1 $rc2; do case $r in 124|137|142) exit $r ;; esac; done; '
  printf 'exit $(( rc1 != 0 ? rc1 : rc2 ))'
}

# POSIX single-quoting, so a generated command can be handed to `sh -c` on the
# far side without the remote shell re-splitting or re-expanding any of it.
sq() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }

# Cap a local command, killing its whole process group when the cap expires.
# `perl -e 'alarm N; exec ...'` is not enough on its own: alarm survives exec, so
# SIGALRM reaches only the immediate child while dune and every compiler it
# spawned keep running -- holding _build locks that the NEXT unit on this machine
# (cc and metal share one worktree) would then contend with, turning one timeout
# into a cascade. Exits 142 on expiry, matching the outcome mapping below.
#
# INT, TERM and HUP are forwarded to that group and REAPED before this exits.
# Putting the child in its own group is what makes forwarding necessary: a signal
# sent to the sweep's group (a terminal Ctrl-C, the scheduler cancelling the
# task) reaches bash and this supervisor but not the child, so without this dune
# would survive while the caller's trap released the lock -- letting the next
# sweep reset the worktree under a still-running build, which is the corruption
# the lock exists to prevent.
capped_perl='
  use POSIX ();
  my $cap = shift;
  my $pid = fork();
  die "fork: $!" unless defined $pid;
  if (!$pid) { setpgrp(0, 0); exec @ARGV; exit 127 }
  my $reap = sub {
    my $code = shift;
    kill "TERM", -$pid;
    for (1 .. 50) {
      last if waitpid($pid, POSIX::WNOHANG()) > 0;
      select undef, undef, undef, 0.1;
    }
    kill "KILL", -$pid;
    waitpid($pid, 0);
    exit $code;
  };
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = sub { $reap->(129) };
  alarm $cap;
  waitpid($pid, 0);
  my $st = $?;
  alarm 0;
  exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
'
# First argument is the budget in seconds, so the remote path can allow for
# far-side cleanup and ssh teardown on top of its own cap.
capped() { perl -e "$capped_perl" -- "$@"; }

# The same supervisor, for the backgrounded unit calls. `exec` is the point:
# backgrounding a shell FUNCTION runs it in a subshell, so `capped ... &` would
# put the SUBSHELL's pid in $! -- and a relayed signal would kill that wrapper
# while the supervisor, and dune under it, carried on. exec replaces the subshell
# so $! names the supervisor itself. Valid only backgrounded: called
# synchronously it would replace this script.
capped_bg() { exec perl -e "$capped_perl" -- "$@"; }

# Put a reused worktree exactly on $full_sha, and PROVE it rather than assume it.
# `checkout --detach` is not sufficient on its own: a tracked edit that does not
# conflict with the target survives the checkout, which still exits 0 -- so the
# suite would run against a tree that is not the commit the history row names.
# `reset --hard` drops such edits and `clean -fd` drops untracked strays, while
# leaving IGNORED files alone: `_build` is ignored, and reusing it is what makes
# a daily cadence affordable. The porcelain check at the end is the proof.
#
# Emitted as shell text, and used by BOTH paths, so local and remote preparation
# cannot drift apart.
prep_cmd() {
  local repo=$1 wt=$2
  printf 'git -C "%s" worktree prune && ' "$repo"
  printf '{ git -C "%s" rev-parse --git-dir >/dev/null 2>&1 || ' "$wt"
  printf 'git -C "%s" worktree add -q --detach "%s" %s; } && ' "$repo" "$wt" "$full_sha"
  printf 'git -C "%s" checkout -q --detach %s && ' "$wt" "$full_sha"
  printf 'git -C "%s" reset -q --hard %s && ' "$wt" "$full_sha"
  printf 'git -C "%s" clean -qfd && ' "$wt"
  printf '[ -z "$(git -C "%s" status --porcelain)" ]' "$wt"
}

# Fatal on write failure: the history file is deliberately the only verdict, so
# a row that did not land is indistinguishable downstream from a unit that never
# ran. Better to abort mid-sweep, loudly, than to hand the consumer a partial
# history it will read as coverage.
record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stamp" "$1" "$2" "$run_sha" "$3" "$4" "${TARGET:-<all>}" "$SLOW" "${5:--}" >>"$HISTORY" ||
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
    # The reachability probe doubles as the way the remote home is resolved, so
    # every generated path is a literal and the whole test command can be
    # single-quoted for `sh -c`. Leaving `$HOME` in it would force double quotes
    # on the far side and leave the command open to re-expansion.
    # ConnectTimeout alone does not bound this: ssh_config(5) scopes it to
    # establishing the connection, the handshake and key exchange -- not to
    # running the remote command. A box that accepts the connection and then
    # wedges its shell would hang the whole sweep here, before any unit records
    # anything, so every ssh in this loop gets an outer bound as well.
    if ! remote_home=$(capped 60 ssh -o BatchMode=yes -o ConnectTimeout=8 \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
         "$host" 'printf %s "$HOME"' 2>/dev/null) ||
       [ -z "$remote_home" ]; then
      echo "  $machine/$backend: skip (unreachable)"
      record "$machine" "$backend" skip 0
      continue
    fi
    wt="$remote_home/ocannl-staging-worktrees/sweep"
    # rog needs the CUDA and WSL lib dirs on PATH; harmless elsewhere.
    path_prefix="export PATH=/usr/local/cuda/bin:/usr/lib/wsl/lib:\$PATH;"
    # Preparation is its own ssh round trip so that its failure -- a connection
    # dropped after the probe, a full disk, a wedged worktree -- is recorded as
    # `error`, matching the local path. Folded into the test command it would
    # have surfaced as a non-zero status in the generic branch below and been
    # written down as a FAILING SUITE, which is the opposite of the truth: a
    # remote that never got as far as dune tested nothing at all.
    remote_repo="$remote_home/ocannl-staging"
    remote_prep="git -C \"$remote_repo\" fetch -q origin master && $(prep_cmd "$remote_repo" "$wt")"
    # Bounded on BOTH sides, for the reason the test leg documents below: an
    # outer bound only kills the local ssh, and a wedged `git fetch` left running
    # on the far side can finish later and reset the shared remote worktree --
    # possibly while a subsequent sweep is building in it. The far-side timeout is
    # what actually stops that; the outer budget is the backstop for a connection
    # that dies without the remote noticing, and is larger so it cannot pre-empt
    # the inner one. Generous overall, since a cold fetch on a slow link is
    # legitimate work.
    if ! capped 900 ssh -o BatchMode=yes \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
         "$host" "timeout -k 10s 600s sh -c $(sq "$path_prefix $remote_prep")" >"$log" 2>&1; then
      echo "  $machine/$backend: error (cannot pin $host to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      fingerprint "$log" >"${log%.log}.fingerprint"
      continue
    fi
    # The cap is applied on the FAR side, where coreutils timeout exists: killing
    # the local ssh would leave the remote dune running. ONE timeout around the
    # whole unit, matching capped() locally -- a per-dune-call cap would let a
    # --slow unit run for twice the budget the script advertises.
    remote="$path_prefix timeout -k 10s ${CAP}s sh -c $(sq "$(test_cmd "$backend" "$wt")")"
    # The far-side cap does not bound the LOCAL ssh: if the connection blackholes
    # after the command starts -- the box suspends, the WiFi drops -- the remote
    # timeout may kill dune while this ssh sits waiting for a status that will
    # never arrive. OpenSSH's defaults do not rescue it (`ssh -G` reports
    # serveraliveinterval 0 and connecttimeout none), and because the unit is in
    # the foreground, the whole sweep stalls behind it: no later units, no rows.
    #
    # Keepalives detect a dead peer in ~5min, and capped() is the backstop for
    # the case where the connection is alive but the far side never returns. Its
    # budget deliberately exceeds $CAP, so it can only fire after the remote
    # timeout has had its chance plus room for cleanup and teardown; otherwise it
    # would cut legitimate long runs short and call them timeouts.
    capped_bg "$(( CAP + 300 ))" \
      ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
      "$host" "$remote" >"$log" 2>&1 &
    UNIT_PID=$!
    wait "$UNIT_PID"; rc=$?
    UNIT_PID=
  else
    wt=$LOCAL_WT
    # Checked, and fatal for this UNIT rather than the run. The worktree is
    # reused, so a checkout that fails -- a conflicting edit, a half-removed
    # worktree -- leaves the previous revision's tree on disk; running the suite
    # against it would record a pass under $run_sha for a commit that was never
    # tested.
    #
    # One error here is sticky and worth recognising: if the directory survives
    # while its administrative entry does not (`git worktree prune` reaps the
    # entry whenever the path is temporarily absent or replaced), every
    # subsequent run reports `fatal: '<path>' already exists`. Recover by hand --
    # move `_build` aside, remove the directory, `git worktree add --detach` it
    # again, move `_build` back. Deliberately not automated: deleting a
    # multi-gigabyte build tree unattended is worse than a loud repeated error.
    if ! /bin/sh -c "$(prep_cmd "$MAIN" "$wt")" >"$log" 2>&1; then
      echo "  $machine/$backend: error (cannot pin $wt to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      fingerprint "$log" >"${log%.log}.fingerprint"
      continue
    fi
    capped_bg "$CAP" /bin/sh -c "$(test_cmd "$backend" "$wt")" >"$log" 2>&1 &
    UNIT_PID=$!
    wait "$UNIT_PID"; rc=$?
    UNIT_PID=
  fi

  elapsed=$(( $(date +%s) - started ))
  # A hang, a lost connection and a red test call for different responses, so
  # keep them apart. 142 is capped()'s local expiry (128+SIGALRM); coreutils
  # `timeout` reports 124, or 137 when its -k had to escalate to SIGKILL. 137 is
  # also what an OOM kill looks like -- both mean the run was destroyed rather
  # than judged, which is the distinction the outcome is carrying.
  #
  # ssh reserves 255 for its own transport errors, so on the remote path that is
  # a connection lost mid-run: nothing was judged there either, which is `error`
  # (non-coverage) rather than a failing suite. Locally 255 is just an exit code.
  #
  # 126 and 127 mean the shell could not run what it was asked to: no opam, no
  # dune in the selected switch, no worktree to cd into. Nothing was judged, so
  # that is non-coverage, not a red suite -- a distinction that matters most on a
  # GPU box used rarely enough for its switch to rot unnoticed.
  case $rc in
    0) outcome=pass ;;
    124 | 137 | 142) outcome=timeout ;;
    126 | 127) outcome=error ;;
    255) [ -n "$host" ] && outcome=error || outcome=fail ;;
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
