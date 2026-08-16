#!/usr/bin/env bash
# Run dune test targets (`dune runtest ...`, `dune build @slow`, directory
# aliases) with a correct, compact, machine-readable verdict -- so that nobody,
# and especially no coding agent, hand-rolls shell around dune again.
#
# Each ingredient answers a failure mode that has actually cost a session:
#   - Unpiped status: `dune runtest 2>&1 | tail` reports TAIL's status (no
#     pipefail), so a promotion diff reads as a green run.
#   - No `status` variable anywhere: in zsh it is a read-only alias for `$?`,
#     so ad-hoc wrappers that use it die before printing their sentinel and a
#     green suite looks failed.
#   - A wall-clock cap: a hung run (macOS XProtect stalling a fresh exe, a
#     wedged backend) otherwise strands whatever is waiting on it.
#   - A verdict FILE, not a process probe: waiter loops on `pgrep -x dune`
#     match the editor's immortal `dune ocaml-merlin` daemons and spin forever
#     (one PR review accumulated ten such stranded shells); `kill -0 $pid` can
#     latch onto a recycled pid. `wait` here polls for the verdict file with a
#     hard timeout, so it cannot strand.
#
# Usage:
#   tools/test-run.sh run   [--cap N] [DUNE ARGS...]   # foreground; digest; dune's status
#   tools/test-run.sh start [--cap N] [DUNE ARGS...]   # detached; survives the session
#   tools/test-run.sh status [RUN|last]                # one-shot, never blocks
#   tools/test-run.sh wait   [RUN|last] [--timeout N]  # bounded; exits with the run's status
#   tools/test-run.sh stop   [RUN|last]                # TERM the run's process group
#   tools/test-run.sh list                             # recent runs and their states
#
# Exit codes: `run` and `wait` exit with dune's status (142 = the cap expired,
# 143/130 = cancelled, 137 = SIGKILLed, 124 = `wait` itself timed out; dune
# never reaches those on its own). `status` exits 0 finished, 3 still running,
# 1 died without a verdict. Usage and lock refusals exit 2.
#
# Everything after the options is dune's argv, verbatim (default: `runtest`):
#   tools/test-run.sh run runtest test/operations
#   tools/test-run.sh run build @slow
#   OCANNL_BACKEND=cuda tools/test-run.sh run runtest
#
# Prefer `run`. In an agent harness, launch `run` through the harness's own
# background mode and let the harness notify on exit -- that already removes
# every reason to write a waiter. `start`/`wait` exist only for a run that must
# outlive the launching session. The cap defaults to $OCANNL_TEST_CAP or 3600s;
# `--cap 0` disables it (then supply your own bound).
#
# One run at a time per worktree, enforced with an flock: a second `run`/`start`
# refuses loudly, pointing at the active run, instead of queueing behind dune's
# own lock -- "I lost track of a run so I started another" is exactly the spiral
# this script exists to prevent. `stop` the active run if it is truly stale.
#
# Windows (Git Bash) is best-effort: the flock and cap work under MSYS perl, but
# process-group kills may only reach dune itself, not its compiler children.

set -u

die() { echo "test-run: $*" >&2; exit 2; }

# Pin to the repo containing THIS script (promote.sh convention): dune then runs
# at this worktree's root no matter where the caller's cwd wandered, and the
# per-worktree lock below keys on the tree actually being tested.
cd "$(dirname "$0")/.." || die "cannot cd to repo root"
command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
command -v dune >/dev/null 2>&1 || die "dune not found (opam environment not set up?)"

RUNS=${OCANNL_TEST_RUNS:-$HOME/.ocannl-test-runs}
mkdir -p "$RUNS" || die "cannot create $RUNS"
# The worktree key makes the `last` symlink per-checkout, so concurrent
# sessions in different worktrees don't read each other's verdicts. The
# readable basename is for humans listing $RUNS; the crc of the full path is
# what keeps two paths differing only in punctuation from sharing a key.
# (The LOCK is deliberately not keyed here -- see take_lock.)
wt_key=$(basename "$PWD" | tr -c 'A-Za-z0-9' '_')$(printf %s "$PWD" | cksum | awk '{print $1}')

# Cap a command, killing its whole process group when the cap expires -- the
# sibling of sweep.sh's supervisor (see the rationale there). `perl -e 'alarm N;
# exec ...'` alone is not enough: alarm survives exec, so SIGALRM would reach
# only dune while every compiler it spawned kept running and holding _build
# locks. Exits 142 on expiry; forwards INT/TERM to the group and reaps it.
# HUP: a foreground run treats it like TERM; a detached run (OCANNL_TESTRUN_BG)
# ignores it, since surviving the launching session is its whole point.
# setpgrp is eval-guarded and the group kill falls back to a plain kill, for
# MSYS perl where process groups are shaky. The child closes fd 9 (the
# per-worktree lock) before exec: dune actions can leave background
# descendants behind, and a descendant inheriting the lock would hold it
# indefinitely after the run's verdict was already recorded. The lock's
# holders are thus exactly the script/wrapper shell and this supervisor --
# the processes whose lifetime IS the managed run. The child also records its
# pgid (only once confirmed to LEAD its own group, so a group-kill can never
# hit the caller on platforms where setpgrp failed): if this supervisor is
# SIGKILLed, the wrapper uses it to reap the surviving dune before
# publishing a verdict and releasing the lock.
capped_perl='
  use POSIX ();
  my $cap = shift;
  my $pid = fork();
  die "fork: $!" unless defined $pid;
  if (!$pid) {
    eval { setpgrp(0, 0) };
    eval {
      if ($ENV{OCANNL_TESTRUN_RD} && getpgrp(0) == $$) {
        open my $fh, ">", "$ENV{OCANNL_TESTRUN_RD}/pgid" or die;
        print $fh $$;
        close $fh;
      }
    };
    POSIX::close(9);
    exec @ARGV;
    exit 127;
  }
  my $blast = sub { my $sig = shift; kill($sig, -$pid) or kill($sig, $pid) };
  my $reap = sub {
    my $code = shift;
    $blast->("TERM");
    for (1 .. 50) {
      last if waitpid($pid, POSIX::WNOHANG()) > 0;
      select undef, undef, undef, 0.1;
    }
    $blast->("KILL");
    waitpid($pid, 0);
    exit $code;
  };
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = $ENV{OCANNL_TESTRUN_BG} ? "IGNORE" : sub { $reap->(129) };
  alarm $cap if $cap > 0;
  waitpid($pid, 0);
  my $st = $?;
  alarm 0;
  exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
'

# Take the per-worktree lock on fd 9, non-blocking. perl takes it and exits;
# the lock lives on the open file DESCRIPTION, which this shell holds through
# fd 9 and the capped supervisor inherits (dune's side closes it, see above)
# -- so it clears exactly when the run's managing processes exit, with
# nothing to reclaim after a crash (see sweep.sh).
#
# The lock file sits BESIDE the worktree it protects (a gitignored dotfile
# dune's scanner also ignores), not under $RUNS: OCANNL_TEST_RUNS is a
# supported override for diagnostics storage, and keying the lock there would
# split the lock namespace while leaving _build shared -- two sessions with
# different overrides would both "acquire" their lock and collide on dune's.
# A lock belongs in the same namespace as the resource it protects (the same
# lesson sweep.sh records).
take_lock() {
  exec 9>>"$PWD/.test-run.lock" || die "cannot open lock file"
  perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&9 || {
    echo "test-run: another test-run is active in this worktree; check it with:" >&2
    echo "  tools/test-run.sh status last" >&2
    echo "(a stale one can be stopped with: tools/test-run.sh stop last)" >&2
    exit 2
  }
}

new_run() {
  run_dir=$RUNS/$(date -u +%Y%m%dT%H%M%SZ)-$$
  mkdir "$run_dir" || die "cannot create $run_dir"
  printf '%s\n' "$*" >"$run_dir/cmd"
  printf '%s\n' "$cap" >"$run_dir/cap"
  : >"$run_dir/log"
  ln -sfn "$run_dir" "$RUNS/last-$wt_key"
  # Runs are throwaway diagnostics; reap old ones so the directory cannot grow
  # without bound. Deletion demands the full run schema -- the timestamped
  # name AND this script's metadata files -- never mere position under $RUNS,
  # because the override can point $RUNS at a directory with other tenants
  # and "has a file named exit" must not mark those for deletion. Age is the
  # VERDICT file's, since a verdict-less directory may be a live run
  # (`--cap 0` is supported and unbounded); those are reaped only on a much
  # longer leash, as crash leftovers.
  find "$RUNS" -maxdepth 1 -type d -name '2*Z-*' 2>/dev/null |
    while IFS= read -r d; do
      [ -f "$d/cmd" ] && [ -f "$d/cap" ] || continue
      if [ -f "$d/exit" ]; then
        [ -n "$(find "$d/exit" -mtime +7 2>/dev/null)" ] && rm -rf "$d"
      else
        [ -n "$(find "$d" -maxdepth 0 -mtime +30 2>/dev/null)" ] && rm -rf "$d"
      fi
    done
}

# "Is the recorded supervisor still running?" -- answered with the pid AND a
# start-time token, because a bare `kill -0` latches onto whatever process
# recycled the pid after a reboot or wrapper crash: `status`/`list` would
# report a stale run as active forever and `stop` would TERM an innocent
# process. An empty token (MSYS ps without lstart) degrades to the plain
# pid check rather than failing.
sup_alive() {
  local pid tok now
  pid=$(cat "$1/pid" 2>/dev/null) || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tok=$(tr -s ' ' <"$1/ptoken" 2>/dev/null) || tok=
  now=$(ps -o lstart= -p "$pid" 2>/dev/null | tr -s ' ')
  [ -z "$tok" ] || [ -z "$now" ] || [ "$tok" = "$now" ]
}

resolve_run() {
  local ref=${1:-last}
  if [ "$ref" = last ]; then
    run_dir=$(readlink "$RUNS/last-$wt_key" 2>/dev/null) ||
      die "no runs recorded for this worktree"
    [ -d "$run_dir" ] || die "no such run: $run_dir"
  elif [ -d "$ref" ]; then
    run_dir=$ref
  elif [ -d "$RUNS/$ref" ]; then
    # The bare identifiers `list` prints resolve here.
    run_dir=$RUNS/$ref
  else
    die "no such run: $ref (neither a directory nor an entry under $RUNS)"
  fi
}

# The compact report `run`, `wait` and `status` all end with. Fingerprint in
# the sweep.sh sense: the `File "..."` and `Error ...` lines, deduplicated, so
# a new failure is distinguishable from a standing one without opening the log.
digest() {
  local dir=$1 rc verdict fp
  rc=$(cat "$dir/exit" 2>/dev/null) || die "no verdict recorded in $dir"
  # 142 is the ONLY code the cap produces (the supervisor's SIGALRM exit), so
  # only it may say "timeout". 137 is a SIGKILL -- an OOM kill or a forced
  # external kill -- and labeling it a timeout would send triage hunting a
  # hang that never happened.
  case $rc in
    0) verdict=pass ;;
    142) verdict="TIMEOUT (cap expired; run was killed, not judged)" ;;
    137) verdict="KILLED (SIGKILL: OOM or forced kill; not judged)" ;;
    129 | 130 | 143) verdict="CANCELLED (run was killed, not judged)" ;;
    126 | 127) verdict="ERROR (toolchain/setup: nothing ran)" ;;
    *) verdict=FAIL ;;
  esac
  echo "command: dune $(cat "$dir/cmd")"
  echo "verdict: $verdict (exit $rc)"
  echo "log:     $dir/log"
  if grep -qE '^File "[^"]*\.expected"|\.corrected' "$dir/log" 2>/dev/null; then
    echo "promotion diffs present -- inspect the log, accept with \`dune promote\`" \
         "(tools/promote.sh on Windows)"
  fi
  if [ "$rc" != 0 ]; then
    fp=$({ grep -hoE '^File "[^"]+", line [0-9]+' "$dir/log"
           grep -hoE '^(Error|Fatal error|Exception)[^,]*' "$dir/log"
         } 2>/dev/null | sort -u | head -40)
    if [ -n "$fp" ]; then
      echo "fingerprint:"
      printf '%s\n' "$fp" | sed 's/^/  /'
    else
      echo "no Error/File lines matched; last 25 log lines:"
      tail -25 "$dir/log" | sed 's/^/  /'
    fi
  fi
}

finish_run() { # rc -> append sentinel, record verdict
  printf 'exit: %s\n' "$1" >>"$run_dir/log"
  # Written aside and renamed: the verdict file's EXISTENCE is the completion
  # signal `status`/`wait` key on, so it must never be observable empty.
  printf '%s\n' "$1" >"$run_dir/exit.tmp" && mv -f "$run_dir/exit.tmp" "$run_dir/exit"
}

sub=${1:-}
[ -n "$sub" ] || die "usage: tools/test-run.sh run|start|status|wait|stop|list ... (see header)"
shift

case $sub in
  run | start)
    cap=${OCANNL_TEST_CAP:-3600}
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) cap=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    # A mistyped cap would reach perl's numeric compare as 0 and silently
    # disable the alarm -- the one property this script must never lose.
    case $cap in '' | *[!0-9]*) die "--cap must be a nonnegative integer of seconds (0 disables)" ;; esac
    [ $# -gt 0 ] || set -- runtest
    take_lock
    new_run "$@"
    # ONE launch shape for both modes: a wrapper subshell owns the supervisor
    # and publishes the verdict, so no fate of the LAUNCHING shell (HUP from a
    # closed terminal, harness cancellation, plain kill) can lose it -- `run`
    # differs from `start` only in staying attached to wait and digest. The
    # wrapper inherits lock fd 9; its pid file is what `stop` signals (the
    # supervisor traps TERM and reaps the group).
    (
      # Immune to group-directed cancellation (`kill -- -PGID` from a task
      # runner reaches launcher, wrapper and supervisor alike): the
      # supervisor takes the TERM, reaps dune and exits 143, while this
      # wrapper must survive those extra seconds to publish that verdict.
      # It exits naturally right after, and KILL remains available.
      trap '' HUP TERM INT
      OCANNL_TESTRUN_BG=1 OCANNL_TESTRUN_RD=$run_dir \
        perl -e "$capped_perl" -- "$cap" dune "$@" >>"$run_dir/log" 2>&1 &
      sup=$!
      printf '%s\n' "$sup" >"$run_dir/pid"
      ps -o lstart= -p "$sup" 2>/dev/null >"$run_dir/ptoken"
      wait "$sup"
      rc=$?
      # A SIGKILLed supervisor cannot reap its process group, and dune would
      # survive it -- still mutating _build while the verdict below releases
      # the lock. The group is recorded (only when it is truly its own, see
      # the supervisor comment), so reap any survivors first.
      if pgid=$(cat "$run_dir/pgid" 2>/dev/null) && kill -0 -- "-$pgid" 2>/dev/null; then
        kill -TERM -- "-$pgid" 2>/dev/null
        sleep 2
        kill -KILL -- "-$pgid" 2>/dev/null
      fi
      finish_run "$rc"
    ) </dev/null >/dev/null 2>&1 &
    wrapper=$!
    if [ "$sub" = run ]; then
      # Attached: forward shell-directed cancellation to the supervisor, and
      # wait for the wrapper -- its exit means the verdict file is published.
      # A trapped signal returns from `wait` early, hence the retry loop.
      trap 'kill -TERM "$(cat "$run_dir/pid" 2>/dev/null)" 2>/dev/null' INT TERM HUP
      while kill -0 "$wrapper" 2>/dev/null; do wait "$wrapper"; done
      trap - INT TERM HUP
      if [ -f "$run_dir/exit" ]; then
        digest "$run_dir"
        exit "$(cat "$run_dir/exit")"
      fi
      echo "run died without recording a verdict (wrapper killed?): $run_dir"
      exit 1
    fi
    disown
    echo "started: $run_dir"
    echo "  command: dune $*"
    echo "  log:     $run_dir/log"
    echo "  check:   tools/test-run.sh status last    # from this worktree; never blocks"
    echo "  gate:    tools/test-run.sh wait last      # bounded; exits with dune's status"
    ;;
  status)
    resolve_run "${1:-last}"
    if [ -f "$run_dir/exit" ]; then
      digest "$run_dir"
    elif [ -f "$run_dir/pid" ] && ! sup_alive "$run_dir"; then
      # Grace period: the wrapper writes `exit` moments after the supervisor
      # dies; without this pause a normal completion caught mid-write would be
      # misreported as a crash.
      sleep 2
      if [ -f "$run_dir/exit" ]; then digest "$run_dir"; else
        echo "run died without recording a verdict (killed externally?): $run_dir"
        exit 1
      fi
    else
      echo "running: dune $(cat "$run_dir/cmd")  (log: $run_dir/log)"
      exit 3
    fi
    ;;
  wait)
    ref=last budget=
    while [ $# -gt 0 ]; do
      case $1 in
        --timeout) budget=$2; shift 2 ;;
        *) ref=$1; shift ;;
      esac
    done
    resolve_run "$ref"
    # Bounded by construction: default budget is the run's own cap plus slack
    # for cleanup, so a `wait` outlives a hung run only briefly -- never forever.
    cap=$(cat "$run_dir/cap" 2>/dev/null) || cap=3600
    [ -n "$budget" ] || budget=$(( cap > 0 ? cap + 120 : 7200 ))
    waited=0
    while [ ! -f "$run_dir/exit" ]; do
      # Every sleep here -- the poll AND the dead-supervisor grace -- is
      # bounded by the remaining budget, so a short --timeout is honored to
      # the second rather than rounded up to an interval.
      remaining=$(( budget - waited ))
      if [ -f "$run_dir/pid" ] && ! sup_alive "$run_dir"; then
        step=$(( remaining < 2 ? remaining : 2 )) # same mid-write grace as `status`
        [ "$step" -gt 0 ] && sleep "$step"
        [ -f "$run_dir/exit" ] && break
        echo "run died without recording a verdict (killed externally?): $run_dir"
        exit 1
      fi
      [ "$remaining" -le 0 ] && { echo "wait timed out after ${budget}s: $run_dir"; exit 124; }
      step=$(( remaining < 5 ? remaining : 5 ))
      sleep "$step"
      waited=$(( waited + step ))
    done
    digest "$run_dir"
    exit "$(cat "$run_dir/exit")"
    ;;
  stop)
    resolve_run "${1:-last}"
    [ -f "$run_dir/exit" ] && { echo "already finished:"; digest "$run_dir"; exit 0; }
    [ -f "$run_dir/pid" ] || die "no pid recorded for $run_dir"
    if sup_alive "$run_dir"; then
      kill -TERM "$(cat "$run_dir/pid")" 2>/dev/null
      echo "sent TERM; confirm with: tools/test-run.sh wait last"
    else
      echo "supervisor already gone (or its pid was recycled); nothing signaled"
    fi
    ;;
  list)
    found=0
    for d in "$RUNS"/2*/; do
      [ -d "$d" ] || continue
      found=1
      d=${d%/}
      if [ -f "$d/exit" ]; then state="exit $(cat "$d/exit")"
      elif [ -f "$d/pid" ] && sup_alive "$d"; then state=running
      else state=unknown
      fi
      printf '%s  %-8s  dune %s\n' "$(basename "$d")" "$state" "$(cat "$d/cmd" 2>/dev/null)"
    done
    [ "$found" = 1 ] || echo "no recorded runs in $RUNS"
    ;;
  *) die "unknown subcommand: $sub (run|start|status|wait|stop|list)" ;;
esac
