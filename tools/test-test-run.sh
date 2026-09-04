#!/usr/bin/env bash
# Tests for the shared `group_alive` used by tools/test-run.sh -- the predicate
# its two callers ask "is anything in this process group still running?" --
# and for the two sentences `stop` prints about a surviving
# process group, which are what that predicate is for (gh-ocannl-742).
#
#   tools/test-test-run.sh          # run every leg
#   tools/test-test-run.sh --keep   # keep the temp dir for inspection
#
# It is the sibling of scripts/test-setup-ocaml-env.sh, whose leg 1 (f) tests
# the same predicate in the SessionStart hook. Neither is a dune test: they
# spawn, STOP and kill process groups, which is a poor fit for `dune runtest`.
# The Ubuntu CI leg runs both directly so Linux decides the kernel-dependent
# zombie-group control (gh-ocannl-795).
#
# It tests the WORKING-TREE copy: `group_alive` is extracted from the shared
# scripts/process-group.sh, `ps_token` from tools/test-run.sh, and the `stop`
# legs drive that same tool as a subprocess. Each extraction is asserted
# structurally before use, so a sed that matched nothing cannot leave every leg
# passing without testing anything.
#
# Legs:
#   1. extraction -- the functions came out of the shipping script.
#   2. a genuinely live group reads ALIVE (the other side of leg 4: without
#      this, a group_alive that answered "dead" always would pass leg 4).
#   3. a group with no members at all reads DEAD, and so does the bare signal.
#   4. a group holding nothing but a ZOMBIE reads dead, where the bare
#      `kill -0 -- -PGID` this replaced reads it as alive. That misreading is
#      what made `stop` able to announce an orphaned group holding only corpses.
#   5. a pgid that is not a positive decimal integer is refused -- 0 and
#      negatives are kill specials (caller's own group, broadcast).
#   6. `stop` on a group whose leader IGNORES TERM says so and escalates --
#      and the escalation kills the whole group. Every fixture group holds two
#      processes, or a leader-only kill would pass for a group kill.
#   7. `stop` on a group whose leader exits on TERM says the TERM went out and
#      asks for a re-run, rather than claiming the group ignored it.
#   8. `stop` on a reachable group that holds no running member says exactly
#      that -- the sentence gh-ocannl-742 added, and the one whose absence let
#      a group of corpses be reported as a runaway dune ignoring TERM.
#   9. `repeat` forces and preserves three identical dune runs while holding the
#      worktree lock for every iteration.
#  10. stdout drift is a distinct red result, with pairwise diff artifacts.
#  11. stderr-only drift is reported separately and remains a green diagnostic.
#  12. a red dune iteration keeps its nonzero exit code even when repeatable.
#  13. `--alone` serializes dune with `-j 1` on every iteration.
#  14. an active repeat is `last`, and `stop last` cancels the whole set after
#      the current iteration rather than launching the remaining ones.

set -u

KEEP=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    # The whole leading comment block, however long it grows: a pinned line
    # range silently truncates --help the first time a leg is added.
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-test-run.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/test-run.sh"
GROUP_SRC="$HERE/../scripts/process-group.sh"
HOOK_SRC="$HERE/../scripts/setup-ocaml-env.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
[ -f "$GROUP_SRC" ] || { echo "no $GROUP_SRC" >&2; exit 2; }
[ -f "$HOOK_SRC" ] || { echo "no $HOOK_SRC" >&2; exit 2; }

failures=0
report() { # report RC LABEL [DETAIL]
  if [ "$1" -eq 0 ]; then
    printf 'PASS  %s\n' "$2"
  else
    failures=$((failures + 1))
    printf 'FAIL  %s\n' "$2"
    [ $# -ge 3 ] && printf '      %s\n' "$3"
  fi
  return 0
}
skipped=0
skip() { # skip LABEL REASON -- a leg this system cannot decide, not a failure
  skipped=$((skipped + 1))
  printf 'SKIP  %s\n      %s\n' "$1" "$2"
  return 0
}

# The harness needs the two facts about a process that `group_alive` needs, read
# INDEPENDENTLY of it: its state and its process group. Neither is available the
# same way everywhere -- a Git Bash/MSYS `ps` takes no `-o` at all, which the
# Windows CI job depends on and which would otherwise leave every state probe
# here reading "gone". So: /proc where it answers (MSYS has one), `ps` where it
# does not, and the prerequisite check below refuses to let a leg run on a
# system where neither does -- an unreadable state must skip a leg, never pass
# or fail one (Codex review round 1, P2).
pstate() { # <pid> -> its one-letter state, empty where this system will not say
  local line
  if [ -r "/proc/$1/stat" ]; then
    # Grouped, not `read ... 2>/dev/null`: the shell reports a failed
    # redirection before the command's own stderr redirection applies, and
    # these readers are asked about pids that are expected to be gone.
    { read -r line <"/proc/$1/stat"; } 2>/dev/null || return 0
    line=${line##*) }               # comm may itself hold ") "
    # shellcheck disable=SC2086
    set -- $line                    # `state ppid pgrp ...`
    printf '%s' "${1:-}"
    return 0
  fi
  ps -o state= -p "$1" 2>/dev/null | tr -d ' ' | cut -c1
}
ppgid() { # <pid> -> its process group, empty where this system will not say
  local line
  if [ -r "/proc/$1/stat" ]; then
    { read -r line <"/proc/$1/stat"; } 2>/dev/null || return 0
    line=${line##*) }
    # shellcheck disable=SC2086
    set -- $line
    printf '%s' "${3:-}"
    return 0
  fi
  ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}
# Probed against a process known to be alive and to have a group -- this one.
# An empty answer here means the reader is absent, which is a different fact
# from a process being gone, and the two are indistinguishable at a leg.
have_state=1; [ -n "$(pstate $$)" ] || have_state=0
have_pgid=1;  [ -n "$(ppgid $$)" ]  || have_pgid=0

echo "testing $SRC and $GROUP_SRC"
printf '  digest %s\n' "$( (cksum <"$SRC") 2>/dev/null || echo '?')"
printf '  group digest %s\n' "$( (cksum <"$GROUP_SRC") 2>/dev/null || echo '?')"
printf '  state reader: %s; pgid reader: %s\n' \
  "$([ "$have_state" = 1 ] && echo present || echo ABSENT)" \
  "$([ "$have_pgid" = 1 ] && echo present || echo ABSENT)"

# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails
# would leave TMP empty and `rm -rf "$TMP"` would be handed the ROOT.
zparent=""   # leg 4's self-stopping zombie maker; cleanup must resume it
livepid=""   # leg 2's live group leader
leader=""    # the stop legs' current group leader
member=""    # and the second process it put in that group
member_token=""  # ... and its start token, since it is not this shell's child
repeat_pid="" # leg 14's active repeat coordinator
TMP="$(mktemp -d "${TMPDIR:-/tmp}/test-run-test.XXXXXX" 2>/dev/null)" || TMP=""
if [ -z "$TMP" ] || [ ! -d "$TMP" ]; then
  echo "could not create a temporary directory under ${TMPDIR:-/tmp}" >&2
  exit 2
fi
cleanup() {
  # Leg 4's zombie maker STOPS ITSELF and is resumed at the end of the leg.
  # Interrupted in between, nothing else would ever resume it: it would be
  # reparented to PID 1 still stopped, still holding its zombie child. Killing
  # it is not the answer either -- that orphans the zombie onto a PID 1 that, in
  # the very environment this leg is about, never reaps. Resume it so it reaps
  # its own child, and only insist if it will not go.
  local waited
  if [ -n "${zparent:-}" ] && kill -0 "$zparent" 2>/dev/null; then
    kill -CONT "$zparent" 2>/dev/null
    for waited in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$zparent" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$zparent" 2>/dev/null
    wait "$zparent" 2>/dev/null
  fi
  if [ -n "${livepid:-}" ] && kill -0 "$livepid" 2>/dev/null; then
    kill -KILL -- "-$livepid" 2>/dev/null
    kill -KILL "$livepid" 2>/dev/null
    wait "$livepid" 2>/dev/null
  fi
  # The stop legs' leader ignores TERM by design, so only KILL removes it --
  # and the group with it, since a leg interrupted before its `stop` leaves the
  # whole fixture group behind.
  if [ -n "${leader:-}" ] && kill -0 "$leader" 2>/dev/null; then
    kill -KILL -- "-$leader" 2>/dev/null
    kill -KILL "$leader" 2>/dev/null
    wait "$leader" 2>/dev/null
  fi
  # The group's second member is not this shell's child, so it is named
  # directly as well: a group kill that failed is exactly the case where it
  # would otherwise be left behind. Identity-checked -- see kill_member.
  if [ -n "${member:-}" ]; then kill_member; fi
  if [ -n "${repeat_pid:-}" ] && kill -0 "$repeat_pid" 2>/dev/null; then
    kill -TERM "$repeat_pid" 2>/dev/null
    wait "$repeat_pid" 2>/dev/null
  fi
  if [ "$KEEP" = 1 ]; then
    echo "kept $TMP"
  elif [ -n "$TMP" ] && [ -d "$TMP" ] && [ "$TMP" != "/" ]; then
    rm -rf "$TMP"
  fi
  return 0
}
trap cleanup EXIT
# Without these, a TERM or a Ctrl-C kills the shell outright and the EXIT trap
# never runs -- which is how an interrupted run could leave a stopped zombie
# maker behind, or a fixture process group outlive the harness that forked it.
# Exiting from the handler is what gets EXIT to fire.
#
# The INT arm covers the way this is actually interrupted, a Ctrl-C at a
# terminal. It is inert when the harness is itself a BACKGROUND job of a
# non-interactive shell -- such a child inherits SIGINT ignored, and a signal
# ignored on entry cannot be re-trapped -- so a scripted test of the cleanup
# path has to signal TERM to see anything happen.
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# Leg 1: extraction
# ---------------------------------------------------------------------------
# Checked structurally -- opens with the header, closes with the brace, has a
# body -- rather than by grepping for one line of it: this guard must still hold
# while the function under test is being mutated to see a leg go red. The header
# is matched as a PREFIX, since it carries a trailing comment.
#
# `ps_token` is extracted alongside `group_alive` because the stop legs below
# have to FORGE a run directory's leader token, and a token spelled differently
# from the one the shipping script recomputes would leave every one of them
# falling through to "nothing left to signal" -- passing no leg, but testing
# neither sentence either.
for fn in group_alive ps_token; do
  if [ "$fn" = group_alive ]; then fn_src="$GROUP_SRC"; else fn_src="$SRC"; fi
  sed -n "/^$fn() {/,/^}/p" "$fn_src" >"$TMP/$fn.sh"
  g_lines="$(wc -l <"$TMP/$fn.sh" | tr -d ' ')"
  g_head="$(head -n1 "$TMP/$fn.sh")"
  case $g_head in "$fn() {"*) g_ok=1 ;; *) g_ok=0 ;; esac
  if [ "$g_ok" = 0 ] \
     || [ "$(tail -n1 "$TMP/$fn.sh")" != "}" ] || [ "$g_lines" -lt 10 ]; then
    report 1 "$fn: extracted" "sed did not capture the function body from $fn_src"
  else
    report 0 "$fn: extracted ($g_lines lines)"
    # shellcheck disable=SC1090
    . "$TMP/$fn.sh"
  fi
done

# The decision behind this issue is one definition, not two copies tested in
# parallel. Count definitions across the shared helper and both production
# callers so either copy coming back fails the harness that CI runs.
group_definition_count() { awk '/^group_alive\(\)/ { n++ } END { print n + 0 }' "$@"; }
one_group_definition() { [ "$(group_definition_count "$@")" = 1 ]; }
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC"; then
  report 0 "group_alive has one shared production definition"
else
  report 1 "group_alive has one shared production definition" \
    "found $(group_definition_count "$GROUP_SRC" "$SRC" "$HOOK_SRC") definitions across $GROUP_SRC, $SRC and $HOOK_SRC"
fi
# Synthetic duplicate: the clean production tree alone cannot distinguish a
# working uniqueness check from one that always answers yes.
cp "$GROUP_SRC" "$TMP/duplicate-process-group.sh"
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC" "$TMP/duplicate-process-group.sh"; then
  report 1 "the shared-definition check rejects a duplicate" \
    "the production definition plus a copied definition still read as unique"
else
  report 0 "the shared-definition check rejects a duplicate"
fi

# ---------------------------------------------------------------------------
# Leg 2: a live group reads alive
# ---------------------------------------------------------------------------
# `set -m` puts the child in a process group of its own, so its pid IS a pgid
# holding exactly one live process -- the shape all four call sites signal.
# That the child really leads its own group has to be CHECKED (a shell whose
# setpgrp does not take would otherwise have this leg quietly testing the
# harness's own group), so both legs need the pgid reader.
live_label="a group holding a running process reads as alive"
empty_label="a group with no members left reads as dead"
if [ "$have_pgid" = 0 ]; then
  skip "$live_label" "no way to read a process's group on this system"
  skip "$empty_label" "no way to read a process's group on this system"
else
  set -m
  sleep 30 >/dev/null 2>&1 </dev/null &
  livepid=$!
  set +m
  lpgid="$(ppgid "$livepid")"
  if [ "$lpgid" != "$livepid" ]; then
    report 1 "$live_label" \
      "the child did not lead its own group (pgid '${lpgid:-gone}' vs pid $livepid)"
  elif ! kill -0 -- "-$livepid" 2>/dev/null; then
    report 1 "$live_label" "the group is not even signal-reachable; the leg tested nothing"
  elif group_alive "$livepid"; then
    report 0 "$live_label"
  else
    report 1 "$live_label" "group_alive said dead for a group with a live sleep in it"
  fi

  # -------------------------------------------------------------------------
  # Leg 3: an empty group reads dead
  # -------------------------------------------------------------------------
  kill -KILL -- "-$livepid" 2>/dev/null
  wait "$livepid" 2>/dev/null   # reaped here, so the group really is empty
  livepid=""
  if kill -0 -- "-$lpgid" 2>/dev/null; then
    report 1 "$empty_label" "pgid $lpgid is still signal-reachable after the kill and reap"
  elif group_alive "$lpgid"; then
    report 1 "$empty_label" "group_alive said alive for an empty group"
  else
    report 0 "$empty_label"
  fi
fi

# ---------------------------------------------------------------------------
# Leg 4: a zombie-only group reads dead
# ---------------------------------------------------------------------------
# The negative control for the whole change: `kill -0` must still say YES here,
# or the leg would pass without exercising the difference. Under an init that
# reaps, such a corpse is transient and the bare probe merely lost a race with
# it; under one that does not -- the ordinary container case -- it is PERMANENT,
# so waiting it out was never the fix.
#
# Making a zombie that is reliably still a zombie when looked at, and that leaves
# nothing behind afterwards, takes care. The parent puts the child in a process
# group of ITS OWN (so the group holds the zombie and nothing else), then STOPs
# itself before the child exits: a stopped shell runs no SIGCHLD handler, so it
# cannot reap, and the child stays a zombie for as long as the leg needs.
#
# All of that needs a state reader, and needs it INDEPENDENT of the function
# under test: without one, "is it a zombie yet" reads the same as "it is gone",
# the leg would sit out its whole retry budget and then judge `group_alive` on a
# premise it never established -- and the cleanup assertion at the end would
# read an empty state as "reaped" and pass on a system that cannot see the
# corpse at all. So the whole leg, zombie maker included, is skipped there.
zlabel="a group holding nothing but a zombie reads as dead"
clabel="the state reader alone rejects a zombie-only group (signal probe forced to say alive)"
rlabel="the zombie leg reaps its own zombie, leaving no process-table entry"
if [ "$have_state" = 0 ]; then
  skip "$zlabel" "no way to read a process's state on this system"
  skip "$clabel" "no way to read a process's state on this system"
  skip "$rlabel" "the zombie leg did not run, so it left nothing to reap"
else
zpidfile="$TMP/zombie.pid"; rm -f "$zpidfile"
bash -c 'set -m
         sleep 0.5 >/dev/null 2>&1 </dev/null &
         echo $! >"$1"
         set +m
         kill -STOP $$
         wait' _ "$zpidfile" >/dev/null 2>&1 </dev/null &
zparent=$!
zpid=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -s "$zpidfile" ] && zpid="$(cat "$zpidfile")" && break
  sleep 0.1
done
# Wait for real zombiehood rather than guessing at it, through `pstate`, which
# is this harness's own reader and shares no code with `group_alive`.
zstate=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    zstate="$(pstate "$zpid")"
    case "$zstate" in Z*) break ;; esac
    sleep 0.1
  done
fi
if [ -z "$zpid" ]; then
  report 1 "$zlabel" "could not start the zombie maker"
  report 1 "$clabel" "could not start the zombie maker"
else
  case "$zstate" in
    Z*)
      # Whether the BARE probe over-reports here is a property of the kernel:
      # Linux (and every container on it) counts a zombie as a group member and
      # says alive, which is the bug; Darwin's killpg answers ESRCH once the
      # group holds only corpses, so the bare probe happens to agree there. The
      # claim itself holds on both, and the control below runs on both.
      if kill -0 -- "-$zpid" 2>/dev/null; then
        zwhere="the bare \`kill -0 -- -$zpid\` says ALIVE here -- the misreading this fixes"
      else
        zwhere="this kernel's killpg already answers dead for a zombie-only group"
      fi
      if group_alive "$zpid"; then
        report 1 "$zlabel" \
          "group_alive counted a zombie as work -- stop can report a phantom orphaned group"
      else
        report 0 "$zlabel ($zwhere)"
      fi
      # The negative control, run everywhere: shadow `kill` so the signal probe
      # inside group_alive succeeds, which is exactly what the bare check did on
      # the kernel where this was reproduced. The state reader must still refuse
      # the group -- otherwise this platform's pass above was the gate's doing
      # alone and the ladder itself is untested here.
      (
        kill() { case "$*" in -0*) return 0 ;; *) command kill "$@" ;; esac; }
        if ! kill -0 -- "-$zpid"; then
          exit 3   # the shadow did not take; the control tested nothing
        fi
        group_alive "$zpid" && exit 1
        exit 0
      )
      case $? in
        0) report 0 "$clabel" ;;
        3) report 1 "$clabel" 'the `kill` shadow did not take effect' ;;
        *) report 1 "$clabel" \
             "with the signal probe forced alive, group_alive called a zombie-only group alive" ;;
      esac
      ;;
    *)
      report 1 "$zlabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$clabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      ;;
  esac
fi
# Let the parent reap its own child rather than orphaning the zombie, then check
# we really left nothing behind.
kill -CONT "$zparent" 2>/dev/null
wait "$zparent" 2>/dev/null
zparent=""
zleft=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    zleft="$(pstate "$zpid")"
    [ -z "$zleft" ] && break
    sleep 0.1
  done
fi
if [ -z "$zleft" ]; then
  report 0 "$rlabel"
else
  report 1 "$rlabel" "pid $zpid is still present as '$zleft'"
fi
fi

# ---------------------------------------------------------------------------
# Leg 5: only a positive decimal integer is a pgid
# ---------------------------------------------------------------------------
# A corrupted or forged pgid file must never reach kill: `kill -0 -- -0` targets
# the CALLER's own group (which is alive, so the answer would be a confident
# yes) and a negative reading broadcasts.
bad=""
for candidate in "" "0" "12x" "-1" "abc" " 7"; do
  if group_alive "$candidate"; then bad="$bad '$candidate'"; fi
done
if [ -z "$bad" ]; then
  report 0 "a pgid that is not a positive decimal integer is refused"
else
  report 1 "a pgid that is not a positive decimal integer is refused" \
    "accepted:$bad"
fi

# ---------------------------------------------------------------------------
# Legs 6-7: the two sentences `stop` prints about a surviving process group
# ---------------------------------------------------------------------------
# They are driven for real: `tools/test-run.sh stop last` against a FORGED run
# directory -- the metadata a launch records (cmd, cap, wt, log, pgid, gtoken)
# written by hand around a process group this harness controls -- and the answer
# read from what stop actually printed.
#
# Nothing of the ambient run history is touched: OCANNL_TOOL_TEST_RUNS is
# pointed at this run's temp dir, so the `last` pointer these legs move is the
# fixture's own and dies with the temp dir. `last` rather than an explicit run
# directory, because that is the spelling an operator uses and it exercises the
# pointer too; its name is keyed on the worktree, and the key is EXTRACTED from
# the shipping script and evaluated with the cwd that script gives itself (its
# own repo root, not the caller's), so a change to the keying moves the fixture
# with it instead of quietly leaving these legs unable to find a run.
#
# The fixture deliberately records no pid/ptoken and no wpid/wtoken and leaves
# no `exit` file: a run with any of those is managed or finished, and the group
# branch is the one reached by a run whose supervisor and wrapper are both gone
# while its process group is not.
l_ignored="stop: a group whose leader ignores TERM is reported with the unreaped-exits caveat"
l_killed="stop: that escalation kills the whole group, not just its leader"
l_took="stop: a group whose leader takes the TERM is reported as TERMed, not as ignoring it"

SRC_ROOT="$(cd -P "$HERE/.." && pwd -P)"
STOP_RUNS="$TMP/runs"
STOP_WT="$TMP/wt"
wt_expr="$(sed -n '/^wt_key=/p' "$SRC")"
key_for() { # <repo root> -> the `last` pointer key the script uses from there
  ( cd -P "$1" 2>/dev/null || exit 0
    eval "$wt_expr" 2>/dev/null
    printf '%s' "${wt_key:-}" )
}

# A leg that cannot establish its premise must skip, not pass: without a pgid
# reader the fixture could name THIS shell's group and stop would signal the
# harness itself, and without a start token group_verified refuses the fixture
# and every leg reads the same "nothing left to signal".
stop_skip=""
if [ "$have_pgid" = 0 ]; then
  stop_skip="no way to read a process's group, so a fixture group cannot be told from this shell's own"
elif [ -z "$wt_expr" ] || [ -z "$(key_for "$SRC_ROOT")" ]; then
  stop_skip="could not derive the \`last\` pointer key from $SRC"
elif [ -z "$(ps_token $$)" ]; then
  stop_skip="this system records no start token, so a forged leader cannot be identity-verified"
fi

mk_fixture() { # <tag> <pgid> <pointer key>; 0 iff the run is reachable as `last`
  local d="$STOP_RUNS/19700101-000000-$1"
  mkdir -p "$d" "$STOP_WT" || return 1
  # cmd and cap are what resolve_run demands before it will trust a directory
  # enough to signal anything named in it.
  printf 'runtest (test-test-run.sh fixture %s)\n' "$1" >"$d/cmd"
  printf '0\n' >"$d/cap"
  printf '%s\n' "$STOP_WT" >"$d/wt"
  : >"$d/log"
  printf '%s\n' "$2" >"$d/pgid"
  ps_token "$2" >"$d/gtoken"
  # group_verified refuses an empty token outright, so it is checked here
  # rather than left to reappear as a wording failure three legs later.
  [ -s "$d/gtoken" ] || return 1
  printf '%s\n' "$d" >"$STOP_RUNS/last-$3" || return 1
  return 0
}

start_leader() { # <marker> <bash -c body>; sets $leader (and its $member)
  local m=$1 body=$2 p g i
  leader=""; member=""; member_token=""
  rm -f "$m"
  # `set -m` puts the child in its own process group -- the shape stop signals,
  # and the only shape it is safe to hand a group kill.
  set -m
  bash -c "$body" _ "$m" >/dev/null 2>&1 </dev/null &
  p=$!
  set +m
  # Recorded BEFORE the checks below rather than after them: job control has
  # just put this child out of reach of any signal aimed at the harness, so an
  # INT arriving in the window between the fork and the recording would leave
  # the EXIT cleanup with nothing to kill and the fixture group running past
  # the run (Codex round 1, P2). Recording early is safe in the case the checks
  # are about to reject: a group kill aimed at a pid that never led a group
  # names a group that does not exist.
  leader=$p
  # The body writes the marker -- carrying the pid of the SECOND member it put
  # in the group -- after installing its TERM disposition, so this wait is what
  # makes the leg's premise true rather than merely likely.
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$m" ] && break
    sleep 0.1
  done
  g="$(ppgid "$p")"
  member="$(tr -dc '0-9' <"$m" 2>/dev/null)"
  member_token="$(ps_token "$member" 2>/dev/null)"
  # The second member is checked into the group, not assumed into it: a fixture
  # whose extra process ended up somewhere else would leave leg 6 judging the
  # escalation on the leader alone again, which is the hole it exists to close.
  if [ ! -s "$m" ] || [ "$g" != "$p" ] || [ -z "$member" ] ||
     [ "$(ppgid "$member")" != "$p" ]; then
    end_leader
    return 1
  fi
  return 0
}
kill_member() { # KILL the group's second member -- only if it is still IT
  # Unlike the leader, the member is a GRANDchild: nothing here holds it as a
  # zombie, so once `stop` has killed it, init reaps it and its pid is free to
  # be recycled. An unconditional numeric kill from the EXIT trap could then
  # land on an unrelated process on a busy host (Codex round 2, P2). The gate
  # is the same one the shipping script uses for every pid it did not just
  # fork: the recorded start token has to still be the one that pid answers
  # with. (lstart's one-second resolution is the known floor there, which is
  # why test-run.sh prefers /proc starttime where it exists.)
  [ -n "${member:-}" ] && [ -n "${member_token:-}" ] || return 0
  [ "$(ps_token "$member" 2>/dev/null)" = "$member_token" ] || return 0
  kill -KILL "$member" 2>/dev/null
  return 0
}
end_leader() { # whatever the leg concluded, the whole fixture group goes
  if [ -n "${leader:-}" ]; then
    kill -KILL -- "-$leader" 2>/dev/null
    kill -KILL "$leader" 2>/dev/null
  fi
  # Named separately as well as reached through the group: this is the
  # harness's backstop for the very defect leg 6 now tests for, an escalation
  # that reached only the leader. The leader needs no such check -- it is this
  # shell's own child, held as a zombie until the `wait` below, so its pid
  # cannot be recycled underneath us.
  kill_member
  if [ -n "${leader:-}" ]; then wait "$leader" 2>/dev/null; fi
  leader=""; member=""; member_token=""
  return 0
}
# Each fixture group holds TWO processes, because a group holding only its
# leader cannot tell a group kill from a leader kill: with one member, the
# incorrect `kill -KILL "$pg"` passes every leg here while real dune children
# would survive it (Codex round 1, P2). The member is a plain background
# `sleep` the leader forks before exec'ing its own, so it is in the group and
# inherits the leader's TERM disposition -- SIG_IGN survives both fork and
# exec, so an IGNORING fixture is TERM-proof as a whole, and a TAKING one dies
# as a whole. The marker doubles as the member's pid, so the harness can track
# a process that is not its own child.
body_ignores='trap "" TERM; sleep 600 & echo $! >"$1"; exec sleep 600'
body_takes='sleep 600 & echo $! >"$1"; exec sleep 600'
stop_out=""; stop_rc=""; stop_pg=""; stop_member=""; stop_err=""; stop_diag=""
stop_probe() { # <tag> <leader body> <script> <pointer key>
  stop_out=""; stop_rc=""; stop_pg=""; stop_member=""; stop_err=""; stop_diag=""
  if ! start_leader "$TMP/$1.marker" "$2"; then
    stop_err="could not start a two-process group leader for the '$1' fixture"
    return 1
  fi
  # Kept past end_leader, which clears the live handles: the escalation claim
  # is about processes that are supposed to be gone by the time it is asked.
  stop_pg=$leader
  stop_member=$member
  if ! mk_fixture "$1" "$stop_pg" "$4"; then
    stop_err="could not build the '$1' fixture run directory under $STOP_RUNS"
    return 1
  fi
  # stdout and stderr kept APART. The sentence is stdout, and it is matched
  # whole; stderr carries diagnostics that are not part of the answer and can
  # appear for reasons that have nothing to do with the wording -- a /proc
  # entry vanishing under group_alive's scan being the one that actually bit
  # (Codex round 2, P2). Merging the two made a passing stop fail an exact
  # match. It is kept and shown on failure rather than discarded, since a
  # failing leg is exactly when it is worth reading.
  stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$3" stop last 2>"$TMP/$1.stderr")"
  stop_rc=$?
  stop_diag="$(cat "$TMP/$1.stderr" 2>/dev/null)"
  return 0
}
said() { # <label> <the whole line stop must have printed>
  # The WHOLE output, not a substring of it, and a clean exit with it: these
  # two legs exist to keep the two sentences apart, and a containment test
  # would pass both for a stop that printed two of them at once (Codex
  # round 1, P2).
  if [ "${stop_rc:-1}" = 0 ] && [ "$stop_out" = "$2" ]; then
    report 0 "$1"
  else
    report 1 "$1" \
      "expected exactly \"$2\"; stop (exit ${stop_rc:-?}) printed: ${stop_out:-<nothing>}"
    [ -n "${stop_diag:-}" ] && printf '      on stderr: %s\n' "$stop_diag"
  fi
}

if [ -n "$stop_skip" ]; then
  skip "$l_ignored" "$stop_skip"
  skip "$l_killed" "$stop_skip"
  skip "$l_took" "$stop_skip"
else
  stop_key="$(key_for "$SRC_ROOT")"

  # -------------------------------------------------------------------------
  # Leg 6: the leader ignores TERM
  # -------------------------------------------------------------------------
  if stop_probe ignores "$body_ignores" "$SRC" "$stop_key"; then
    said "$l_ignored" \
      "orphaned process group $stop_pg survived TERM (possibly only as unreaped exited processes); escalated to KILL"
    # The sentence is a claim about what stop DID, so the doing is checked too:
    # an escalation that only announced itself would leave the group holding the
    # worktree lock, which is the whole reason stop reaches for KILL here.
    if [ "$have_state" = 0 ]; then
      skip "$l_killed" "no way to read a process's state on this system"
    else
      # Both members, so a KILL that reached only the leader fails here.
      k_left=""
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        k_left=""
        for k_pid in $stop_pg $stop_member; do
          case "$(pstate "$k_pid")" in '' | Z*) ;; *) k_left="$k_left $k_pid" ;; esac
        done
        [ -z "$k_left" ] && break
        sleep 0.1
      done
      if [ -z "$k_left" ]; then
        report 0 "$l_killed"
      else
        report 1 "$l_killed" \
          "still running two seconds after the escalation:$k_left (group $stop_pg)"
      fi
    fi
  else
    report 1 "$l_ignored" "$stop_err"
    report 1 "$l_killed" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 7: the leader takes the TERM
  # -------------------------------------------------------------------------
  # The difference from leg 6 is one `trap` in the leader and nothing else, so
  # a stop that reported both the same way would fail exactly one of them.
  if stop_probe takes "$body_takes" "$SRC" "$stop_key"; then
    said "$l_took" "sent TERM to the orphaned process group $stop_pg; re-run stop to confirm"
  else
    report 1 "$l_took" "$stop_err"
  fi
  end_leader
fi

# ---------------------------------------------------------------------------
# Legs 9-13: repeat mode's output and exit-code contract
# ---------------------------------------------------------------------------
repeat_root=$TMP/repeat-repo
repeat_bin=$TMP/repeat-bin
mkdir -p "$repeat_root/tools" "$repeat_bin"
cp "$SRC" "$repeat_root/tools/test-run.sh"
chmod +x "$repeat_root/tools/test-run.sh"
cat >"$repeat_bin/dune" <<'EOF'
#!/usr/bin/env bash
set -u
# Close the inherited lock descriptor before probing through a fresh open.
# Acquiring here would prove repeat released its one set-wide lock too early.
if perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' \
   9>&- <"$REPEAT_TEST_ROOT/.test-run.lock"; then
  echo "repeat fixture acquired the supposedly held worktree lock" >&2
  exit 91
fi
# Repeat establishes a fresh context with `dune clean` before each measured
# invocation. The fixture keeps setup out of the iteration count and streams.
if [ "${1:-}" = clean ]; then
  printf 'clean %s\n' "$*" >>"$REPEAT_TEST_CALLS"
  exit 0
fi
n=0
[ ! -f "$REPEAT_TEST_COUNTER" ] || n=$(cat "$REPEAT_TEST_COUNTER")
n=$((n + 1))
printf '%s\n' "$n" >"$REPEAT_TEST_COUNTER"
printf '%s\n' "$*" >>"$REPEAT_TEST_CALLS"
if [ -n "${REPEAT_TEST_WAIT_PREFIX:-}" ]; then
  : >"$REPEAT_TEST_WAIT_PREFIX.ready"
  while [ ! -e "$REPEAT_TEST_WAIT_PREFIX.release" ]; do sleep 0.05; done
fi
case $REPEAT_TEST_MODE in
  stable) printf 'stable stdout\n'; printf 'stable stderr\n' >&2 ;;
  stdout) printf 'stdout %s\n' "$n"; printf 'stable stderr\n' >&2 ;;
  stderr) printf 'stable stdout\n'; printf 'stderr %s\n' "$n" >&2 ;;
  fail) printf 'stable stdout\n'; printf 'stable failure\n' >&2; exit 7 ;;
  *) echo "unknown repeat fixture mode: $REPEAT_TEST_MODE" >&2; exit 92 ;;
esac
EOF
chmod +x "$repeat_bin/dune"

repeat_out= repeat_rc= repeat_dir=
repeat_probe() { # tag mode [repeat options/count/dune argv...]
  local tag=$1 mode=$2 runs=$TMP/repeat-runs-$1
  shift 2
  mkdir -p "$runs"
  : >"$TMP/$tag.counter"
  : >"$TMP/$tag.calls"
  REPEAT_TEST_ROOT=$repeat_root \
  REPEAT_TEST_MODE=$mode \
  REPEAT_TEST_COUNTER=$TMP/$tag.counter \
  REPEAT_TEST_CALLS=$TMP/$tag.calls \
  REPEAT_TEST_WAIT_PREFIX= \
  OCANNL_TOOL_TEST_RUNS=$runs \
  PATH=$repeat_bin:$PATH \
    "$repeat_root/tools/test-run.sh" repeat "$@" >"$TMP/$tag.out" 2>"$TMP/$tag.err"
  repeat_rc=$?
  repeat_out=$(cat "$TMP/$tag.out")
  repeat_dir=$(find "$runs" -mindepth 1 -maxdepth 1 -type d -name '2*Z-*' | head -1)
}

repeat_probe repeat-identical stable 3 build @cheap
if [ "$repeat_rc" = 0 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$repeat_out" \
   && [ "$(cat "$TMP/repeat-identical.counter")" = 3 ] \
   && [ "$(grep -c '^clean ' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--force' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--cache=disabled' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--build-dir=' "$TMP/repeat-identical.calls")" = 6 ] \
   && [ -n "$repeat_dir" ] \
   && [ ! -e "$repeat_dir/build" ] \
   && [ "$(find "$repeat_dir" \( -name stdout -o -name stderr \) | wc -l | tr -d ' ')" = 6 ]; then
  report 0 "repeat: identical forced runs retain every stdout/stderr"
else
  report 1 "repeat: identical forced runs retain every stdout/stderr" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}; stderr: $(cat "$TMP/repeat-identical.err")"
fi

repeat_probe repeat-stdout stdout 3 build @cheap
if [ "$repeat_rc" = 1 ] && grep -q '^repeat result: DIFFERING -- ' <<<"$repeat_out" \
   && [ -s "$repeat_dir/diffs/1-2.stdout" ]; then
  report 0 "repeat: stdout drift is red and pairwise-diffed"
else
  report 1 "repeat: stdout drift is red and pairwise-diffed" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-stderr stderr 3 build @cheap
if [ "$repeat_rc" = 0 ] && grep -q '^repeat result: STDERR-ONLY -- ' <<<"$repeat_out" \
   && [ -s "$repeat_dir/diffs/1-2.stderr" ]; then
  report 0 "repeat: stderr-only drift is distinct and diagnostic-green"
else
  report 1 "repeat: stderr-only drift is distinct and diagnostic-green" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-red fail 2 build @cheap
if [ "$repeat_rc" = 7 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$repeat_out"; then
  report 0 "repeat: a repeatable red dune leg keeps its exit code"
else
  report 1 "repeat: a repeatable red dune leg keeps its exit code" \
    "expected exit 7; got $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-alone stable --alone 2 build @cheap
if [ "$repeat_rc" = 0 ] && [ "$(grep -c -- '-j 1' "$TMP/repeat-alone.calls")" = 2 ] \
   && grep -q 'iteration 1/2 -- dune (alone, -j 1)' <<<"$repeat_out"; then
  report 0 "repeat: --alone serializes every dune iteration"
else
  report 1 "repeat: --alone serializes every dune iteration" \
    "exit $repeat_rc; calls: $(tr '\n' ';' <"$TMP/repeat-alone.calls"); output: ${repeat_out:-<nothing>}"
fi

# An active repeat must replace `last`, and stop must signal the OUTER
# coordinator so it records cancellation and refuses to start iteration two.
repeat_stop_runs=$TMP/repeat-runs-stop
repeat_stop_prefix=$TMP/repeat-stop
mkdir -p "$repeat_stop_runs"
: >"$TMP/repeat-stop.counter"
: >"$TMP/repeat-stop.calls"
REPEAT_TEST_ROOT=$repeat_root \
REPEAT_TEST_MODE=stable \
REPEAT_TEST_COUNTER=$TMP/repeat-stop.counter \
REPEAT_TEST_CALLS=$TMP/repeat-stop.calls \
REPEAT_TEST_WAIT_PREFIX=$repeat_stop_prefix \
OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 3 build @cheap \
  >"$TMP/repeat-stop.out" 2>"$TMP/repeat-stop.err" &
repeat_pid=$!
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  [ -e "$repeat_stop_prefix.ready" ] && break
  sleep 0.1
done
status_out=$(OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
  "$repeat_root/tools/test-run.sh" status last 2>"$TMP/repeat-stop-status.err")
status_rc=$?
stop_out=$(OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
  "$repeat_root/tools/test-run.sh" stop last 2>"$TMP/repeat-stop-stop.err")
stop_rc=$?
touch "$repeat_stop_prefix.release"
wait "$repeat_pid"
repeat_stop_rc=$?
repeat_pid=
if [ "$status_rc" = 3 ] && grep -q '^running: ' <<<"$status_out" \
   && [ "$stop_rc" = 0 ] && grep -q '^sent TERM to the repeat coordinator; ' <<<"$stop_out" \
   && [ "$repeat_stop_rc" = 143 ] \
   && [ "$(cat "$TMP/repeat-stop.counter")" = 1 ] \
   && grep -q '^repeat result: CANCELLED -- completed 1 of 3 iterations$' "$TMP/repeat-stop.out"; then
  report 0 "repeat: last resolves active state and stop cancels the whole set"
else
  report 1 "repeat: last resolves active state and stop cancels the whole set" \
    "status $status_rc: ${status_out:-<nothing>}; stop $stop_rc: ${stop_out:-<nothing>}; repeat $repeat_stop_rc: $(cat "$TMP/repeat-stop.out")"
fi

echo
# The skip count is printed on every run, not only when it is nonzero: "all legs
# passed" over a run that decided three of them is the reading to prevent.
if [ "$failures" -eq 0 ]; then
  echo "all legs passed ($skipped skipped)"
else
  echo "$failures leg(s) failed ($skipped skipped)"
fi
exit $(( failures > 0 ? 1 : 0 ))
