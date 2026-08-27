#!/usr/bin/env bash
# Hand-run tests for `group_alive` in tools/test-run.sh -- the predicate the
# three group probes there ask "is anything in this process group still
# running?" (gh-ocannl-742).
#
#   tools/test-test-run.sh          # run every leg
#   tools/test-test-run.sh --keep   # keep the temp dir for inspection
#
# It is the sibling of scripts/test-setup-ocaml-env.sh, whose leg 1 (f) tests
# the same predicate in the SessionStart hook, and it is deliberately NOT wired
# into any dune alias for the same reason: it spawns, STOPs and kills processes,
# which is a poor fit for `dune runtest`. (What dune does check about this file
# is that it parses -- test/operations/shell_scripts_parse globs tools/.)
#
# It tests the WORKING-TREE copy: `group_alive` is extracted from the
# tools/test-run.sh next to this script and sourced, so the legs exercise the
# text that ships rather than a paraphrase of it, and the extraction is asserted
# structurally before anything uses it -- a sed that matched nothing would
# otherwise leave every leg passing without testing anything.
#
# Legs:
#   1. extraction -- the function came out of the shipping script.
#   2. a genuinely live group reads ALIVE (the other side of leg 4: without
#      this, a group_alive that answered "dead" always would pass leg 4).
#   3. a group with no members at all reads DEAD, and so does the bare signal.
#   4. a group holding nothing but a ZOMBIE reads dead, where the bare
#      `kill -0 -- -PGID` this replaced reads it as alive. That misreading is
#      what made `stop` able to announce an orphaned group holding only corpses.
#   5. a pgid that is not a positive decimal integer is refused -- 0 and
#      negatives are kill specials (caller's own group, broadcast).

set -u

KEEP=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-test-run.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/test-run.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }

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

echo "testing $SRC"
printf '  digest %s\n' "$( (cksum <"$SRC") 2>/dev/null || echo '?')"

# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails
# would leave TMP empty and `rm -rf "$TMP"` would be handed the ROOT.
zparent=""   # leg 4's self-stopping zombie maker; cleanup must resume it
livepid=""   # leg 2's live group leader
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
# maker behind. Exiting from the handler is what gets EXIT to fire.
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# Leg 1: extraction
# ---------------------------------------------------------------------------
sed -n '/^group_alive() {/,/^}/p' "$SRC" >"$TMP/group_alive.sh"
# Checked structurally -- opens with the header, closes with the brace, has a
# body -- rather than by grepping for one line of it: this guard must still hold
# while the function under test is being mutated to see a leg go red. The header
# is matched as a PREFIX, since it carries a trailing comment.
g_lines="$(wc -l <"$TMP/group_alive.sh" | tr -d ' ')"
g_head="$(head -n1 "$TMP/group_alive.sh")"
case $g_head in "group_alive() {"*) g_ok=1 ;; *) g_ok=0 ;; esac
if [ "$g_ok" = 0 ] \
   || [ "$(tail -n1 "$TMP/group_alive.sh")" != "}" ] || [ "$g_lines" -lt 10 ]; then
  report 1 "group_alive: extracted" "sed did not capture the function body from $SRC"
else
  report 0 "group_alive: extracted ($g_lines lines)"
fi
# shellcheck disable=SC1090
. "$TMP/group_alive.sh"

# ---------------------------------------------------------------------------
# Leg 2: a live group reads alive
# ---------------------------------------------------------------------------
# `set -m` puts the child in a process group of its own, so its pid IS a pgid
# holding exactly one live process -- the shape all three call sites signal.
set -m
sleep 30 >/dev/null 2>&1 </dev/null &
livepid=$!
set +m
lpgid="$(ps -o pgid= -p "$livepid" 2>/dev/null | tr -d ' ')"
if [ "$lpgid" != "$livepid" ]; then
  report 1 "a group holding a running process reads as alive" \
    "the child did not lead its own group (pgid '${lpgid:-gone}' vs pid $livepid)"
elif ! kill -0 -- "-$livepid" 2>/dev/null; then
  report 1 "a group holding a running process reads as alive" \
    "the group is not even signal-reachable; the leg tested nothing"
elif group_alive "$livepid"; then
  report 0 "a group holding a running process reads as alive"
else
  report 1 "a group holding a running process reads as alive" \
    "group_alive said dead for a group with a live sleep in it"
fi

# ---------------------------------------------------------------------------
# Leg 3: an empty group reads dead
# ---------------------------------------------------------------------------
kill -KILL -- "-$livepid" 2>/dev/null
wait "$livepid" 2>/dev/null   # reaped here, so the group really is empty
livepid=""
if kill -0 -- "-$lpgid" 2>/dev/null; then
  report 1 "a group with no members left reads as dead" \
    "pgid $lpgid is still signal-reachable after the kill and reap"
elif group_alive "$lpgid"; then
  report 1 "a group with no members left reads as dead" \
    "group_alive said alive for an empty group"
else
  report 0 "a group with no members left reads as dead"
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
# Wait for real zombiehood rather than guessing at it, and read the state with
# `ps -p`, independently of the `group_alive` under test.
zstate=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    zstate="$(ps -o state= -p "$zpid" 2>/dev/null | tr -d ' ')"
    case "$zstate" in Z*) break ;; esac
    sleep 0.1
  done
fi
zlabel="a group holding nothing but a zombie reads as dead"
clabel="the state reader alone rejects a zombie-only group (signal probe forced to say alive)"
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
    zleft="$(ps -o state= -p "$zpid" 2>/dev/null | tr -d ' ')"
    [ -z "$zleft" ] && break
    sleep 0.1
  done
fi
if [ -z "$zleft" ]; then
  report 0 "the zombie leg reaps its own zombie, leaving no process-table entry"
else
  report 1 "the zombie leg reaps its own zombie, leaving no process-table entry" \
    "pid $zpid is still present as '$zleft'"
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

echo
if [ "$failures" -eq 0 ]; then
  echo "all legs passed"
else
  echo "$failures leg(s) failed"
fi
exit $(( failures > 0 ? 1 : 0 ))
