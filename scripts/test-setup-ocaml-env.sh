#!/usr/bin/env bash
# Hand-run tests for the "staleness against origin/master" section of
# scripts/setup-ocaml-env.sh (the SessionStart hook).
#
#   scripts/test-setup-ocaml-env.sh          # run every leg
#   scripts/test-setup-ocaml-env.sh --keep   # keep the temp dir for inspection
#
# Run it after editing that section. It takes about 90 seconds, nearly all of it
# spent sitting out watchdog timeouts, and is deliberately NOT wired into any
# dune alias: it spawns and kills process groups and would be a poor fit for
# `dune runtest`.
#
# It tests the WORKING-TREE copy of the hook, not the committed one: the script
# next to this file is copied into every throwaway clone. During the six review
# rounds of PR #430 an ad-hoc harness silently tested the committed script after
# cloning the repo, and a `run` helper executed its own label as a command — so
# this harness prints the source path and digest it is testing, and every hook
# run asserts it actually reached the section under test.
#
# Nothing here touches the real repository: the clones live under a `mktemp -d`
# directory, run with `env -i` and a sanitised PATH (no opam, so the hook stops
# right after the section under test), and see neither the user's global nor the
# system gitconfig.
#
# Legs:
#   1. `bounded` — the watchdog: TERM at the bound, KILL 5s later, process-group
#      kill, rc preservation, no orphans.
#   2. counting — behind/ahead wording and recovery command, offline fallback,
#      ref-ambiguity, FETCH_HEAD untouched, no-origin silence.
#   3. SSH launcher gating — which program git ends up invoking and whether the
#      OpenSSH options were appended to it.

set -u

KEEP=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-setup-ocaml-env.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
HOOK_SRC="$HERE/setup-ocaml-env.sh"
[ -f "$HOOK_SRC" ] || { echo "no $HOOK_SRC" >&2; exit 2; }
BASH_BIN="$(command -v bash)"

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

echo "testing $HOOK_SRC"
printf '  digest %s\n' "$( (cksum <"$HOOK_SRC") 2>/dev/null || echo '?')"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/setup-ocaml-env-test.XXXXXX")"
cleanup() {
  # A hook broken in the way leg 1 probes for can leave orphans behind. They
  # carry this run's pid in their duration (see D_* below), so they are
  # unambiguously ours to reap and no concurrent run is disturbed.
  local orphans
  orphans="$(pgrep -x sleep -a 2>/dev/null | awk -v p="$$" '$3 ~ ("^[0-9]+[.]" p "$") { print $1 }')"
  [ -n "$orphans" ] && kill -KILL $orphans 2>/dev/null
  if [ "$KEEP" = 1 ]; then echo "kept $TMP"; else rm -rf "$TMP"; fi
  return 0
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Leg 1: bounded
# ---------------------------------------------------------------------------
# Extract the function from the working-tree hook and source it, so the legs
# below exercise the very text that ships rather than a paraphrase of it.
sed -n '/^bounded() {/,/^}/p' "$HOOK_SRC" >"$TMP/bounded.sh"
# Checked structurally — opens with the header, closes with the brace, has a
# body — rather than by grepping for one line of it: this guard exists to catch
# a sed that matched nothing, and must still hold while a leg under test is
# being mutated to see the assertion fail.
b_lines="$(wc -l <"$TMP/bounded.sh" | tr -d ' ')"
if [ "$(head -n1 "$TMP/bounded.sh")" != "bounded() {" ] \
   || [ "$(tail -n1 "$TMP/bounded.sh")" != "}" ] || [ "$b_lines" -lt 5 ]; then
  report 1 "bounded: extracted" "sed did not capture the function body from $HOOK_SRC"
else
  report 0 "bounded: extracted ($b_lines lines)"
fi
# shellcheck disable=SC1090
. "$TMP/bounded.sh"

# Survivors are counted by EXACT duration: `pgrep -x sleep` alone also matches
# the watchdog's own `sleep 3` / `sleep 1`, and a substring `pgrep sleep` would
# additionally match this harness's command line. The durations also carry this
# run's pid, so a second copy of this script (or a leftover from a crashed one)
# cannot be miscounted as this run's orphan.
D_TERM="91.$$"      # (a) honours TERM
D_IGN_CHILD="92.$$" # (b) the child that ignores TERM
D_IGN_PARENT="93.$$" # (b) the parent that does not
D_IGN_ALL="94.$$"   # (c) everything ignores TERM
D_DAEMON="95.$$"    # (d) the daemon left behind by an exit-0 command
D_WATCHDOG="97.$$"  # (e) the bound, i.e. the watchdog's own sleep
survivors() { # survivors DURATION -> count of live `sleep DURATION`
  pgrep -x sleep -a 2>/dev/null | awk -v d="$1" '$3 == d { n++ } END { print n + 0 }'
}
settled_survivors() { # settled_survivors DURATION -- allow for asynchronous reaping
  local i n
  n=0
  for i in 1 2 3 4 5; do
    n="$(survivors "$1")"
    [ "$n" = 0 ] && break
    sleep 0.2
  done
  printf '%s\n' "$n"
}

BOUND=3

if ! command -v pgrep >/dev/null 2>&1 || ! command -v awk >/dev/null 2>&1; then
  report 1 "bounded: prerequisites" "pgrep and awk are required to count survivors"
else
  # Guard: `bounded` signals the process GROUP. If `set -m` did not give the
  # child a group of its own, that group is OURS and the first leg would kill
  # this harness. Check before running any of them.
  self_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
  set -m
  sleep 98.$$ >/dev/null 2>&1 </dev/null & probe=$!
  set +m
  probe_pgid="$(ps -o pgid= -p "$probe" 2>/dev/null | tr -d ' ')"
  if [ -n "$self_pgid" ] && [ "$self_pgid" = "$probe_pgid" ]; then
    kill "$probe" 2>/dev/null
    report 1 "bounded: job control gives the child its own process group" \
      "pgid $probe_pgid == harness pgid; skipping the bounded legs rather than killing this shell"
  else
    kill -KILL -- -"$probe" 2>/dev/null
    wait "$probe" 2>/dev/null
    report 0 "bounded: job control gives the child its own process group"

    # (a) The command honours TERM: return at the bound, rc=143, nothing left.
    t0=$SECONDS
    bounded "$BOUND" sleep "$D_TERM" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_TERM")"
    if [ "$rc" = 143 ] && [ "$el" -ge "$BOUND" ] && [ "$el" -le $((BOUND + 2)) ] && [ "$left" = 0 ]; then
      report 0 "bounded (a): TERM-honouring command returns at the bound, rc=143, no orphans"
    else
      report 1 "bounded (a): TERM-honouring command returns at the bound, rc=143, no orphans" \
        "rc=$rc elapsed=${el}s (want ${BOUND}s) survivors=$left"
    fi

    # (b) Parent dies on TERM, a child ignores it: the group is not empty, so
    #     `bounded` must stay until the KILL escalation rather than returning
    #     with an orphan still running.
    t0=$SECONDS
    bounded "$BOUND" bash -c '(trap "" TERM; exec sleep "$1") & sleep "$2"' _ \
      "$D_IGN_CHILD" "$D_IGN_PARENT" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_IGN_CHILD")"; left2="$(settled_survivors "$D_IGN_PARENT")"
    if [ "$el" -ge $((BOUND + 4)) ] && [ "$el" -le $((BOUND + 7)) ] && [ "$left" = 0 ] && [ "$left2" = 0 ]; then
      report 0 "bounded (b): TERM-ignoring child holds the return until KILL, no orphans"
    else
      report 1 "bounded (b): TERM-ignoring child holds the return until KILL, no orphans" \
        "rc=$rc elapsed=${el}s (want $((BOUND + 5))s) survivors=$left/$left2"
    fi

    # (c) Everything ignores TERM: killed at bound+5, rc=137.
    t0=$SECONDS
    bounded "$BOUND" bash -c 'trap "" TERM; exec sleep "$1"' _ "$D_IGN_ALL" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_IGN_ALL")"
    if [ "$rc" = 137 ] && [ "$el" -ge $((BOUND + 4)) ] && [ "$el" -le $((BOUND + 7)) ] && [ "$left" = 0 ]; then
      report 0 "bounded (c): TERM-ignoring command is KILLed at bound+5s, rc=137"
    else
      report 1 "bounded (c): TERM-ignoring command is KILLed at bound+5s, rc=137" \
        "rc=$rc elapsed=${el}s (want $((BOUND + 5))s) survivors=$left"
    fi

    # (d) The command exits 0 early but leaves a daemon in the group: the
    #     contract is that nothing of the group survives the return.
    t0=$SECONDS
    bounded "$BOUND" bash -c 'sleep "$1" >/dev/null 2>&1 </dev/null & exit 0' _ \
      "$D_DAEMON" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_DAEMON")"
    if [ "$rc" = 0 ] && [ "$el" -ge "$BOUND" ] && [ "$el" -le $((BOUND + 2)) ] && [ "$left" = 0 ]; then
      report 0 "bounded (d): daemon left by an exit-0 command is cleaned up at the bound"
    else
      report 1 "bounded (d): daemon left by an exit-0 command is cleaned up at the bound" \
        "rc=$rc elapsed=${el}s (want ${BOUND}s) survivors=$left"
    fi

    # (e) A fast exit preserves the command's status and cancels the watchdog
    #     (no `sleep 97` left behind).
    t0=$SECONDS
    bounded "$D_WATCHDOG" bash -c 'exit 42' >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_WATCHDOG")"
    if [ "$rc" = 42 ] && [ "$el" -le 3 ] && [ "$left" = 0 ]; then
      report 0 "bounded (e): fast exit preserves rc and reaps the watchdog"
    else
      report 1 "bounded (e): fast exit preserves rc and reaps the watchdog" \
        "rc=$rc elapsed=${el}s (want ~0s) watchdog-sleeps=$left"
    fi

  fi
fi

# ---------------------------------------------------------------------------
# Shared machinery for the hook-invoking legs
# ---------------------------------------------------------------------------
# A PATH with no opam on it: the hook then stops right after the section under
# test ("=== stopped: opam required ==="), so no run of it can pin packages or
# otherwise disturb this machine's opam state.
BIN="$TMP/bin"
mkdir -p "$BIN"
for tool in git env sh sleep basename dirname tr grep sed cat; do
  p="$(command -v "$tool" 2>/dev/null)" || continue
  ln -sf "$p" "$BIN/$tool"
done
if [ ! -e "$BIN/git" ]; then
  echo "git is required" >&2; exit 2
fi
if PATH="$BIN" command -v opam >/dev/null 2>&1; then
  echo "sanitised PATH still resolves opam; refusing to run the hook" >&2; exit 2
fi

FAKEHOME="$TMP/home"; mkdir -p "$FAKEHOME"
RUN_ENV=()
RUN_PATH="$BIN"

run_hook() { # run_hook CLONE OUTFILE   (extra env comes from the RUN_ENV array)
  local clone="$1" out="$2"
  ( cd "$clone" && env -i \
      HOME="$FAKEHOME" PATH="$RUN_PATH" \
      GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
      ${RUN_ENV[@]+"${RUN_ENV[@]}"} \
      "$BASH_BIN" "$clone/scripts/setup-ocaml-env.sh" ) >"$out" 2>&1
  # Every run must have reached (and passed) the section under test. Without
  # this, a hook that died earlier would make the absence assertions pass.
  if ! grep -qF 'stopped: opam required' "$out"; then
    report 1 "harness: hook run in $(basename "$clone") reached the opam stop" \
      "output was: $(tr '\n' '|' <"$out")"
  fi
  RUN_ENV=()
  RUN_PATH="$BIN"
}

has() { grep -qF -- "$2" "$1"; }

git_q() { git -c advice.detachedHead=false -c init.defaultBranch=master \
              -c user.name=test -c user.email=test@example.invalid \
              -c commit.gpgsign=false "$@"; }

# ---------------------------------------------------------------------------
# Leg 2: counting
# ---------------------------------------------------------------------------
ORIGIN="$TMP/origin"
mkdir -p "$ORIGIN"
git_q -C "$ORIGIN" init -q
for n in 1 2 3 4 5 6 7 8 9 10; do
  echo "$n" >"$ORIGIN/f"
  git_q -C "$ORIGIN" add f
  git_q -C "$ORIGIN" commit -q -m "c$n"
done
# A second branch at a different commit, for leg (f) to seed FETCH_HEAD from.
git_q -C "$ORIGIN" branch side master~7

new_clone() { # new_clone NAME -> path of a fresh clone carrying the working-tree hook
  local d="$TMP/$1"
  rm -rf "$d"
  git_q clone -q "$ORIGIN" "$d"
  git_q -C "$d" config advice.detachedHead false
  git_q -C "$d" config user.name test
  git_q -C "$d" config user.email test@example.invalid
  mkdir -p "$d/scripts"
  cp "$HOOK_SRC" "$d/scripts/setup-ocaml-env.sh"
  printf '%s\n' "$d"
}

# (a) behind only -> the merge --ff-only recovery.
c="$(new_clone c-behind)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
run_hook "$c" "$TMP/out-behind"
if has "$TMP/out-behind" 'WARNING HEAD is 5 commit(s) behind origin/master' \
   && has "$TMP/out-behind" 'recover with: git merge --ff-only refs/remotes/origin/master'; then
  report 0 "count (a): 5 behind, 0 ahead -> WARNING + merge --ff-only recovery"
else
  report 1 "count (a): 5 behind, 0 ahead -> WARNING + merge --ff-only recovery" \
    "$(tr '\n' '|' <"$TMP/out-behind")"
fi

# (b) behind and ahead -> the rebase recovery, with the replay count.
c="$(new_clone c-diverged)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
git_q -C "$c" commit -q --allow-empty -m local1
git_q -C "$c" commit -q --allow-empty -m local2
run_hook "$c" "$TMP/out-diverged"
if has "$TMP/out-diverged" 'WARNING HEAD is 5 commit(s) behind origin/master' \
   && has "$TMP/out-diverged" 'recover with: git rebase refs/remotes/origin/master  (2 local commit(s) to replay)'; then
  report 0 "count (b): 5 behind, 2 ahead -> rebase recovery naming the 2 local commits"
else
  report 1 "count (b): 5 behind, 2 ahead -> rebase recovery naming the 2 local commits" \
    "$(tr '\n' '|' <"$TMP/out-diverged")"
fi

# (c) up to date.
c="$(new_clone c-current)"
run_hook "$c" "$TMP/out-current"
if has "$TMP/out-current" '  ok    up to date with origin/master' \
   && ! has "$TMP/out-current" 'WARNING'; then
  report 0 "count (c): up to date -> 'ok    up to date with origin/master'"
else
  report 1 "count (c): up to date -> 'ok    up to date with origin/master'" \
    "$(tr '\n' '|' <"$TMP/out-current")"
fi

# (d) fetch failure still counts, against the last successful fetch. The clone
#     already holds an origin/master from cloning; the origin URL is then
#     switched to https and pointed at a dead proxy.
c="$(new_clone c-offline)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
git_q -C "$c" remote set-url origin https://example.invalid/ocannl.git
RUN_ENV=(https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9)
run_hook "$c" "$TMP/out-offline"
if has "$TMP/out-offline" '  skip  fetching origin/master failed (offline?)' \
   && has "$TMP/out-offline" 'WARNING HEAD is 5 commit(s) behind origin/master (as of the last successful fetch)'; then
  report 0 "count (d): failed fetch -> skip, then the count as of the last successful fetch"
else
  report 1 "count (d): failed fetch -> skip, then the count as of the last successful fetch" \
    "$(tr '\n' '|' <"$TMP/out-offline")"
fi

# (e) a local branch AND a tag both called `origin/master`, at a different
#     commit, must not be read in place of the tracking ref.
c="$(new_clone c-ambiguous)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
git_q -C "$c" branch origin/master refs/remotes/origin/master~3
git_q -C "$c" tag origin/master refs/remotes/origin/master~3
run_hook "$c" "$TMP/out-ambiguous"
if has "$TMP/out-ambiguous" 'WARNING HEAD is 5 commit(s) behind origin/master' \
   && ! has "$TMP/out-ambiguous" 'HEAD is 3 commit(s) behind'; then
  report 0 "count (e): a branch and a tag named origin/master do not change the count"
else
  report 1 "count (e): a branch and a tag named origin/master do not change the count" \
    "$(tr '\n' '|' <"$TMP/out-ambiguous")"
fi

# (f) the probe must leave FETCH_HEAD byte-identical (--no-write-fetch-head):
#     someone may be keeping it for a later `git merge FETCH_HEAD`.
#     FETCH_HEAD is seeded from `side`, NOT from master: seeded from master the
#     hook's own fetch would write back the very same line, and the comparison
#     could not tell "did not write" from "wrote the same thing" — which is
#     exactly how this leg first shipped, passing against a hook with
#     `--no-write-fetch-head` removed.
c="$(new_clone c-fetchhead)"
git_q -C "$c" fetch -q origin side
master_sha="$(git_q -C "$c" rev-parse refs/remotes/origin/master 2>/dev/null || true)"
seeded_sha="$(head -n1 "$c/.git/FETCH_HEAD" 2>/dev/null | cut -f1)"
if [ ! -s "$c/.git/FETCH_HEAD" ]; then
  report 1 "count (f): FETCH_HEAD is byte-identical across a run" \
    "the setup fetch wrote no FETCH_HEAD, so there is nothing to compare"
elif [ -z "$master_sha" ] || [ "$seeded_sha" = "$master_sha" ]; then
  report 1 "count (f): FETCH_HEAD is byte-identical across a run" \
    "seeded FETCH_HEAD already names master ($seeded_sha) — the leg cannot discriminate"
else
  cp "$c/.git/FETCH_HEAD" "$TMP/fetch-head.before"
  run_hook "$c" "$TMP/out-fetchhead"
  if cmp -s "$TMP/fetch-head.before" "$c/.git/FETCH_HEAD"; then
    report 0 "count (f): FETCH_HEAD is byte-identical across a run"
  else
    report 1 "count (f): FETCH_HEAD is byte-identical across a run" \
      "before: $(cat "$TMP/fetch-head.before") / after: $(cat "$c/.git/FETCH_HEAD")"
  fi
fi

# (g) no `origin` remote: the section prints nothing at all.
c="$(new_clone c-noremote)"
git_q -C "$c" remote remove origin
run_hook "$c" "$TMP/out-noremote"
if ! grep -q 'origin/master' "$TMP/out-noremote"; then
  report 0 "count (g): no origin remote -> the section prints nothing"
else
  report 1 "count (g): no origin remote -> the section prints nothing" \
    "$(tr '\n' '|' <"$TMP/out-noremote")"
fi

# ---------------------------------------------------------------------------
# Leg 3: SSH launcher gating
# ---------------------------------------------------------------------------
# A logging fake launcher over an ssh:// origin answers two questions per case:
# WHICH program git ended up invoking, and whether the OpenSSH options were
# appended to it. The options are only correct where OpenSSH is certain.
#
# Every case here costs the FULL `bounded` bound (30s), which is why they are
# launched concurrently and joined once rather than run one after another. The
# reason is a property of the hook, not of this harness: git's ssh child is
# still a zombie in the command's process group at the instant `bounded` tests
# the group for emptiness, so the fast `exit 255` is nevertheless waited out to
# the bound. Each case has its own clone, log, launcher directory and output
# file, so concurrency is safe.
SSH_BASE="$TMP/ssh-base"
mkdir -p "$SSH_BASE/scripts"
git_q -C "$SSH_BASE" init -q
git_q -C "$SSH_BASE" commit -q --allow-empty -m base
git_q -C "$SSH_BASE" remote add origin ssh://git@example.invalid/x.git

CASEDIR=""; CASELOG=""; LAUNCHER=""
ssh_prepare() { # ssh_prepare SLUG LAUNCHER_BASENAME -- sets CASEDIR/CASELOG/LAUNCHER
  CASEDIR="$TMP/ssh-$1"
  CASELOG="$TMP/ssh-$1.log"
  rm -rf "$CASEDIR"
  cp -r "$SSH_BASE" "$CASEDIR"
  cp "$HOOK_SRC" "$CASEDIR/scripts/setup-ocaml-env.sh"
  : >"$CASELOG"
  mkdir -p "$TMP/launchers/$1"
  LAUNCHER="$(ssh_launcher "$1" "$2")"
}

ssh_launcher() { # ssh_launcher SLUG BASENAME -> path of a fake logging into that case's log
  local p="$TMP/launchers/$1/$2"
  { printf '#!/bin/sh\n'
    printf 'echo "$0 $*" >> "%s"\n' "$TMP/ssh-$1.log"
    printf 'exit 255\n'
  } >"$p"
  chmod +x "$p"
  printf '%s\n' "$p"
}

SSH_SLUGS=(); SSH_LABELS=(); SSH_PROGS=(); SSH_OPTS=()
ssh_launch() { # ssh_launch SLUG LABEL EXPECTED_PROGRAM_BASENAME yes|no
  SSH_SLUGS+=("$1"); SSH_LABELS+=("$2"); SSH_PROGS+=("$3"); SSH_OPTS+=("$4")
  local clone="$TMP/ssh-$1"
  ( cd "$clone" && env -i \
      HOME="$FAKEHOME" PATH="$RUN_PATH" \
      GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
      ${RUN_ENV[@]+"${RUN_ENV[@]}"} \
      "$BASH_BIN" "$clone/scripts/setup-ocaml-env.sh" ) >"$TMP/out-ssh-$1" 2>&1 &
  RUN_ENV=()
  RUN_PATH="$BIN"
}

# Options appended: GIT_SSH_COMMAND whose program is `ssh`.
ssh_prepare cmd-ssh ssh
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-ssh "ssh (1): GIT_SSH_COMMAND named ssh -> options appended" ssh yes

# Options appended: core.sshCommand whose program is `ssh`.
ssh_prepare cfg-ssh ssh
git_q -C "$CASEDIR" config core.sshCommand "$LAUNCHER"
ssh_launch cfg-ssh "ssh (2): core.sshCommand named ssh -> options appended" ssh yes

# Options appended: nothing configured, so git's default `ssh` off PATH.
ssh_prepare path-ssh ssh
RUN_PATH="$TMP/launchers/path-ssh:$BIN"
ssh_launch path-ssh "ssh (3): PATH default ssh -> options appended" ssh yes

# Options appended: an `ssh.exe` basename is OpenSSH too (Git for Windows).
ssh_prepare cmd-sshexe ssh.exe
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-sshexe "ssh (4): GIT_SSH_COMMAND named ssh.exe -> options appended" ssh.exe yes

# Options NOT appended: a custom wrapper name, variant unknown.
ssh_prepare cmd-wrapper my-ssh-wrapper
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-wrapper "ssh (5): custom wrapper name -> options NOT appended" my-ssh-wrapper no

# Options NOT appended: plink and its kin.
ssh_prepare cmd-plink plink
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-plink "ssh (6): plink -> options NOT appended" plink no

ssh_prepare cmd-tortoise tortoiseplink
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-tortoise "ssh (7): tortoiseplink -> options NOT appended" tortoiseplink no

ssh_prepare cmd-putty putty
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-putty "ssh (8): putty -> options NOT appended" putty no

# Options NOT appended: ssh.variant=simple, even for a launcher named ssh.
ssh_prepare cfg-simple ssh
git_q -C "$CASEDIR" config ssh.variant simple
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cfg-simple "ssh (9): ssh.variant=simple -> options NOT appended" ssh no

# Options NOT appended: GIT_SSH is a program, not a shell string — declined on
# the mechanism, not the name, so this fake is deliberately called `ssh`.
ssh_prepare env-gitssh ssh
RUN_ENV=(GIT_SSH="$LAUNCHER")
ssh_launch env-gitssh "ssh (10): GIT_SSH program named ssh -> options NOT appended" ssh no

# GIT_SSH_VARIANT outranks ssh.variant, and an explicit `ssh` variant is enough
# even for a custom program name.
ssh_prepare variant-override my-ssh-wrapper
git_q -C "$CASEDIR" config ssh.variant simple
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER" GIT_SSH_VARIANT=ssh)
ssh_launch variant-override \
  "ssh (11): GIT_SSH_VARIANT=ssh outranks ssh.variant=simple -> options appended" my-ssh-wrapper yes

# Git's own precedence GIT_SSH_COMMAND > core.sshCommand > GIT_SSH survives the
# probe: three launchers with distinct names, one dropped per case. Custom names
# throughout, so the script appends nothing and cannot itself be the reason a
# given launcher won.
ssh_prepare prec-cmd ssh-a
git_q -C "$CASEDIR" config core.sshCommand "$(ssh_launcher prec-cmd ssh-b)"
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER" GIT_SSH="$(ssh_launcher prec-cmd ssh-c)")
ssh_launch prec-cmd "ssh (12): GIT_SSH_COMMAND beats core.sshCommand and GIT_SSH" ssh-a no

ssh_prepare prec-cfg ssh-b
git_q -C "$CASEDIR" config core.sshCommand "$LAUNCHER"
RUN_ENV=(GIT_SSH="$(ssh_launcher prec-cfg ssh-c)")
ssh_launch prec-cfg "ssh (13): core.sshCommand beats GIT_SSH" ssh-b no

ssh_prepare prec-env ssh-c
RUN_ENV=(GIT_SSH="$LAUNCHER")
ssh_launch prec-env "ssh (14): GIT_SSH is used when nothing outranks it" ssh-c no

wait

i=0
while [ "$i" -lt "${#SSH_SLUGS[@]}" ]; do
  slug="${SSH_SLUGS[$i]}"; label="${SSH_LABELS[$i]}"
  want_prog="${SSH_PROGS[$i]}"; want_opts="${SSH_OPTS[$i]}"
  log="$TMP/ssh-$slug.log"; out="$TMP/out-ssh-$slug"
  i=$((i + 1))
  if ! grep -qF 'stopped: opam required' "$out"; then
    report 1 "$label" "the hook did not reach the opam stop: $(tr '\n' '|' <"$out")"
    continue
  fi
  # Git probes an unrecognised launcher with `-G` to detect its variant; that
  # line is not the fetch invocation, so it is filtered out.
  line="$(grep -vE '(^| )-G( |$)' "$log" | head -n1)"
  if [ -z "$line" ]; then
    report 1 "$label" "no ssh launcher invocation logged (raw log: $(tr '\n' '|' <"$log"))"
    continue
  fi
  prog="$(basename "${line%% *}")"
  case "$line" in
    *"-o BatchMode=yes"*) opts=yes ;;
    *) opts=no ;;
  esac
  if [ "$prog" = "$want_prog" ] && [ "$opts" = "$want_opts" ]; then
    report 0 "$label"
  else
    report 1 "$label" "invoked '$prog' with options=$opts (wanted '$want_prog' / options=$want_opts); line: $line"
  fi
done

# ---------------------------------------------------------------------------
echo
if [ "$failures" -eq 0 ]; then
  echo "all legs passed"
  exit 0
else
  echo "$failures leg(s) FAILED"
  exit 1
fi
