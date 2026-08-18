#!/usr/bin/env bash
# Automated repo-wide formatting sweep (the CLAUDE.md "Formatting" policy).
#
# Agents and humans type in their own style during feature work; this script
# periodically reformats the whole repository in standalone commits, so feature
# diffs stay free of style hunks. A repo-wide formatting commit conflicts with
# every branch in flight, so the sweep only proceeds during a quiet period:
#   - run from the main checkout, on master, clean, fast-forwardable to
#     origin/master (the script ff-pulls);
#   - no open PRs (checked with gh -- fails closed if gh is unreachable);
#   - no other worktree that is dirty or has commits ahead of origin/master.
#
# Formatting cannot change behavior, but it can break goldens that pin source
# LOCATIONS: %expect blocks and .expected files embedding file:line shift when
# formatting moves lines, and promoting those expectations rewrites source that
# may itself need reformatting. So the sweep iterates format -> @check ->
# runtest -> promote to a fixed point (a few rounds), VALIDATING each
# promotion: with location tokens masked out, promoted files must be unchanged
# -- anything else (a value regression, platform drift, nondeterminism) aborts
# rather than being blessed into the sweep commit. Non-convergence aborts too;
# its one known benign cause is an ocamlformat-hostile golden missing from
# .ocamlformat-ignore (the sweep formats it, its test's promotion reverts it,
# forever) -- ppx-expectation files (test/ppx/*_expected.ml) must all be listed
# there.
#
# One sweep runs at a time per checkout (an atomic lock under _build), and
# EXIT/INT/TERM traps reset the checkout to its pre-sweep state on any failure
# or interruption -- a dead sweep must not leave master dirty or ahead, which
# would wedge every later scheduled run at the entry gates.
#
# Each sweep lands as two commits on master: the reformat itself, then a
# one-liner recording the reformat's SHA in .git-blame-ignore-revs (GitHub's
# blame view honors that file; locally, run once:
#   git config blame.ignoreRevsFile .git-blame-ignore-revs).
# If origin/master moves while the sweep runs, or the quiet period ends, or
# the push fails, the sweep is dropped (reset to the pre-sweep base) rather
# than merged or left behind -- it is cheap to regenerate.
#
# Usage: tools/format-sweep.sh [--force] [--no-push]
#   --force    skip the quiet-period gates (open PRs, other worktrees); the
#              on-master/clean/in-sync requirements still apply
#   --no-push  land the commits locally but do not push (for testing the sweep)

set -euo pipefail

FORCE=0
NO_PUSH=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --no-push) NO_PUSH=1 ;;
    *) echo "format-sweep: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
# Physical path: `git worktree list` reports physical paths, and the self-skip
# comparison below must survive being invoked through a symlink (or Git Bash's
# /c/ vs C:/ forms -- both sides are computed by the shell here).
ROOT=$(pwd -P)
MAX_ITER=4

# A gate failure before we touch anything: report and leave the tree alone.
die() { echo "format-sweep: $*" >&2; exit 1; }
# A failure after formatting started: the EXIT trap resets the checkout.
abort() { echo "format-sweep: $*" >&2; exit 1; }

# Acquire tools/test-run.sh's per-worktree flock on fd 7 and HOLD it: the
# lock lives on the open file description, which bash's fd 7 keeps alive
# after the probing perl exits (test-run's own take_lock uses this same
# exec-then-flock(STDIN) pattern). Every source rewrite in this script runs
# under the held lock, so a test-run invocation arriving mid-rewrite refuses
# loudly (its designed behavior) instead of starting dune under the rewrite;
# the lock is released around the sweep's own test phases, which take it
# themselves. Waits up to just above test-run's 3600s cap; returns 1 past it.
# $1 (optional): wait bound in seconds; defaults to just above test-run's
# 3600s cap. cleanup passes a short bound instead -- a scheduler's
# TERM-to-KILL grace would not survive a long wait there.
hold_test_run_lock() {
  local bound=${1:-3900} rc=0
  exec 7>>"$ROOT/.test-run.lock" || return 1
  if ! perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&7; then
    # The waiter runs as a background child under an interruptible `wait`:
    # a foreground perl would defer INT/TERM for the whole wait (see
    # run_phase below for the same pattern and why).
    echo "format-sweep: waiting for the active tools/test-run.sh run to finish..." >&2
    perl -e 'use Fcntl ":flock"; alarm '"$bound"'; $SIG{ALRM} = sub { exit 1 };
             exit(flock(STDIN, LOCK_EX) ? 0 : 1)' <&7 &
    PHASE_PID=$!
    wait "$PHASE_PID" || rc=$?
    PHASE_PID=""
    [ "$rc" -eq 0 ] || { exec 7>&- || true; return 1; }
  fi
  # Clear stale ownership metadata now that WE hold the flock: a completed
  # earlier phase leaves .owner naming its finished run, and test-run's
  # `stop` would otherwise attribute our held lock to that run and TERM its
  # recorded process group -- which the sweep shares, i.e. kill the sweep.
  # With the metadata gone, stop refuses to reap and points a human here.
  rm -f "$ROOT/.test-run.lock.owner" "$ROOT/.test-run.lock.launcher"
}
# NB: no redirection on `exec 7>&-`: exec with only redirections applies them
# to the REST OF THE SCRIPT, so a onetime `2>/dev/null` here once swallowed
# every later stderr message, abort messages included (closing an unopened fd
# is already silent).
release_test_run_lock() { exec 7>&- || true; }

# Content hash of every tracked difference from HEAD (staged and unstaged).
tree_hash() { git diff HEAD | git hash-object --stdin; }

# The tracked tree may only change in the sweep's own rewrite steps. A
# mismatch here means someone edited the main checkout mid-sweep; committing
# it (git add -u) would smuggle unrelated work into the formatting commit,
# and resetting would DESTROY that work -- so leave everything in place and
# abort; the entry gates then block further sweeps until a human resolves it.
assert_tree_unchanged() {
  [ "$(tree_hash)" = "$SNAP_TREE" ] || {
    KEEP=1 # disarm cleanup's reset: it would destroy the foreign edits
    abort "tracked files changed outside the sweep's own rewrites (mid-sweep edit in the main checkout?); leaving the tree as is -- resolve it before the next sweep"
  }
}

# One sweep at a time per checkout: a second invocation racing the first past
# the clean-tree gate would format/restore the sources underneath the first
# one's running dune. mkdir is the portable atomic lock (same reason
# tools/sweep.sh locks whole runs, not just the test-run.sh command lock).
mkdir -p "$ROOT/_build"
LOCKDIR="$ROOT/_build/format-sweep.lock"
mkdir "$LOCKDIR" 2>/dev/null \
  || die "another format-sweep is running (or a stale lock after a hard kill: $LOCKDIR)"

# Any exit that did not deliberately keep its result (KEEP=1: pushed, or
# --no-push retained) resets the checkout to BASE (once BASE is recorded) --
# a sweep killed by the scheduler or failed mid-way must not leave master
# dirty or ahead, or every later scheduled run dies at the entry gates. The
# reset itself must not rewrite sources under a live dune either: when the
# abort CAME from an external test-run grabbing the lock ahead of one of the
# sweep's own test phases, cleanup waits for that run before resetting; past
# the bound it resets anyway, since leaving master dirty wedges everything.
KEEP=0
BASE=""
cleanup() {
  if [ -n "$BASE" ] && [ "$KEEP" = "0" ]; then
    # Hold the test-run lock THROUGH the reset -- releasing between waiting
    # and resetting would let a fresh run start dune exactly then. Short
    # bound: a KILL chasing the TERM must not find us still waiting here.
    hold_test_run_lock 60 \
      || echo "format-sweep: warning: resetting without the test-run lock (held past the bound)" >&2
    git reset -q --hard "$BASE"
    release_test_run_lock
  fi
  rmdir "$LOCKDIR" 2>/dev/null
}
trap cleanup EXIT

# Run a test phase as a background child and wait for it: bash defers signal
# traps while a foreground child runs, so a TERM arriving mid-phase would
# otherwise sit until the phase ends -- past any scheduler's TERM-then-KILL
# grace, and a KILL runs no cleanup at all. `wait` IS interruptible, so the
# trap fires promptly, relays the TERM to the phase child, and exits into
# cleanup (which waits on the dying run's flock before resetting). Sets RC.
PHASE_PID=""
run_phase() {
  RC=0
  tools/test-run.sh run "$@" &
  PHASE_PID=$!
  wait "$PHASE_PID" || RC=$?
  PHASE_PID=""
}
on_sig() {
  [ -n "$PHASE_PID" ] && kill -TERM "$PHASE_PID" 2>/dev/null
  exit 130
}
trap on_sig INT TERM

# --- Quiet-period gates -----------------------------------------------------

# The dynamic gates (open PRs, other worktrees) are a function because they run
# twice: before the sweep, and again right before the push -- the test rounds
# take long enough for a PR to open or a worktree to wake up in between.
quiet_gates() {
  local open_prs busy wt
  open_prs=$(gh pr list --state open --json number --jq 'length') \
    || { echo "format-sweep: gh failed (auth/network?); failing closed -- use --force to override" >&2; return 1; }
  [ "$open_prs" = "0" ] \
    || { echo "format-sweep: $open_prs open PR(s); not a quiet period" >&2; return 1; }

  busy=""
  while IFS= read -r wt; do
    # Canonicalize before comparing with ROOT: a stale/foreign path fails the
    # cd and is skipped; the sweep's own checkout must be skipped by identity,
    # or the final recheck would see its own sweep commits as "ahead".
    wt=$(cd "$wt" 2>/dev/null && pwd -P) || continue
    [ "$wt" = "$ROOT" ] && continue
    if [ -n "$(git -C "$wt" status --porcelain)" ]; then
      busy="$busy  $wt (dirty)\n"
    elif [ "$(git -C "$wt" rev-list --count origin/master..HEAD)" != "0" ]; then
      busy="$busy  $wt (commits ahead of origin/master)\n"
    fi
  done < <(git worktree list --porcelain | sed -n 's/^worktree //p')
  [ -z "$busy" ] || {
    echo "format-sweep: active worktree(s); not a quiet period:"
    printf '%b' "$busy"
    return 1
  } >&2
  return 0
}

# The toolchain may not be on a scheduled process's PATH, and MSYS shells need
# opam-env.sh's path rewriting even when dune is discoverable (AGENTS.md; same
# pattern as tools/promote.sh, strengthened for Windows).
case "$(uname -s 2>/dev/null)" in
  MINGW* | MSYS* | CYGWIN*) . tools/opam-env.sh ;;
  *) command -v dune >/dev/null 2>&1 || . tools/opam-env.sh ;;
esac

# ocamlformat is a dev tool, not a package dependency: `opam install .
# --deps-only` does not provision it, so a fresh host would otherwise fail
# @fmt in a way that reads as non-convergence. Require it up front, at the
# exact version .ocamlformat pins (ocamlformat refuses other versions).
# tr -d '\r' on both sides: ocamlformat.exe emits CRLF on Windows, and an
# autocrlf checkout gives .ocamlformat CRLF line endings; command
# substitution strips only the trailing LF.
OCF_PINNED=$(sed -n 's/^version *= *//p' .ocamlformat | tr -d '\r')
command -v ocamlformat >/dev/null 2>&1 \
  || die "ocamlformat is not installed; run: opam install ocamlformat.$OCF_PINNED"
OCF_HAVE=$(ocamlformat --version 2>/dev/null | tr -d '\r' || true)
[ "$OCF_HAVE" = "$OCF_PINNED" ] \
  || die "ocamlformat $OCF_HAVE does not match the .ocamlformat pin $OCF_PINNED; run: opam install ocamlformat.$OCF_PINNED"

BRANCH=$(git symbolic-ref --quiet --short HEAD || echo "(detached)")
[ "$BRANCH" = "master" ] || die "not on master (on $BRANCH); the sweep only runs on master"
# Require the MAIN checkout, not master checked out in a linked worktree: a
# nested linked worktree without its generated dune-workspace would make the
# unqualified dune commands root at the parent checkout and sweep the wrong
# tree. The main checkout is the one whose git-dir IS the common git-dir.
[ "$(git rev-parse --path-format=absolute --git-dir)" = "$(git rev-parse --path-format=absolute --git-common-dir)" ] \
  || die "not the main checkout (master is checked out in a linked worktree); the sweep only runs from the main checkout"
[ -z "$(git status --porcelain)" ] || die "working tree not clean"

# Never rewrite sources under a live dune: at entry, refuse while a
# tools/test-run.sh run is active in this checkout (probe its per-worktree
# flock, the same way its own lock_held does); once underway, every source
# rewrite -- each fmt round, the promotion, the cleanup reset -- runs WITH
# the flock held (hold_test_run_lock above), so a test-run invocation
# arriving mid-rewrite refuses loudly instead of starting dune under it.
# The lock cannot be held across the sweep's OWN test phases -- they go
# through test-run.sh, whose take_lock opens its own file description and
# would refuse against ours -- so it is released around them and re-acquired
# (waiting for any run that slipped in) before the next rewrite.
if [ -e "$ROOT/.test-run.lock" ] \
  && perl -e 'use Fcntl ":flock";
              open(my $fh, ">>", $ARGV[0]) or exit 1;
              exit(flock($fh, LOCK_EX | LOCK_NB) ? 1 : 0)' "$ROOT/.test-run.lock" 2>/dev/null; then
  die "a tools/test-run.sh run is active in this checkout; not formatting under it"
fi

git fetch origin
# --ff-only alone is not enough: merging an OLDER origin/master into a local
# master that is ahead succeeds as "already up to date", and the end-of-sweep
# race check would then reset --hard over the unpushed local commits. Require
# strict equality after the ff-pull. The ff-pull rewrites tracked sources
# when origin is ahead, so it too runs under the held test-run lock.
hold_test_run_lock \
  || die "a tools/test-run.sh run held its lock past the bound; giving up"
git merge --ff-only origin/master >/dev/null \
  || { release_test_run_lock; die "master diverged from origin/master; resolve that first"; }
release_test_run_lock
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)" ] \
  || die "master has local commits not on origin/master; push them first"

if [ "$FORCE" -eq 0 ]; then
  quiet_gates || die "quiet-period gate failed (above)"

  STALE=$(git branch -r --no-merged origin/master | grep -v ' -> ' || true)
  [ -z "$STALE" ] && : || echo "format-sweep: note: unmerged remote branches (no open PR, so proceeding):
$STALE" >&2
fi

BASE=$(git rev-parse HEAD) # arms cleanup()'s reset from here on
SNAP_TREE=$(tree_hash)     # clean at BASE; every rewrite re-baselines it

# --- Format / test / promote to a fixed point --------------------------------

fmt_clean() { dune build @fmt >/dev/null 2>&1; }

iter=0
while :; do
  iter=$((iter + 1))
  [ "$iter" -le "$MAX_ITER" ] || abort "no fixed point after $MAX_ITER rounds; \
likely an ocamlformat-hostile golden missing from .ocamlformat-ignore (see header)"
  hold_test_run_lock \
    || abort "a tools/test-run.sh run held its lock past the bound; giving up"
  # The lock wait can be long; a foreign edit made during it must not be
  # swept into the formatting about to run.
  assert_tree_unchanged
  echo "format-sweep: round $iter: formatting"

  if ! fmt_clean; then
    dune fmt >/dev/null 2>&1 || true # promotes; exit status is not informative
    fmt_clean || abort "dune fmt did not converge (ocamlformat error? run 'dune build @fmt')"
  fi

  if [ "$iter" -eq 1 ] && [ -z "$(git status --porcelain)" ]; then
    KEEP=1
    echo "format-sweep: repository already formatted; nothing to do"
    exit 0
  fi
  release_test_run_lock # our own test phases take the lock themselves
  SNAP_TREE=$(tree_hash) # the test phases must not move this

  # test-run exits 2 on a lock refusal (an external run grabbed the lock in
  # the gap after our wait) -- report that as the benign race it is, not as a
  # test failure the operator would then hunt for.
  # After each test phase: FIRST check the tree did not move (that abort
  # disarms cleanup's reset, preserving a foreign edit -- which may even be
  # what made the phase fail), and only then interpret the exit code, whose
  # abort paths reset.
  echo "format-sweep: round $iter: compiling (@check)"
  run_phase build @check
  assert_tree_unchanged
  [ "$RC" -ne 2 ] || abort "test-run refused the @check phase (external run started mid-sweep); rerun later"
  [ "$RC" -eq 0 ] || abort "@check failed after formatting"

  echo "format-sweep: round $iter: running the regular test suite"
  run_phase runtest
  assert_tree_unchanged
  [ "$RC" -ne 2 ] || abort "test-run refused the runtest phase (external run started mid-sweep); rerun later"
  if [ "$RC" -eq 0 ]; then
    break
  fi

  # Formatting can only break goldens that pin source LOCATIONS (%expect
  # blocks and .expected files embedding file:line); promote them and go
  # around again -- the promoted source may need reformatting, which shifts
  # lines once more. The snapshots hash CONTENT (git diff), not `status
  # --porcelain`: a promotion that rewrites a file formatting already
  # modified leaves the status lines identical. tools/promote.sh rather than
  # plain `dune promote`, so a sweep run from Windows Git Bash does not copy
  # CRLF into the goldens.
  hold_test_run_lock \
    || abort "a tools/test-run.sh run held its lock past the bound; giving up"
  assert_tree_unchanged # same long-wait window as the pre-fmt hold
  # -u, not -A: the baseline must not stage untracked artifacts a failing
  # test may have left, or they would ride the final `git add -u` unnoticed.
  git add -u # stage the post-format pre-promote state, the validation baseline
  SNAP_BEFORE=$(tree_hash)
  tools/promote.sh || abort "test suite failed and promotion errored"
  SNAP_AFTER=$(tree_hash)
  [ "$SNAP_BEFORE" != "$SNAP_AFTER" ] \
    || abort "test suite failed with nothing to promote -- a real failure, not golden drift"

  # Validate the promotion: with source-location tokens masked out, each
  # promoted file must be UNCHANGED. Anything else -- a value regression, a
  # platform difference, nondeterminism -- is a real failure that a blanket
  # promote would silently bless into the sweep commit (the gh-ocannl-601
  # trap), so the sweep aborts instead.
  # Mask ONLY the numeric coordinates, keeping filenames and keywords:
  # formatting can move a location within its file, never to another file,
  # so "a.ml:3" -> "b.ml:3" must NOT compare equal after masking.
  strip_locs() {
    sed -E -e 's|([[:alnum:]_./-]+\.mli?):[0-9]+(:[0-9]+)?|\1:<LOC>|g' \
      -e 's|([Ll]ines?) [0-9]+([-,][0-9]+)?|\1 <LOC>|g' \
      -e 's|characters [0-9]+-[0-9]+|characters <LOC>|g'
  }
  BAD=""
  while IFS= read -r f; do
    if ! cmp -s <(git show ":$f" | strip_locs) <(strip_locs < "$f"); then
      BAD="$BAD $f"
    fi
  done < <(git diff --name-only)
  [ -z "$BAD" ] \
    || abort "promotion changed more than source locations in:$BAD -- a real output change, not formatting drift"
  SNAP_TREE=$SNAP_AFTER # the promoted state is the next pre-rewrite baseline
  release_test_run_lock # the next round's test phases take it themselves
done

if [ -z "$(git status --porcelain)" ]; then
  KEEP=1
  echo "format-sweep: repository already formatted; nothing to do"
  exit 0
fi

# --- Land -------------------------------------------------------------------

# A non-ignored untracked file appearing during the sweep is anomalous: some
# test process left an artifact outside _build. Committing it (-A) would be
# wrong, and retaining it past a successful push would leave the checkout
# dirty and wedge every later run at the clean-tree gate -- so abort and
# leave the stray in place for inspection (the trap resets only tracked
# state; the entry gate keeps failing loudly until someone looks).
STRAYS=$(git status --porcelain | sed -n 's/^?? //p')
[ -z "$STRAYS" ] \
  || abort "untracked file(s) appeared during the sweep; inspect and remove them: $STRAYS"
assert_tree_unchanged # nothing may have moved since the passing round's fmt

echo "format-sweep: committing"
# -u, not -A: formatting and promotion only modify tracked files, and -A
# would swallow any non-ignored stray a test process left in the tree.
git add -u
git commit -q -m "Automated formatting sweep

Produced by tools/format-sweep.sh: repo-wide dune fmt plus the %expect
re-promotions its line shifts require. Recorded in .git-blame-ignore-revs
by the follow-up commit.

Co-authored-by: OCANNL format sweep <format-sweep@users.noreply.github.com>"
SWEEP_SHA=$(git rev-parse HEAD)

# `date +%F`, not `date -I`: BSD date on the supported macOS hosts has no -I.
printf '\n# %s formatting sweep\n%s\n' "$(date +%F)" "$SWEEP_SHA" >> .git-blame-ignore-revs
git add .git-blame-ignore-revs
git commit -q -m "Record formatting sweep in .git-blame-ignore-revs

Co-authored-by: OCANNL format sweep <format-sweep@users.noreply.github.com>"

if [ "$NO_PUSH" -eq 1 ]; then
  KEEP=1
  echo "format-sweep: --no-push: sweep committed locally as $SWEEP_SHA (+ blame-ignore commit); not pushing"
  exit 0
fi

# Recheck the dynamic gates: a PR opened or a worktree woken mid-sweep is
# exactly the conflict the quiet-period policy exists to prevent. On any
# failure from here on, the EXIT trap drops the sweep (reset to BASE) -- it is
# cheap to regenerate, and leaving the commits behind would wedge every later
# run at the in-sync gate.
if [ "$FORCE" -eq 0 ] && ! quiet_gates; then
  die "quiet period ended during the sweep; dropped the sweep (rerun later)"
fi

git fetch origin
[ "$(git rev-parse origin/master)" = "$BASE" ] \
  || die "origin/master moved during the sweep; dropped the sweep (rerun later)"
git push origin master || die "push failed; dropped the sweep (rerun later)"
KEEP=1
echo "format-sweep: pushed $SWEEP_SHA"
