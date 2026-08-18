#!/usr/bin/env bash
# Automated repo-wide formatting sweep (the CLAUDE.md "Formatting" policy).
#
# Agents and humans type in their own style during feature work; this script
# periodically reformats the whole repository in standalone commits, so feature
# diffs stay free of style hunks. A repo-wide formatting commit conflicts with
# every branch in flight, so the sweep only proceeds during a quiet period:
#   - run from the main checkout, on master, clean, fast-forwardable to
#     origin/master (the script ff-pulls);
#   - no open PRs (checked with gh — fails closed if gh is unreachable);
#   - no other worktree that is dirty or has commits ahead of origin/master.
#
# Formatting cannot change behavior, but it can break goldens that pin source
# text: %expect blocks capturing file:line backtraces shift when formatting
# moves lines, and promoting those expectations rewrites source that may itself
# need reformatting. So the sweep iterates format -> @check -> runtest ->
# promote to a fixed point (a few rounds), and aborts + reverts if it does not
# converge. Non-convergence has one known cause worth checking first: an
# ocamlformat-hostile golden missing from .ocamlformat-ignore (the sweep
# formats it, its test's promotion reverts it, forever) — ppx-expectation
# files (test/ppx/*_expected.ml) must all be listed there.
#
# Each sweep lands as two commits on master: the reformat itself, then a
# one-liner recording the reformat's SHA in .git-blame-ignore-revs (GitHub's
# blame view honors that file; locally, run once:
#   git config blame.ignoreRevsFile .git-blame-ignore-revs).
# If origin/master moves while the sweep runs, the sweep is dropped
# (reset to origin/master) rather than merged — it is cheap to regenerate.
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
ROOT=$(pwd)
MAX_ITER=4

# A gate failure before we touch anything: report and leave the tree alone.
die() { echo "format-sweep: $*" >&2; exit 1; }
# A failure after formatting started: revert our changes, then report.
abort() {
  echo "format-sweep: $*" >&2
  echo "format-sweep: reverting formatting changes (git restore .)" >&2
  git restore .
  exit 1
}

# --- Quiet-period gates -----------------------------------------------------

BRANCH=$(git symbolic-ref --quiet --short HEAD || echo "(detached)")
[ "$BRANCH" = "master" ] || die "not on master (on $BRANCH); the sweep only runs on master"
[ -z "$(git status --porcelain)" ] || die "working tree not clean"

git fetch origin
git merge --ff-only origin/master \
  || die "master has local commits not on origin/master; resolve that first"

if [ "$FORCE" -eq 0 ]; then
  OPEN_PRS=$(gh pr list --state open --json number --jq 'length') \
    || die "gh failed (auth/network?); failing closed — use --force to override"
  [ "$OPEN_PRS" = "0" ] || die "$OPEN_PRS open PR(s); not a quiet period"

  BUSY=""
  while IFS= read -r wt; do
    [ "$wt" = "$ROOT" ] && continue
    [ -d "$wt" ] || continue
    if [ -n "$(git -C "$wt" status --porcelain)" ]; then
      BUSY="$BUSY  $wt (dirty)\n"
    elif [ "$(git -C "$wt" rev-list --count origin/master..HEAD)" != "0" ]; then
      BUSY="$BUSY  $wt (commits ahead of origin/master)\n"
    fi
  done < <(git worktree list --porcelain | sed -n 's/^worktree //p')
  [ -z "$BUSY" ] || die "active worktree(s); not a quiet period:
$(printf '%b' "$BUSY")"

  STALE=$(git branch -r --no-merged origin/master | grep -v ' -> ' || true)
  [ -z "$STALE" ] && : || echo "format-sweep: note: unmerged remote branches (no open PR, so proceeding):
$STALE" >&2
fi

BASE=$(git rev-parse HEAD)

# --- Format / test / promote to a fixed point --------------------------------

fmt_clean() { dune build @fmt >/dev/null 2>&1; }

iter=0
while :; do
  iter=$((iter + 1))
  [ "$iter" -le "$MAX_ITER" ] || abort "no fixed point after $MAX_ITER rounds; \
likely an ocamlformat-hostile golden missing from .ocamlformat-ignore (see header)"
  echo "format-sweep: round $iter: formatting"

  if ! fmt_clean; then
    dune fmt >/dev/null 2>&1 || true # promotes; exit status is not informative
    fmt_clean || abort "dune fmt did not converge (ocamlformat error? run 'dune build @fmt')"
  fi

  if [ "$iter" -eq 1 ] && [ -z "$(git status --porcelain)" ]; then
    echo "format-sweep: repository already formatted; nothing to do"
    exit 0
  fi

  echo "format-sweep: round $iter: compiling (@check)"
  tools/test-run.sh run build @check || abort "@check failed after formatting"

  echo "format-sweep: round $iter: running the regular test suite"
  if tools/test-run.sh run runtest; then
    break
  fi

  # Formatting can only break goldens that pin source text (%expect blocks
  # embedding file:line); promote them and go around again — the promoted
  # source may need reformatting, which shifts lines once more.
  SNAP_BEFORE=$(git status --porcelain | git hash-object --stdin)
  dune promote || abort "test suite failed and 'dune promote' errored"
  SNAP_AFTER=$(git status --porcelain | git hash-object --stdin)
  [ "$SNAP_BEFORE" != "$SNAP_AFTER" ] \
    || abort "test suite failed with nothing to promote — a real failure, not golden drift"
done

if [ -z "$(git status --porcelain)" ]; then
  echo "format-sweep: repository already formatted; nothing to do"
  exit 0
fi

# --- Land -------------------------------------------------------------------

echo "format-sweep: committing"
git add -A
git commit -q -m "Automated formatting sweep

Produced by tools/format-sweep.sh: repo-wide dune fmt plus the %expect
re-promotions its line shifts require. Recorded in .git-blame-ignore-revs
by the follow-up commit."
SWEEP_SHA=$(git rev-parse HEAD)

printf '\n# %s formatting sweep\n%s\n' "$(date -I)" "$SWEEP_SHA" >> .git-blame-ignore-revs
git add .git-blame-ignore-revs
git commit -q -m "Record formatting sweep in .git-blame-ignore-revs"

if [ "$NO_PUSH" -eq 1 ]; then
  echo "format-sweep: --no-push: sweep committed locally as $SWEEP_SHA (+ blame-ignore commit); not pushing"
  exit 0
fi

git fetch origin
if [ "$(git rev-parse origin/master)" != "$BASE" ]; then
  git reset --hard origin/master
  die "origin/master moved during the sweep; dropped the sweep (rerun later)"
fi
git push origin master
echo "format-sweep: pushed $SWEEP_SHA"
