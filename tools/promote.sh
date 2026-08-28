#!/usr/bin/env bash
# Promote dune test outputs and normalize line endings in the promoted goldens.
#
# `dune promote` on Windows copies CRLF test output into `.expected` files
# (test exe stdout is text-mode); goldens are LF in the repo (.gitattributes).
# This wrapper promotes and then strips trailing CRs from the promoted files,
# replacing the manual `sed -i 's/\r$//'` ritual. It also works from worktrees
# nested inside the repo, where plain `dune promote` resolves the PARENT
# checkout: `dune promotion apply` accepts the `--root .` override.
#
# It also guards the mid-merge promotion trap (staging PR #487, ~90 minutes):
# promotion writes the WORKING TREE, but `git commit` during a merge takes the
# INDEX, so a promotion made after `git add` is committed as the pre-promotion
# content. Nothing local complains -- every `dune runtest` reads the working
# tree and passes -- while CI builds the committed tree and fails on the golden
# diff. After promoting, this script stages what it promoted (and says so), or
# names the files it would not stage. Outside a merge it is a no-op.
#
# Usage: tools/promote.sh [FILES...]
#   Run from anywhere; extra arguments are passed through to dune (e.g. paths
#   of specific files to promote).

set -eu
cd "$(dirname "$0")/.."

command -v dune >/dev/null 2>&1 || . tools/opam-env.sh

# Are we mid-merge? `git rev-parse --verify MERGE_HEAD` rather than testing
# `.git/MERGE_HEAD`: in a linked worktree `.git` is a FILE, and MERGE_HEAD
# lives in the per-worktree gitdir that only git can resolve. A non-repository
# cwd answers "no" here, which is the right answer for the guard.
merging=0
if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
  merging=1
fi

# The promotion list has to be taken BEFORE applying -- afterwards there is
# nothing pending left to name. It is only needed for the guard, so outside a
# merge the extra dune invocation is skipped entirely. `list` filters its
# arguments exactly as `apply` does, prints one root-relative path per line on
# stdout, and sends "Nothing to promote for X." to stderr.
promoted=""
if [ "$merging" -eq 1 ]; then
  promoted="$(dune promotion list --root . "$@" 2>/dev/null || true)"
fi

dune promotion apply --root . "$@"

# Strip trailing CRs from a promoted golden. perl -i, not sed -i: BSD sed
# (macOS) requires a backup-suffix argument for -i, so GNU-style `sed -i`
# errors there; perl is portable (and already a dependency of
# tools/test-run.sh).
strip_crs() { # strip_crs FILE -- no-op unless FILE is a golden we pin to LF
  case "$1" in
    *.expected | test/ppx/*_expected.ml) ;;
    *) return 0 ;;
  esac
  [ -f "$1" ] && perl -i -pe 's/\r$//' "$1"
  return 0
}

# Any promoted golden that now differs from the index.
git diff --name-only -z -- '*.expected' 'test/ppx/*_expected.ml' \
  | while IFS= read -r -d '' f; do
      strip_crs "$f"
    done

[ "$merging" -eq 1 ] || exit 0

# Mid-merge: stage what was promoted, so the commit carries it.
#
# A file still UNMERGED in the index is left alone: `git add` on one records a
# resolution, and whether this promotion is that resolution is the caller's
# call, not the script's. Everything else -- already resolved, or an entirely
# new golden -- is staged, which is what closes the trap.
staged=()
unmerged=()
while IFS= read -r f; do
  [ -n "$f" ] || continue
  # Promotion may have introduced CRs into a file the `git diff` pass above
  # could not see (a new golden, absent from the index), so re-check here;
  # strip_crs is idempotent and the staged content must be LF either way.
  strip_crs "$f"
  if [ -n "$(git ls-files --unmerged -- "$f")" ]; then
    unmerged+=("$f")
  else
    staged+=("$f")
  fi
done <<EOF
$promoted
EOF

if [ ${#staged[@]} -gt 0 ]; then
  # Reported, never fatal. The promotion has already been applied by this
  # point, so exiting here on a `git add` that will not take a path (an ignored
  # golden, say) would leave exactly the state this guard exists to prevent,
  # and leave it SILENTLY -- `set -e` prints nothing.
  if add_err="$(git add -- "${staged[@]}" 2>&1)"; then
    printf '\npromote.sh: mid-merge, so staged what it promoted:\n' >&2
    printf '  %s\n' "${staged[@]}" >&2
    printf 'A merge commit takes the index, not the working tree, so without this the\n' >&2
    printf 'promotion would be dropped from the commit and fail in CI only.\n' >&2
  else
    printf '\npromote.sh: WARNING -- promoted, but `git add` REFUSED to stage:\n' >&2
    printf '  %s\n' "${staged[@]}" >&2
    printf '%s\n' "$add_err" >&2
    printf 'A merge commit takes the index, so until these are staged the promotion\n' >&2
    printf 'is dropped from the commit and fails in CI only.\n' >&2
  fi
fi

if [ ${#unmerged[@]} -gt 0 ]; then
  printf '\npromote.sh: WARNING -- promoted, but still UNMERGED, so NOT staged:\n' >&2
  printf '  %s\n' "${unmerged[@]}" >&2
  printf 'Staging one of these records it as the conflict resolution, which is yours\n' >&2
  printf 'to decide. But a merge commit takes the index, so until you do, the\n' >&2
  printf 'promotion is dropped from the commit and fails in CI only. To accept:\n' >&2
  printf '  git add --' >&2
  printf ' %s' "${unmerged[@]}" >&2
  printf '\n' >&2
fi
