#!/usr/bin/env bash

# The executable half of the pin-revisions composite action. It lives outside
# action.yml so tools/test-pin-revisions.sh can exercise the exact production
# code with hermetic opam and git fixtures.

set -euo pipefail

hash12() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | cut -c1-12
  else
    shasum -a 256 | cut -c1-12
  fi
}

# setup-ocaml updates opam-repository before this action. Ask opam's solver for
# the self-contained package set selected for every project package with the
# same test/doc dependency flags as the install steps. This keys the actual
# ordinary-package versions rather than a version-specific detail of opam's
# repository storage.
opam_files=(./*.opam)
[ -e "${opam_files[0]}" ] \
  || { echo "no project opam files found for dependency solution" >&2; exit 1; }
package_names=$(printf '%s\n' "${opam_files[@]}" \
  | sed 's#^\./##; s#\.opam$##' \
  | LC_ALL=C sort -u)
request=$(printf '%s\n' "$package_names" | paste -sd, -)
solution=$(opam --cli=2.1 list --safe --color=never --resolve="$request" \
  --with-test --with-doc --columns=package --short \
  | LC_ALL=C sort -u)
[ -n "$solution" ] \
  || { echo "opam dependency solution was empty" >&2; exit 1; }
echo "Resolved opam package solution:"
printf '%s\n' "$solution" | sed 's/^/  /'
# The digest also covers the solved packages' definitions as the switch holds
# them, so a metadata-only change to a selected version (a new patch, a depext)
# still invalidates the key. The project's own packages are left out of that
# part: the caller pins them from the checkout, and opam records the checkout's
# git ref in the pinned definition -- `#master` on a branch, `#HEAD` on the
# detached checkout every pull_request run gets -- so hashing them keyed the
# cache by EVENT TYPE. Every pull request missed the switch master had just
# saved for the same tree and paid the full dependency build; the fetch that
# then died on a rolled upstream archive checksum is what surfaced it
# (gh-ocannl-889). Their content is already in the callers' keys through
# `hashFiles('*.opam')`.
definition_packages=()
project_packages=
while IFS= read -r package; do
  name=${package%%.*}
  if printf '%s\n' "$package_names" | grep -qxF -- "$name"; then
    project_packages="$project_packages $name"
  else
    definition_packages+=("$package")
  fi
done <<< "$solution"
# Without this guard an all-project solution would run `opam show` with no
# package argument, which describes the current directory instead of failing.
[ "${#definition_packages[@]}" -gt 0 ] \
  || { echo "opam dependency solution holds only project packages" >&2; exit 1; }
echo "Project packages left out of the definition digest:$project_packages"
solution_digest=$(
  {
    printf '%s\n' "$solution"
    opam --cli=2.1 show --safe --color=never --raw --sort "${definition_packages[@]}"
  } | hash12
)
echo "solution-digest=$solution_digest" >>"$GITHUB_OUTPUT"

# `--cli=2.1` fixes the table format this parser consumes across opam upgrades.
# Local project pins are reported as git+file:// URLs; omit them because the
# source checkout already selects their revision and a dependency cache must
# not miss on every project commit. Every remote git pin, whether explicit or
# introduced by pin-depends, comes from the registry populated by the caller's
# preceding `opam pin -n` steps. LC_ALL=C keeps the digest stable across runner
# locales.
specs=$(opam --cli=2.1 pin list --safe --color=never \
  | sed -n 's/.* \(git+[^[:space:]]*\).*/\1/p' \
  | sed '/^git+file:\/\//d' \
  | LC_ALL=C sort -u)
# An empty list means the registry extraction stopped matching (or the caller
# ran this too early). Never collapse that defect into the digest of an empty
# string: that would silently freeze the cache key.
[ -n "$specs" ] || { echo "no remote git pins found in opam pin registry" >&2; exit 1; }
echo "Derived remote git pin specs:"
printf '%s\n' "$specs" | sed 's/^/  /'
shas=
while IFS= read -r spec; do
  spec=${spec#git+}
  url=${spec%%#*}
  ref=${spec#*#}
  [ "$ref" != "$spec" ] || ref=HEAD
  # `awk` rather than `cut | head`: an annotated tag answers on two lines (the
  # tag and its peeled commit), and a `head` that leaves early can hand its
  # upstream a SIGPIPE, which `pipefail` would read as a failed resolution. awk
  # consumes the whole answer and prints one line.
  sha=$(git ls-remote "$url" "$ref" | awk 'NR==1{print $1}')
  # An empty sha would silently collapse the key onto whatever it hashed to
  # last time, serving a cache built from different sources.
  [ -n "$sha" ] || { echo "could not resolve $ref of $url" >&2; exit 1; }
  # Recording the exact revisions in the log is a second reason to run this at
  # all: nothing else currently reports them.
  echo "$url#$ref -> $sha"
  shas=$shas$sha
done <<< "$specs"
# macOS ships shasum, not sha256sum; hash12 handles both runners.
echo "digest=$(printf %s "$shas" | hash12)" >>"$GITHUB_OUTPUT"
