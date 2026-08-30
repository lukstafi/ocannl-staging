#!/usr/bin/env bash
# Hermetic regression and mutation tests for the pin-revisions action.
#
# The action's production shell lives in resolve.sh rather than embedded YAML
# specifically so this harness can run that exact file. Fake `opam` output is
# shaped like the opam 2.5.2 Actions run that invalidated the old repository-
# stamp implementation; fake `git` makes revision resolution deterministic and
# keeps every leg offline. OPAMCOLOR=always is hostile on purpose.
#
# Each shipping assertion has a fault-injected twin below. A mutation must make
# the same oracle reject the subject, proving the test would go red if that
# defect were reintroduced instead of merely restating today's implementation.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/.github/actions/pin-revisions/resolve.sh"
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

TMP="$(mktemp -d "${TMPDIR:-/tmp}/pin-revisions-test.XXXXXX" 2>/dev/null)" || TMP=""
if [ -z "$TMP" ] || [ ! -d "$TMP" ]; then
  echo "could not create a temporary directory under ${TMPDIR:-/tmp}" >&2
  exit 2
fi
cleanup() {
  [ -n "$TMP" ] && [ -d "$TMP" ] && [ "$TMP" != / ] && rm -rf "$TMP"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$TMP/bin" "$TMP/project"
printf 'opam-version: "2.0"\n' >"$TMP/project/arrayjit.opam"
printf 'opam-version: "2.0"\n' >"$TMP/project/neural_nets_lib.opam"

# Recorded-output fixture. It deliberately returns an unsorted solution with a
# duplicate, the 2.5.2 pin-table shape containing both local git+file pins and
# duplicate remote pins, and ANSI wrappers unless every query says
# `--color=never`. Calls unsupported by the current CLI solution approach fail;
# in particular, `opam var root` exposes the retired repository-stamp approach.
cat >"$TMP/bin/opam" <<'FAKE_OPAM'
#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_OPAM_CALLS"

has_arg() {
  local wanted=$1 arg
  shift
  for arg in "$@"; do [ "$arg" = "$wanted" ] && return 0; done
  return 1
}
emit() {
  if [ "${OPAMCOLOR:-}" = always ] && ! has_arg --color=never "$@"; then
    while IFS= read -r line; do printf '\033[36m%s\033[0m\n' "$line"; done
  else
    cat
  fi
}

case " $* " in
  *" var root "*)
    printf '%s\n' "$FAKE_OPAM_ROOT"
    ;;
  *" pin list "*)
    has_arg --cli=2.1 "$@" && has_arg --safe "$@" \
      || { echo "unexpected opam pin list contract: $*" >&2; exit 64; }
    case "${FAKE_PIN_FIXTURE:-mixed-a}" in
      empty)
        printf '%s\n' 'package version kind target' | emit "$@"
        ;;
      mixed-a)
        printf '%s\n' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'arrayjit.dev git git+file:///checkout#deadbeef' \
          'alpha.dev git git+https://example.invalid/alpha.git' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'neural_nets_lib.dev git git+file:///checkout#deadbeef' | emit "$@"
        ;;
      mixed-b)
        printf '%s\n' \
          'alpha.dev git git+https://example.invalid/alpha.git' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'neural_nets_lib.dev git git+file:///checkout#deadbeef' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'arrayjit.dev git git+file:///checkout#deadbeef' | emit "$@"
        ;;
      *) echo "unknown pin fixture: $FAKE_PIN_FIXTURE" >&2; exit 64 ;;
    esac
    ;;
  *" list "*)
    has_arg --cli=2.1 "$@" && has_arg --safe "$@" \
      && has_arg --with-test "$@" && has_arg --with-doc "$@" \
      && has_arg --columns=package "$@" && has_arg --short "$@" \
      && has_arg --resolve=arrayjit,neural_nets_lib "$@" \
      || { echo "unexpected opam list contract: $*" >&2; exit 64; }
    printf '%s\n' beta.2.0 ocannl.dev alpha.1.0 beta.2.0 | emit "$@"
    ;;
  *" show "*)
    has_arg --cli=2.1 "$@" && has_arg --safe "$@" \
      && has_arg --raw "$@" && has_arg --sort "$@" \
      && has_arg alpha.1.0 "$@" && has_arg beta.2.0 "$@" \
      && has_arg ocannl.dev "$@" \
      || { echo "unexpected opam show contract: $*" >&2; exit 64; }
    printf '%s\n' \
      'opam-version: "2.0"' 'name: "alpha"' 'version: "1.0"' \
      'opam-version: "2.0"' 'name: "beta"' 'version: "2.0"' \
      'opam-version: "2.0"' 'name: "ocannl"' 'version: "dev"' | emit "$@"
    ;;
  *)
    echo "unsupported opam 2.5.2 fixture call: $*" >&2
    exit 64
    ;;
esac
FAKE_OPAM

cat >"$TMP/bin/git" <<'FAKE_GIT'
#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_GIT_CALLS"
[ "${1:-}" = ls-remote ] || { echo "unsupported fake git call: $*" >&2; exit 64; }
case "${2:-}" in
  https://example.invalid/alpha.git)
    [ "${FAKE_RESOLUTION:-ok}" = fail ] \
      || printf '%s\trefs/heads/main\n' 1111111111111111111111111111111111111111
    ;;
  https://example.invalid/zeta.git)
    printf '%s\trefs/heads/main\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ;;
  git+file:*|file:*)
    printf '%s\trefs/heads/local\n' cccccccccccccccccccccccccccccccccccccccc
    ;;
  '')
    # Lets the empty-registry mutant complete with a silent digest, so the loud
    # failure oracle proves the guard itself rather than fake git's refusal.
    printf '%s\trefs/heads/empty\n' ffffffffffffffffffffffffffffffffffffffff
    ;;
  *) echo "unknown fake remote: ${2:-}" >&2; exit 64 ;;
esac
FAKE_GIT
chmod +x "$TMP/bin/opam" "$TMP/bin/git"

printf '%s\n' \
  'solution-digest=b5782e031996' \
  'digest=623c3bf9bb0e' >"$TMP/expected-output"
printf '%s\n' \
  'ls-remote https://example.invalid/alpha.git HEAD' \
  'ls-remote https://example.invalid/zeta.git main' >"$TMP/expected-git-calls"

run_subject() { # run_subject SUBJECT LABEL PIN_FIXTURE RESOLUTION
  local subject=$1 label=$2 pins=$3 resolution=$4 dir="$TMP/runs/$2" status
  rm -rf "$dir"
  mkdir -p "$dir"
  : >"$dir/github-output"
  : >"$dir/opam.calls"
  : >"$dir/git.calls"
  (
    cd "$TMP/project" || exit 125
    PATH="$TMP/bin:$PATH" \
      OPAMCOLOR=always \
      FAKE_PIN_FIXTURE="$pins" \
      FAKE_RESOLUTION="$resolution" \
      FAKE_OPAM_ROOT="$TMP/opam-root" \
      FAKE_OPAM_CALLS="$dir/opam.calls" \
      FAKE_GIT_CALLS="$dir/git.calls" \
      GITHUB_OUTPUT="$dir/github-output" \
      bash "$subject" >"$dir/stdout" 2>"$dir/stderr"
  )
  status=$?
  printf '%s\n' "$status" >"$dir/status"
  return 0
}

has_escape() { # has_escape FILE...
  LC_ALL=C grep -q "$(printf '\033')" "$@"
}

oracle_happy() { # oracle_happy SUBJECT LABEL PIN_FIXTURE
  local subject=$1 label=$2 pins=${3:-mixed-a} dir="$TMP/runs/$2"
  run_subject "$subject" "$label" "$pins" ok
  [ "$(cat "$dir/status")" -eq 0 ] \
    && cmp -s "$TMP/expected-output" "$dir/github-output" \
    && cmp -s "$TMP/expected-git-calls" "$dir/git.calls" \
    && grep -q '^  alpha\.1\.0$' "$dir/stdout" \
    && grep -q '^  beta\.2\.0$' "$dir/stdout" \
    && grep -q '^  git+https://example\.invalid/alpha\.git$' "$dir/stdout" \
    && ! grep -q 'git+file:' "$dir/stdout" \
    && [ "$(grep -c -- '--color=never' "$dir/opam.calls")" -eq 3 ] \
    && ! has_escape "$dir/stdout" "$dir/stderr" "$dir/github-output"
}

oracle_deterministic() { # oracle_deterministic SUBJECT LABEL
  local subject=$1 label=$2 a="$TMP/runs/$2-a" b="$TMP/runs/$2-b"
  oracle_happy "$subject" "$label-a" mixed-a \
    && oracle_happy "$subject" "$label-b" mixed-b \
    && cmp -s "$a/github-output" "$b/github-output" \
    && cmp -s "$a/git.calls" "$b/git.calls"
}

oracle_empty_loud() { # oracle_empty_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  run_subject "$subject" "$label" empty ok
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^no remote git pins found in opam pin registry$' "$dir/stderr" \
    && ! grep -q '^digest=' "$dir/github-output"
}

oracle_resolution_loud() { # oracle_resolution_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  run_subject "$subject" "$label" mixed-a fail
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^could not resolve HEAD of https://example.invalid/alpha.git$' "$dir/stderr" \
    && ! grep -q '^digest=' "$dir/github-output"
}

if oracle_happy "$SRC" shipping-happy mixed-a; then
  report 0 "opam 2.5.2 output: exact solution and pin digests"
  report 0 "local git+file pins: excluded from resolution and digest"
  report 0 "OPAMCOLOR=always: no ANSI reaches names, URLs, or outputs"
else
  report 1 "shipping happy path" "see $TMP/runs/shipping-happy"
fi
if oracle_deterministic "$SRC" shipping-order; then
  report 0 "pin ordering and duplicates: one stable resolution order and digest"
else
  report 1 "pin ordering and duplicates" "see $TMP/runs/shipping-order-{a,b}"
fi
if oracle_empty_loud "$SRC" shipping-empty; then
  report 0 "empty pin registry: fails loudly without a pin digest"
else
  report 1 "empty pin registry" "see $TMP/runs/shipping-empty"
fi
if oracle_resolution_loud "$SRC" shipping-resolution; then
  report 0 "empty git resolution: fails loudly without a pin digest"
else
  report 1 "empty git resolution" "see $TMP/runs/shipping-resolution"
fi

mutant() { # mutant NAME AWK_PROGRAM
  local name=$1 program=$2 out="$TMP/$1.sh"
  awk "$program" "$SRC" >"$out" || return 1
  bash -n "$out" || return 1
  printf '%s' "$out"
}
expect_rejected() { # expect_rejected LABEL SUBJECT ORACLE
  local label=$1 subject=$2 oracle=$3
  if "$oracle" "$subject" "mutant-$(printf '%s' "$label" | tr ' ' '-')"; then
    report 1 "negative control: $label" "the shipping oracle accepted the mutant"
  else
    report 0 "negative control: $label"
  fi
}

local_mutant=$(mutant local-pin-filter \
  'index($0, "| sed") && index($0, "git+file:") { changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$local_mutant" ]; then
  expect_rejected "removing local-pin exclusion is detected" "$local_mutant" oracle_happy
else
  report 1 "negative control: local-pin mutant constructed"
fi

empty_mutant=$(mutant empty-registry-guard \
  'index($0, "[ -n \"$specs\" ] ||") { print "if [ -z \"$specs\" ]; then"; print "  echo \"digest=$(printf %s \\\"\\\" | hash12)\" >>\"$GITHUB_OUTPUT\""; print "  exit 0"; print "fi"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$empty_mutant" ]; then
  expect_rejected "silent empty-registry digest is detected" "$empty_mutant" oracle_empty_loud
else
  report 1 "negative control: empty-registry mutant constructed"
fi

storage_mutant=$(mutant opam-storage \
  '/^set -euo pipefail$/ { print; print "root=$(opam var root)"; print "repo_file=\"$root/repo/default/repo\""; print "stamp=$(sed -n '\''s/^stamp:.*\"\\([^\"]*\\)\".*/\\1/p'\'' \"$repo_file\")"; print "[ -n \"$stamp\" ] || exit 1"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$storage_mutant" ]; then
  expect_rejected "opam 2.5.0 repository-stamp assumption is detected" "$storage_mutant" oracle_happy
else
  report 1 "negative control: opam-storage mutant constructed"
fi

color_mutant="$TMP/opam-color.sh"
sed 's/ --color=never//g' "$SRC" >"$color_mutant"
if [ "$(grep -c -- '--color=never' "$SRC")" -eq 3 ] \
  && ! grep -q -- '--color=never' "$color_mutant" \
  && bash -n "$color_mutant"; then
  expect_rejected "removing color suppression is detected" "$color_mutant" oracle_happy
else
  report 1 "negative control: OPAMCOLOR mutant constructed"
fi

sort_mutant=$(mutant pin-sort \
  'BEGIN { in_specs=0 } /^specs=\$\(/ { in_specs=1 } in_specs && /LC_ALL=C sort -u\)/ { sub(/LC_ALL=C sort -u/, "cat"); changed++; in_specs=0 } { print } END { if (changed != 1) exit 9 }')
if [ -n "$sort_mutant" ]; then
  expect_rejected "removing pin sort/dedup is detected" "$sort_mutant" oracle_deterministic
else
  report 1 "negative control: pin-sort mutant constructed"
fi

resolution_mutant=$(mutant resolution-guard \
  'index($0, "[ -n \"$sha\" ] ||") { print "  [ -n \"$sha\" ] || sha=0000000000000000000000000000000000000000"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$resolution_mutant" ]; then
  expect_rejected "silent empty resolution is detected" "$resolution_mutant" oracle_resolution_loud
else
  report 1 "negative control: resolution mutant constructed"
fi

if [ "$failures" -ne 0 ]; then
  printf '%s pin-revisions test failure(s)\n' "$failures" >&2
  exit 1
fi
