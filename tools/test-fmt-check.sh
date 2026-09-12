#!/usr/bin/env bash

# Opposing controls for fmt-check.sh: clean output, a soft odoc warning, and a
# formatter failure must remain three different outcomes.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
subject="$script_dir/fmt-check.sh"
fixture_dir=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-test-fmt-check.XXXXXX") || exit 2
trap 'rm -rf "$fixture_dir"' EXIT

fixture="$fixture_dir/formatter"
cat >"$fixture" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  clean)
    echo "Formatting is clean"
    ;;
  warning)
    echo "Warning: Invalid documentation comment:" >&2
    echo 'File "fixture.ml", line 1, characters 0-0:' >&2
    echo "End of text is not allowed in '[...]' (code)." >&2
    ;;
  failure)
    echo "Warning: Invalid documentation comment:" >&2
    echo "Formatter diff" >&2
    exit 7
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "$fixture"

failures=0

check() {
  want=$1
  label=$2
  shift 2
  set +e
  output=$("$subject" "$fixture" "$@" 2>&1)
  got=$?
  set -e
  if [ "$got" -ne "$want" ]; then
    echo "FAIL: $label exited $got, expected $want" >&2
    printf '%s\n' "$output" >&2
    failures=$((failures + 1))
  else
    echo "PASS: $label exits $want"
  fi
}

check 0 "clean formatter output" clean
check 1 "invalid documentation warning" warning
check 7 "formatter failure status preservation" failure

exit "$failures"
