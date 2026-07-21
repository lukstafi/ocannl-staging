# Bring the opam switch environment into any POSIX shell:
#
#   . tools/opam-env.sh          # or: source tools/opam-env.sh
#
# Motivation: on Windows, `opam env --shell=sh` emits cygwin-style paths
# (`/cygdrive/c/...`, plus `/usr/...` entries relative to opam's bundled cygwin
# root). Git Bash (MSYS) mounts drives at `/c` and has its own `/usr`, so
# eval'ing that output verbatim leaves the toolchain half-broken — dune finds
# the wrong `cygpath`/mingw tools and linking fails with errors like
# `cygpath: error converting ... -lpthread`. This script rewrites those
# prefixes for MSYS before eval'ing, so plain Git Bash sessions (including
# non-interactive ones that never saw a profile-primed environment) can build.
#
# Costs one `opam env` call; safe to source multiple times.

if ! command -v opam >/dev/null 2>&1; then
  echo "tools/opam-env.sh: opam not found on PATH" >&2
  return 1 2>/dev/null || exit 1
fi

_opam_env="$(opam env --shell=sh)" || {
  echo "tools/opam-env.sh: 'opam env' failed" >&2
  return 1 2>/dev/null || exit 1
}

case "$(uname -o 2>/dev/null)" in
  Msys)
    # Drive mounts: /cygdrive/c/... -> /c/...
    # Switch-internal cygwin paths: /usr/... -> <opam root>/.cygwin/root/usr/...
    _opam_cygroot="$(cygpath -u "$(opam var root)")/.cygwin/root"
    _opam_env="$(printf '%s\n' "$_opam_env" \
      | sed -e "s|:/usr/|:$_opam_cygroot/usr/|g" -e "s|/cygdrive/|/|g")"
    ;;
esac

eval "$_opam_env"
unset _opam_env _opam_cygroot
