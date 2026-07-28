#!/usr/bin/env bash
# Bring the OCANNL development environment up, incrementally.
#
#   scripts/setup-ocaml-env.sh          # check and self-heal; seconds, never compiles
#   scripts/setup-ocaml-env.sh --deps   # additionally install/refresh opam dependencies
#
# Every step inspects the current state and acts only when something is missing
# or broken, so running it repeatedly is safe and each run advances the setup by
# whatever is still outstanding.
#
# This runs as a SessionStart hook (.claude/settings.json), which is why the
# default mode is cheap: it queries opam, repairs the cygwin pkgconf breakage
# described below, and reports what is left, but never builds anything. Pass
# --deps when you actually want the (10-20 minute) dependency install.
#
# Steps that would install software are only taken automatically in Claude Code
# cloud sessions (CLAUDE_CODE_REMOTE=true), where the machine is disposable. On
# a real workstation they are reported as suggestions instead — this script does
# not install opam or create switches behind your back.
#
# Note: this file used to restore a pre-built switch from a GitHub release. That
# "fast setup" machinery was reverted in 9cf51f51 and the release it referenced
# no longer exists, so that path is gone; docs/fast-setup.md describes the
# abandoned design.

set -u

MODE_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --deps|--full) MODE_DEPS=1 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup-ocaml-env.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

# Only cloud sessions may install things unattended.
if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then MAY_INSTALL=1; else MAY_INSTALL=0; fi

status=0
ok()   { printf '  ok    %s\n' "$*"; }
fixed(){ printf '  fixed %s\n' "$*"; }
todo() { printf '  todo  %s\n' "$*"; status=1; }
fail() { printf '  FAIL  %s\n' "$*" >&2; status=1; }

echo "=== OCANNL environment ==="

# --- worktree dune root --------------------------------------------------
# Claude Code creates worktrees under `.claude/worktrees/`, i.e. INSIDE the
# repository. Dune takes as its root the outermost ancestor holding a
# `dune-workspace` (failing that, a `dune-project`) and ignores dot-directories,
# so from such a worktree the main checkout wins and the worktree is invisible to
# dune: targeted commands fail with "Don't know about directory
# .claude/worktrees/...", and — the quiet one — a bare `dune build` / `dune
# runtest` builds and tests the PARENT checkout instead of the branch you are on.
#
# A `dune-workspace` at the worktree root makes the worktree the root again and
# gives it its own `_build`. That only holds while the main checkout has none
# (outermost wins), which is why the file is generated per worktree and
# gitignored rather than committed. This step comes before the opam section: it
# needs no toolchain, and a missing opam exits the script early below.
self_top="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
wt_top="$(git -C "${self_top:-.}" rev-parse --show-toplevel 2>/dev/null || true)"
git_common="$(git -C "${self_top:-.}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
# <main>/.git for the main checkout and for every worktree of it; the inequality
# confirms the suffix was really stripped before the prefix tests below.
main_top="${git_common%/.git}"
if [ -n "$wt_top" ] && [ -n "$main_top" ] && [ "$main_top" != "$git_common" ]; then
  case "$wt_top" in
    "$main_top") ;;   # the main checkout itself: dune already roots here
    "$main_top"/*)    # a worktree nested inside the main checkout
      if [ -e "$main_top/dune-workspace" ]; then
        fail "dune-workspace in $main_top shadows this worktree's — dune will build the parent checkout"
      elif [ -e "$wt_top/dune-workspace" ]; then
        ok "worktree dune root"
      else
        lang="$(grep -m1 '^(lang dune ' "$wt_top/dune-project" 2>/dev/null || true)"
        if printf '%s\n' "${lang:-(lang dune 3.18)}" >"$wt_top/dune-workspace"; then
          fixed "worktree dune root (dune-workspace written; dune would have built the parent checkout)"
        else
          fail "could not write $wt_top/dune-workspace"
        fi
      fi
      ;;
    *) ;;             # worktree outside the repo: nothing shadows it
  esac
fi

# --- opam itself ---------------------------------------------------------
if command -v opam >/dev/null 2>&1; then
  ok "opam $(opam --version)"
elif [ "$MAY_INSTALL" = 1 ]; then
  echo "  ...   installing opam"
  if bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)" -- --no-backup >/dev/null 2>&1; then
    fixed "opam $(opam --version)"
  else
    fail "could not install opam"
  fi
else
  todo "opam is not on PATH — see https://opam.ocaml.org/doc/Install.html"
fi

command -v opam >/dev/null 2>&1 || { echo "=== stopped: opam required ==="; exit $status; }

# --- opam root -----------------------------------------------------------
OPAM_ROOT_RAW="$(opam var root 2>/dev/null || true)"
if [ -z "$OPAM_ROOT_RAW" ]; then
  if [ "$MAY_INSTALL" = 1 ]; then
    echo "  ...   initialising opam"
    opam init -y --bare --disable-sandboxing >/dev/null 2>&1 \
      && fixed "opam root initialised" || fail "opam init failed"
    OPAM_ROOT_RAW="$(opam var root 2>/dev/null || true)"
  else
    todo "opam root not initialised — run: opam init --bare"
  fi
else
  ok "opam root $OPAM_ROOT_RAW"
fi

# opam prints a native path; the rest of this script needs a POSIX one.
if command -v cygpath >/dev/null 2>&1 && [ -n "$OPAM_ROOT_RAW" ]; then
  OPAM_ROOT="$(cygpath -u "$OPAM_ROOT_RAW")"
else
  OPAM_ROOT="$OPAM_ROOT_RAW"
fi

# --- switch --------------------------------------------------------------
# Any switch will do as long as it satisfies the packages' `ocaml >= 5.3.0`; the
# repo does not mandate a local `_opam`, and creating one unasked would be a
# surprising, expensive side effect.
ocaml_version="$(opam exec -- ocamlopt -version 2>/dev/null || true)"
if [ -z "$ocaml_version" ]; then
  todo "no usable switch — run: opam switch create . 5.3.0 --no-install"
else
  case "$ocaml_version" in
    5.[3-9]*|5.[1-9][0-9]*|[6-9]*) ok "ocaml $ocaml_version" ;;
    *) todo "ocaml $ocaml_version is below the required 5.3.0" ;;
  esac
fi

# --- cygwin pkgconf ------------------------------------------------------
# Cygwin's pkgconf 3.0.4-1 (2026-07-26) broke `pkgconf --personality=<triplet>`:
# personality files are only looked up in /usr/lib/pkgconfig/personality.d while
# the package installs them to /etc/pkgconfig/personality.d, and even once loaded
# the personality's DefaultSearchPaths are ignored. Every opam conf-mingw-w64-*
# package is exactly that invocation, so dependency installs fail outright.
#
# The probe is behavioural rather than a version comparison, so this repairs a
# broken pkgconf whatever its version and goes quiet by itself once cygwin ships
# a fix. Restoring 2.5.1 needs libpkgconf7 too: pkgconf.exe links
# cygpkgconf-7.dll, and 3.0.4 only brought in cygpkgconf-8.dll.
CYGROOT="${OPAM_ROOT:+$OPAM_ROOT/.cygwin/root}"
if [ -n "${CYGROOT:-}" ] && [ -x "$CYGROOT/bin/pkgconf.exe" ]; then
  personality_ok() {
    "$CYGROOT/bin/pkgconf.exe" --personality=x86_64-w64-mingw32 --dump-personality 2>/dev/null \
      | grep -q '^Triplet: x86_64-w64-mingw32'
  }
  # Pick a tar that understands zstd: Git Bash's GNU tar shells out to a zstd
  # binary that is not installed, while Windows' bundled bsdtar has it built in.
  zstd_tar() {
    for candidate in "${SYSTEMROOT:-/c/Windows}/System32/tar.exe" /c/Windows/System32/tar.exe tar; do
      [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1 || continue
      if "$candidate" -tf "$1" >/dev/null 2>&1; then echo "$candidate"; return 0; fi
    done
    return 1
  }

  if personality_ok; then
    ok "cygwin pkgconf personality ($("$CYGROOT/bin/pkgconf.exe" --version 2>&1))"
  else
    echo "  ...   repairing cygwin pkgconf (personality lookup broken)"
    work="$(mktemp -d)"
    base="https://cygwin.mirror.constant.com/x86_64/release/pkgconf"
    repaired=1
    for pkg in pkgconf-2.5.1-1-x86_64 libpkgconf7/libpkgconf7-2.5.1-1-x86_64; do
      file="$work/$(basename "$pkg").tar.zst"
      curl -sSLf -o "$file" "$base/$pkg.tar.zst" || { repaired=0; break; }
      tarbin="$(zstd_tar "$file")" || { repaired=0; break; }
      "$tarbin" -xf "$file" -C "$work" || { repaired=0; break; }
    done
    if [ "$repaired" = 1 ] && [ -d "$work/usr/bin" ]; then
      # Cygwin mounts /usr/bin at <root>/bin and nothing reads a physical
      # <root>/usr/bin, so the payload is copied rather than unpacked over the root.
      cp -f "$work"/usr/bin/* "$CYGROOT/bin/" || repaired=0
    else
      repaired=0
    fi
    rm -rf "$work"
    if [ "$repaired" = 1 ] && personality_ok; then
      fixed "cygwin pkgconf restored to $("$CYGROOT/bin/pkgconf.exe" --version 2>&1)"
    else
      fail "could not repair cygwin pkgconf; conf-mingw-w64-* packages will not install"
    fi
  fi
fi

# --- pins ----------------------------------------------------------------
# Kept in step with .github/workflows/ci.yml; see the comments there for why each
# one is pinned. `opam pin -n` is only invoked for pins that are actually absent,
# so a repeat run neither re-resolves nor reports spurious changes.
pinned="$(opam pin list --short 2>/dev/null || true)"
# grep -qx, not a substring test: these lists are newline-separated, and a
# substring match would also let `printbox` satisfy a check for `printbox-text`.
pin_if_missing() {
  if printf '%s\n' "$pinned" | grep -qx "$1"; then
    ok "pin $1"
  elif opam pin -n "$1" "$2" >/dev/null 2>&1; then
    fixed "pin $1"
  else
    fail "could not pin $1"
  fi
}
pin_if_missing printbox-text https://github.com/c-cube/printbox.git
pin_if_missing pcre https://github.com/mmottl/pcre-ocaml.git
pin_if_missing dataprep https://github.com/lukstafi/ocaml-dataprep.git

# --- dependencies --------------------------------------------------------
# A representative sample rather than a full solve: `opam install --dry-run`
# would be authoritative but costs a solver run on every session start.
installed="$(opam list --installed --short 2>/dev/null || true)"
missing=""
for pkg in base ppxlib ppx_minidebug dataprep; do
  printf '%s\n' "$installed" | grep -qx "$pkg" || missing="$missing $pkg"
done

if [ -z "$missing" ]; then
  ok "dependencies present"
elif [ "$MODE_DEPS" = 1 ]; then
  echo "  ...   installing dependencies (this takes 10-20 minutes)"
  if opam install . -y --deps-only --with-test --with-doc; then
    fixed "dependencies installed"
  else
    fail "opam install failed"
  fi
else
  todo "missing:$missing — re-run with --deps to install"
fi

# --- verification --------------------------------------------------------
dune_version="$(opam exec -- dune --version 2>/dev/null || true)"
[ -n "$dune_version" ] && ok "dune $dune_version" || todo "dune not available"

# Claude Code picks up environment for subsequent commands from this file.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  opam env --shell=sh >>"$CLAUDE_ENV_FILE" 2>/dev/null && ok "environment persisted"
fi

if [ "$status" = 0 ]; then echo "=== ready ==="; else echo "=== incomplete (see 'todo' above) ==="; fi
# A hook must not fail the session; report state through the summary instead.
exit 0
