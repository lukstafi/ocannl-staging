#!/usr/bin/env bash
# The fleet's run-time correctness slot around one tools/test-run.sh batch
# (gh-ocannl-1004). tools/test-run.sh's supervisor runs this in place of dune
# when the box is a fleet box (plan_slot there decided that); it resolves
# whether the batch holds a GPU, then execs
#   fleet-worker.sh execution slot --wait <s> --cpu|--gpu -- <dune> <args...>
# which takes one of the box's slots (a GPU token unless --cpu), and execs dune
# under it. So the pid the supervisor caps and signals ends up being dune's,
# exactly as without a slot, and dune's status is this script's.
#
# Usage (from the repository root): tools/fleet-slot-run.sh <fleet-worker> <wait> <dune> [DUNE ARGS...]
#
# Why the runner takes the slot at all: every wave brief had to name the
# `execution slot` wrapper and every worker had to remember it; a worker that
# forgot ran unslotted, and one that forgot `--cpu` for a cc batch held one of
# rog-nv-linux's two GPU tokens for nothing (gh-ocannl-1065). A worker that
# still wraps the runner is harmless: the fleet's nested-slot rule
# (lukstafi/ludics-lite, FLEET_SLOT_HELD) runs this batch inside the wrapper's
# slot rather than taking a second one.
#
# The kind is RESOLVED, not guessed, and fails closed. First, a stanza that
# NAMES its backend (`; ocannl-backend: cuda -- …`, which env_var_deps
# enforces on every stanza that does not read the configuration) holds that
# backend whatever the configuration says, so a run that can reach one naming
# cuda, hip or metal is --gpu (test/config's ocannl_slot_kind reads the markers
# and what the argv reaches; Test_utils.Slot_kind says how). Then, for the
# stanzas that do read it: OCANNL_BACKEND is only
# one of the places a backend comes from: an ordinary cc batch leaves it unset
# and gets cc from the config its stanzas copy. So the backend is read the way
# a test run reads it, by `ocannl_read_config` (test/config/, built from the
# same Utils resolution: config file, environment, command line), from each
# directory whose `ocannl_config` a test can read -- test/config (copied by
# every test/* directory and by bin/) and arrayjit/test (its own). `--cpu` is
# declared only when every answer names a CPU backend. No backend at all is
# NOT cc: `Context.auto` then tries metal, cuda and hip first. A GPU backend,
# an unrecognized or empty name, a build or read that fails: `--gpu`, the
# slot's own default. So is a dune argv that can pick a backend the configs do
# not show: `exec`, whose program may choose its own, or a command-line
# `--ocannl_backend` naming anything but a CPU backend (the command line
# outranks the files). A misreading therefore costs only a wait for a GPU
# token, never a GPU batch outside the tokens.

set -u

[ $# -ge 3 ] || { echo "usage: tools/fleet-slot-run.sh <fleet-worker> <wait> <dune> [DUNE ARGS...]" >&2; exit 2; }
fw=$1 wait=$2 dune=$3
shift 3

# shellcheck source=box-jobs.sh
. tools/box-jobs.sh || { echo "fleet-slot-run: cannot read tools/box-jobs.sh" >&2; exit 126; }

# The directories whose `ocannl_config` a test run can read, relative to the
# repository root: the shared test configuration, and arrayjit's own.
SLOT_CONFIG_DIRS="test/config arrayjit/test"

# 0 iff the dune argv can select a backend the configurations do not show;
# says which on stderr.
argv_picks_backend() { # dune argv
  local a v
  if [ "${1:-}" = exec ]; then
    echo "test-run: fleet slot: dune exec runs a program that may pick its own backend: --gpu" >&2
    return 0
  fi
  for a; do
    case $a in
      --ocannl[_-]backend=*) v=${a#*=} ;;
      *ocannl[_-]backend*) v= ;;
      *) continue ;;
    esac
    box_jobs_cpu_backend "$v" && continue
    echo "test-run: fleet slot: the command line names a backend (${a}): --gpu" >&2
    return 0
  done
  return 1
}

# Prints `cpu` or `gpu`, and on stderr what it was decided from. Two questions,
# both of which must answer CPU: can the run reach a stanza that NAMES a GPU
# backend (ocannl_slot_kind, reading the `; ocannl-backend:` markers
# env_var_deps enforces -- no configuration value shows those), and do the
# configurations the other stanzas read resolve a CPU backend
# (ocannl_read_config).
resolve_kind() { # dune argv
  local reader reach d b seen=
  if [ -n "${OCANNL_TOOL_READ_CONFIG:-}" ] && [ -n "${OCANNL_TOOL_SLOT_KIND:-}" ]; then
    reader=$OCANNL_TOOL_READ_CONFIG reach=$OCANNL_TOOL_SLOT_KIND
  else
    # Built under the worktree lock the supervisor holds, before the slot:
    # a few seconds warm, and their libraries are the batch's own anyway.
    "$dune" build ./test/config/ocannl_read_config.exe ./test/config/ocannl_slot_kind.exe >&2 ||
      { echo "test-run: fleet slot: the backend readers did not build, so the kind is unread: --gpu" >&2
        echo gpu; return 0; }
    reader=$PWD/_build/default/test/config/ocannl_read_config.exe
    reach=$PWD/_build/default/test/config/ocannl_slot_kind.exe
  fi
  b=$("$reach" "$@" 2>/dev/null)
  case $b in
    cpu) ;;
    gpu:*) echo "test-run: fleet slot: ${b#gpu: }: --gpu" >&2; echo gpu; return 0 ;;
    *) echo "test-run: fleet slot: ocannl_slot_kind answered '${b:-nothing}': --gpu" >&2; echo gpu; return 0 ;;
  esac
  for d in $SLOT_CONFIG_DIRS; do
    if ! b=$(cd "$d" && "$reader" --read=backend --output=stdout 2>/dev/null); then
      echo "test-run: fleet slot: the backend for $d is unreadable: --gpu" >&2
      echo gpu
      return 0
    fi
    box_jobs_cpu_backend "$b" ||
      { echo "test-run: fleet slot: $d resolves backend=${b:-<none>, so Context.auto tries GPUs first}: --gpu" >&2
        echo gpu; return 0; }
    seen="${seen:+$seen, }$d: $b"
  done
  echo "test-run: fleet slot: every test configuration resolves a CPU backend ($seen): --cpu" >&2
  echo cpu
}

if argv_picks_backend "$@"; then
  kind=gpu
else
  kind=$(resolve_kind "$@")
fi
exec "$fw" execution slot --wait "$wait" "--$kind" -- "$dune" "$@"
