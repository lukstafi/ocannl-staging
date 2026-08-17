#!/bin/bash
# gh-612 measurement cells: the gh-573 / gh-574 HIP legs against one re-established denominator.
# Driver for benchmarks/report-gh612-hip.md. Every number in that report comes from these
# subcommands, so the report quotes invocations rather than restating commands — a transcribed
# command drifts from the one that ran, and four of this session's review findings were exactly
# that drift.
#
# Usage:
#   gh612_cells.sh search  <tree> <label> <rep> [extra ocannl flags...]
#   gh612_cells.sh snap    <tree> <label> <rep> [extra ocannl flags...]
#   gh612_cells.sh profile <tree> <label> <rep> [n-harness-runs]
#   gh612_cells.sh finger  <label> <rep>
#   gh612_cells.sh sweep   <tree> <reps> <cap>...      # balanced cap order, see below
#   gh612_cells.sh roofline <tree>
#
# <tree> is a built checkout (dune build @check bin/ benchmarks/) whose
# benchmarks/fixtures/gpt2_mini.safetensors is the SAME file in every tree under comparison —
# symlink one fixture rather than regenerating per tree, so the input cannot differ.
#
# Results land under $OUT_ROOT/<label>/r<rep> (default /tmp/gh612).
set -u
# Every subcommand `cd`s into a checkout, so any caller-supplied path that has to survive that cd
# must be absolutized HERE, before the first cd — a relative OUT_ROOT (or <tree>) would otherwise be
# created next to the invocation directory and then re-resolved inside the checkout, and the first
# redirection into it fails. FIXTURE is the deliberate exception: it is relative to <tree>/benchmarks,
# which is exactly where the runner is invoked from.
#
# And no path may be produced by a command substitution that yields an EMPTY string on failure:
# `rm -rf "$OUT_ROOT/$label/r$rep"` with an empty OUT_ROOT is `rm -rf /<label>/r<rep>`. `set -u` does
# not catch that (the variable is set, just empty), so resolution ABORTS instead of falling through.
resolve_dir() {                      # absolute path to an existing usable directory, or abort
  local d=$1 abs
  mkdir -p "$d" 2>/dev/null || { echo "gh612_cells: cannot create directory: $d" >&2; exit 2; }
  abs=$(cd "$d" 2>/dev/null && pwd) || abs=""
  [ -n "$abs" ] && [ -d "$abs" ] || { echo "gh612_cells: not a usable directory: $d" >&2; exit 2; }
  printf '%s\n' "$abs"
}
require_dir() {                      # like resolve_dir but must already exist (never creates)
  local d=$1 abs
  abs=$(cd "$d" 2>/dev/null && pwd) || abs=""
  [ -n "$abs" ] && [ -d "$abs" ] || { echo "gh612_cells: no such directory: $d" >&2; exit 2; }
  printf '%s\n' "$abs"
}
# NOTE: an `exit` inside $( ) exits only the substitution's subshell, so the abort has to be
# re-asserted at the call site. Every use of resolve_dir/require_dir is followed by this check.
OUT_ROOT=$(resolve_dir "${OUT_ROOT:-/tmp/gh612}")
case ${OUT_ROOT:-} in /?*) ;; *) echo "gh612_cells: OUT_ROOT unusable, refusing to run" >&2; exit 2;; esac
PIN=${PIN:-taskset -c 0-15}
FIXTURE=${FIXTURE:-fixtures/gpt2_mini.safetensors}
ARCH=${ARCH:-gfx1151}
EXE=../_build/default/benchmarks/runners/ocannl/bench_gpt.exe
ROUTINE=cross_entropy_loss_fwd

# The autotune cache dir is per label AND per rep, never shared. Two independent reasons, both
# load-bearing: the cache key omits the Numerics policy (gh-ocannl-568), and a warm cache makes an
# A/B vacuous by replaying the other arm's crowned schedule (report-gh481-cuda.md measured
# compile_s 1.76 s warm against 29 s cold, with the "winner" being whatever the other arm found).
cell_dir() { echo "$OUT_ROOT/$1/r$2"; }

cmd_search() { (
  local tree=$1 label=$2 rep=$3; shift 3
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  local out; out=$(cell_dir "$label" "$rep")
  rm -rf "$out"; mkdir -p "$out"            # mkdir BEFORE any redirection into it
  cd "$tree/benchmarks" || exit 1
  local t0=$SECONDS
  BENCH_FIXTURE=$FIXTURE BENCH_TUNE=1 $PIN "$EXE" --ocannl_backend=hip \
    --ocannl_autotune_cache_dir="$out/cache" \
    --ocannl_autotune_log=true --ocannl_schedule_log_declines=true "$@" \
    > "$out/search.out" 2> "$out/search.err"
  local st=$?
  echo "$label r$rep: exit $st, $((SECONDS - t0))s"
  grep -h '^{' "$out/search.out" | tail -1
  # The three claim-bearing lines: the deterministic untuned baseline, the crowned artifact per
  # arm, and which arm shipped.
  grep -E 'untuned-default pipeline|winner replay ok|tune_placements: winner' "$out/search.err"
  grep -oh 'finer_fission [a-z]*' "$out"/cache/*.sexp 2>/dev/null | sort -u
) }

# Warm replay of the cell's cached winner, capturing the emitted source and the launch geometry.
# Arm A compiles first and arm B overwrites the same path, so snapshot by polling on content.
cmd_snap() { (
  local tree=$1 label=$2 rep=$3; shift 3
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  local out; out=$(cell_dir "$label" "$rep")
  local snap="$out/armsnap"; rm -rf "$snap"; mkdir -p "$snap"
  cd "$tree/benchmarks" || exit 1
  rm -rf build_files
  local f=build_files/bench_gpt/${ROUTINE}__seg.hip
  ( while :; do
      if [ -f "$f" ]; then h=$(md5sum "$f" 2>/dev/null | cut -d' ' -f1)
        [ -n "$h" ] && [ ! -f "$snap/$h.hip" ] && cp "$f" "$snap/$h.hip"; fi
      sleep 0.02
    done ) & local w=$!
  BENCH_FIXTURE=$FIXTURE BENCH_TUNE=1 $PIN "$EXE" --ocannl_backend=hip \
    --ocannl_autotune_cache_dir="$out/cache" \
    --ocannl_output_debug_files_in_build_directory=true \
    --ocannl_schedule_log_launches=true "$@" \
    > "$out/replay.out" 2> "$out/launches.err"
  local st=$?
  kill $w 2>/dev/null; wait $w 2>/dev/null
  echo "$label r$rep replay: exit $st"
  pick_armA "$out"
) }

# Arm A is the FIRST fissioned compile of the routine in the launch log; the `seg i/N` totals are
# the real fission widths. Do NOT read the count off an `F_saved[fine N segs]` label -- that N is
# the number of saved per-segment PLACEMENT ENTRIES, and on this workload it reads 77 where the
# emitted arm holds 136 kernels.
#
# Then validate before selecting. A content-polling watcher can copy a partially written file, and
# a truncated capture can already carry every `__global__` line while its last function body is
# incomplete -- so the kernel count alone does not identify a usable snapshot, and glob order is
# hash order, i.e. arbitrary. Require balanced braces AND a clean compile, which is the check
# report-gh569-hip.md used and the only one that rules out a torn tail.
pick_armA() {
  local out=$1
  local n; n=$(grep -o "$ROUTINE seg 0/[0-9]*" "$out/launches.err" | awk -F/ '$2>1{print $2; exit}')
  [ -n "$n" ] || { echo "no fissioned compile in $out/launches.err" >&2; return 1; }
  local f
  for f in "$out"/armsnap/*.hip; do
    [ "$(grep -c '__global__' "$f")" = "$n" ] || continue
    python3 - "$f" <<'EOF' || continue
import sys
s=open(sys.argv[1]).read()
sys.exit(0 if s.count('{') == s.count('}') and s.count('{') > 0 else 1)
EOF
    hipcc --offload-arch="$ARCH" -O2 -include hip/hip_runtime.h -c -o /dev/null "$f" \
      2>/dev/null || { echo "  $(basename "$f"): $n kernels but INCOMPLETE (no compile)" >&2; continue; }
    echo "$f" > "$out/armA.path"
    echo "  arm A = $f ($n kernels, braces balanced, compiles)"
    return 0
  done
  echo "no complete snapshot with $n kernels in $out/armsnap" >&2; return 1
}

cmd_profile() { (
  local tree=$1 label=$2 rep=$3 n=${4:-3}
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  local out; out=$(cell_dir "$label" "$rep")
  local src; src=$(cat "$out/armA.path")
  # Clear this subcommand's OWN artifact set before regenerating it. Every producing subcommand here
  # does that (`search` rm -rf's the cell, `snap` the snapshot dir); `profile` writing a
  # caller-chosen NUMBER of files is the case where forgetting it is silent rather than obvious --
  # a later run with a smaller count leaves the earlier run's CSVs behind and every consumer
  # medians over a mixture of two profiles.
  rm -f "$out"/kernels-*.csv "$out"/kernels-*.err "$out"/bucket-*.txt
  cd "$tree/benchmarks" || exit 1
  python3 gpt2_kernel_harness.py --source "$src" --launches "$out/launches.err" \
          --out "$out/harness.hip" || exit 1
  hipcc --offload-arch="$ARCH" -O2 -o "$out/harness" "$out/harness.hip" || exit 1
  local i
  for i in $(seq "$n"); do
    $PIN "$out/harness" > "$out/kernels-$i.csv" 2> "$out/kernels-$i.err"
    # stderr's last line is the sum-vs-step validation the report quotes. Pair it against THIS
    # cell's own step p50 (search.out / replay.out): another rep is a different crowned artifact,
    # so pairing across reps compares a profile to a step it is not a profile of.
    tail -1 "$out/kernels-$i.err"
    python3 gpt2_bucket.py --source "$src" --stats "$out/kernels-$i.csv" --steps 1 \
            > "$out/bucket-$i.txt" 2>&1
  done
  echo "  paired step p50 for this cell:"
  grep -ho '"p50":[0-9.]*' "$out/search.out" "$out/replay.out" 2>/dev/null
) }

# The acceptance fingerprints, read off the emitted source. Kernel parameter lists span multiple
# lines, so this cannot be done with a line-oriented grep -- `[^)]*` hits the newline before it
# ever sees the closing paren and every pattern silently matches nothing. Parse with re.S, as
# gpt2_kernel_harness.py does.
cmd_finger() {
  local out; out=$(cell_dir "$1" "$2")
  # `|| return` on the COMMAND line: absent fingerprints must not look like passing ones. Without
  # it the geometry dump below still succeeds and `finger` exits 0 having printed no evidence.
  python3 - "$(cat "$out/armA.path")" "$out" <<'EOF' || return 1
import re,sys,csv,statistics,collections
src=open(sys.argv[1]).read(); out=sys.argv[2]
SIG=re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+__seg(\d+))\s*\(([^)]*)\)', re.S)
sigs={int(i):[" ".join(p.split()).split()[-1].lstrip("*") for p in ps.split(",") if p.strip()]
      for _,i,ps in SIG.findall(src)}
# DISCOVER the CSV set rather than assuming how many runs `profile` was asked for: it takes a
# caller-chosen count, so a hard-coded 1..3 silently drops runs above 3 and, before `profile`
# started clearing its own outputs, could median across two different profiles.
import glob, os
t=collections.defaultdict(list)
csvs=sorted(glob.glob(os.path.join(out,"kernels-*.csv")))
if not csvs: sys.exit(f"no kernels-*.csv in {out} -- run `profile` first")
for f in csvs:
    for r in csv.DictReader(open(f)):
        t[int(r["Name"].rsplit("__seg",1)[1])].append(float(r["TotalDurationNs"])/1e6)
ms={i:statistics.median(v) for i,v in t.items()}
print(f"medians over {len(csvs)} harness run(s): {', '.join(os.path.basename(f) for f in csvs)}")
def show(title, pred):
    hits=[i for i in sorted(sigs) if pred(sigs[i])]
    tot=sum(ms.get(i,0) for i in hits)
    print(f"{title}: n={len(hits)} {tot:.3f} ms")
    for i in hits:
        pre=[n for n in sigs[i] if re.fullmatch(r'l\d+_ffn_b2', n)]
        print(f"  seg{i:<4} {ms.get(i,0):7.4f} ms  params={len(sigs[i]):2} ffn_b2_prefix={len(pre)}"
              f"  {', '.join(sigs[i])}")
# gh-573: the LayerNorm prefix must be BOUNDED and must RESET, not ramp with depth.
show("LayerNorm sites", lambda ns: any(n.startswith(("gamma_","beta_")) for n in ns))
# gh-574: the lm_head GEMM alone, the row-max its own kernel -- not one fused segment.
show("lm_head / CE tail", lambda ns: any(n in ("wte","logits","max_logits") for n in ns))
show("QK^T sites", lambda ns: any(n.endswith("_q") for n in ns) and any(n.endswith("_k") for n in ns))
EOF
  echo "--- launch geometry (arm A) ---"
  local n; n=$(grep -o "$ROUTINE seg 0/[0-9]*" "$out/launches.err" | awk -F/ '$2>1{print $2; exit}')
  grep -E "$ROUTINE seg [0-9]+/$n " "$out/launches.err" \
    | sed -E 's/.*(seg [0-9]+).*grid=(\[[^]]*\]).*block=(\[[^]]*\]).*/\1 grid=\2 block=\3/'
}

# Cap sweep with BALANCED order: rep r runs the cap list forward on even r and reversed on odd, so
# no cap is permanently earlier in the session than another. Without this, cap 4 always precedes
# cap 8 and thermal or driver drift is indistinguishable from the cap's effect -- the confound
# report-gh481-cuda.md's arm-order rule exists for.
cmd_sweep() {
  local tree=$1 reps=$2; shift 2
  tree=$(require_dir "$tree")        # BEFORE the loop: each cell must see the same absolute tree
  case ${tree:-} in /?*) ;; *) return 2;; esac
  local caps=("$@") r cap ordered
  for r in $(seq "$reps"); do
    if [ $((r % 2)) -eq 1 ]; then ordered=("${caps[@]}")
    else ordered=(); for ((i=${#caps[@]}-1; i>=0; i--)); do ordered+=("${caps[$i]}"); done; fi
    echo "=== sweep rep $r, order: ${ordered[*]} ==="
    for cap in "${ordered[@]}"; do
      cmd_search "$tree" "sweep-cap$cap" "$r" --ocannl_virtualize_max_inline_fanin="$cap" \
        || { echo "gh612_cells: cap $cap rep $r failed; aborting the sweep rather than reporting a" \
                  "half-balanced series" >&2; return 1; }
    done
  done
  echo "NOTE: read the untuned column and the arm A KERNEL count first. A cap whose kernel count"
  echo "      matches cap -1's is not losing a trade-off, it is never firing."
}

cmd_roofline() { (
  local tree=$1
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  cd "$tree" || exit 1                      # roofline_hip.cpp path is relative to the TREE ROOT
  hipcc --offload-arch="$ARCH" -O3 -o "$OUT_ROOT/roofline" benchmarks/roofline_hip.cpp \
        -I/opt/rocm/include -L/opt/rocm/lib -lrocblas || exit 1
  # CPU quiet: the bandwidth leg shares the LPDDR5X controller with it on this APU.
  $PIN "$OUT_ROOT/roofline"
) }

sub=${1:-}; shift || true
case $sub in
  search)   cmd_search   "$@" ;;
  snap)     cmd_snap     "$@" ;;
  profile)  cmd_profile  "$@" ;;
  finger)   cmd_finger   "$@" ;;
  sweep)    cmd_sweep    "$@" ;;
  roofline) cmd_roofline "$@" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
