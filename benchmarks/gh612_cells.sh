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
OUT_ROOT=${OUT_ROOT:-/tmp/gh612}
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

cmd_search() {
  local tree=$1 label=$2 rep=$3; shift 3
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
}

# Warm replay of the cell's cached winner, capturing the emitted source and the launch geometry.
# Arm A compiles first and arm B overwrites the same path, so snapshot by polling on content.
cmd_snap() {
  local tree=$1 label=$2 rep=$3; shift 3
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
}

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

cmd_profile() {
  local tree=$1 label=$2 rep=$3 n=${4:-3}
  local out; out=$(cell_dir "$label" "$rep")
  local src; src=$(cat "$out/armA.path")
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
}

# The acceptance fingerprints, read off the emitted source. Kernel parameter lists span multiple
# lines, so this cannot be done with a line-oriented grep -- `[^)]*` hits the newline before it
# ever sees the closing paren and every pattern silently matches nothing. Parse with re.S, as
# gpt2_kernel_harness.py does.
cmd_finger() {
  local out; out=$(cell_dir "$1" "$2")
  python3 - "$(cat "$out/armA.path")" "$out" <<'EOF'
import re,sys,csv,statistics,collections
src=open(sys.argv[1]).read(); out=sys.argv[2]
SIG=re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+__seg(\d+))\s*\(([^)]*)\)', re.S)
sigs={int(i):[" ".join(p.split()).split()[-1].lstrip("*") for p in ps.split(",") if p.strip()]
      for _,i,ps in SIG.findall(src)}
t=collections.defaultdict(list)
for k in (1,2,3):
    try: rows=list(csv.DictReader(open(f"{out}/kernels-{k}.csv")))
    except OSError: continue
    for r in rows: t[int(r["Name"].rsplit("__seg",1)[1])].append(float(r["TotalDurationNs"])/1e6)
ms={i:statistics.median(v) for i,v in t.items()}
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
  local caps=("$@") r cap ordered
  for r in $(seq "$reps"); do
    if [ $((r % 2)) -eq 1 ]; then ordered=("${caps[@]}")
    else ordered=(); for ((i=${#caps[@]}-1; i>=0; i--)); do ordered+=("${caps[$i]}"); done; fi
    echo "=== sweep rep $r, order: ${ordered[*]} ==="
    for cap in "${ordered[@]}"; do
      cmd_search "$tree" "sweep-cap$cap" "$r" --ocannl_virtualize_max_inline_fanin="$cap"
    done
  done
  echo "NOTE: read the untuned column and the arm A KERNEL count first. A cap whose kernel count"
  echo "      matches cap -1's is not losing a trade-off, it is never firing."
}

cmd_roofline() {
  local tree=$1
  cd "$tree" || exit 1                      # roofline_hip.cpp path is relative to the TREE ROOT
  hipcc --offload-arch="$ARCH" -O3 -o "$OUT_ROOT/roofline" benchmarks/roofline_hip.cpp \
        -I/opt/rocm/include -L/opt/rocm/lib -lrocblas || exit 1
  # CPU quiet: the bandwidth leg shares the LPDDR5X controller with it on this APU.
  $PIN "$OUT_ROOT/roofline"
}

mkdir -p "$OUT_ROOT"
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
