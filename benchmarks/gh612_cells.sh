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
#   gh612_cells.sh diff    <labelA> <repA> <labelB> <repB>
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
# Third path-safety finding in a row (relative paths, then empty paths, now traversal), so the gate
# lives HERE -- the single place every subcommand's paths are built -- rather than at each `rm -rf`.
# A label is caller input and feeds `rm -rf`: `../../../home/me/data` would escape OUT_ROOT entirely.
# Two independent checks, because either alone is bypassable: a character whitelist, and a
# containment test on the resolved parent.
cell_dir() {
  local label=$1 rep=$2 parent
  case $label in
    ""|*/*|*..*) echo "gh612_cells: bad label '$label' (no '/', no '..')" >&2; exit 2;;
    *[!A-Za-z0-9._-]*) echo "gh612_cells: bad label '$label' (allowed: A-Za-z0-9._-)" >&2; exit 2;;
  esac
  case $rep in ""|*[!0-9]*) echo "gh612_cells: bad rep '$rep' (digits only)" >&2; exit 2;; esac
  # Belt and braces: resolve the parent that will actually be operated on and require it to sit
  # beneath OUT_ROOT, so a whitelist gap cannot turn into a deletion outside the results tree.
  parent=$(mkdir -p "$OUT_ROOT/$label" 2>/dev/null && cd "$OUT_ROOT/$label" && pwd) || parent=""
  case ${parent:-} in
    "$OUT_ROOT"/*) ;;
    *) echo "gh612_cells: cell path escaped OUT_ROOT: '$OUT_ROOT/$label'" >&2; exit 2;;
  esac
  printf '%s\n' "$parent/r$rep"
}

cmd_search() { (
  local tree=$1 label=$2 rep=$3; shift 3
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  local out; out=$(cell_dir "$label" "$rep")
  case ${out:-} in /?*) ;; *) exit 2;; esac
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
  # The crowned arm A candidate's calibration line: analytic FLOPs and bytes at the emitted kernel
  # count. The report's constant-FLOPs check and its 528 -> 472 MB traffic figure are read from here.
  local lbl; lbl=$(sed -n '1,/arm A (default placements) best/p' "$out/search.err" \
    | grep -o 'best: [0-9.]* ms ([A-Za-z_]*\[[^]]*\]' | sed 's/.*(//')
  [ -n "${lbl:-}" ] && grep -F "calibration: $lbl" "$out/search.err" | head -1
) }

# Warm replay of the cell's cached winner, capturing the emitted source and the launch geometry.
# Arm A compiles first and arm B overwrites the same path, so snapshot by polling on content.
cmd_snap() { (
  local tree=$1 label=$2 rep=$3; shift 3
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  local out; out=$(cell_dir "$label" "$rep")
  case ${out:-} in /?*) ;; *) exit 2;; esac
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
  # A failed replay can still leave a complete source and launch log behind (it may have died after
  # compiling), so `pick_armA` succeeding proves nothing about the run. Refuse the artifact.
  [ "$st" -eq 0 ] || { echo "gh612_cells: replay failed (exit $st); refusing the snapshot" >&2; exit "$st"; }
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
  case ${out:-} in /?*) ;; *) exit 2;; esac
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
  # `profile` always measures the saved ARM A source, but the step p50 belongs to whichever arm the
  # search SHIPPED. Pairing them is only valid where arm A shipped; elsewhere the sum and the step
  # describe different routines and printing them together would invite exactly the wrong inference.
  local shipped; shipped=$(grep -ho '"shipped":"[AB]"' "$out/search.out" 2>/dev/null | head -1 | grep -o '[AB]')
  if [ "${shipped:-}" = "A" ]; then
    echo "  paired step p50 (this cell shipped arm A, so the pairing is valid):"
    grep -ho '"p50":[0-9.]*' "$out/search.out" "$out/replay.out" 2>/dev/null
  else
    echo "  NO paired step p50: this cell shipped arm ${shipped:-?}, and the profile above is arm A."
    echo "  An arm-B step time is not a validation of an arm-A kernel sum; they are different routines."
  fi
  # Every run, not just the first: Part 1 quotes run-1/2/3 shares as its stability evidence, so a
  # transcript that shows only run 1 cannot expose it.
  local bf
  for bf in "$out"/bucket-*.txt; do
    echo "--- $(basename "$bf") ---"
    sed -n '/| bucket/,/directly seeded/p' "$bf"
  done
) }

# The acceptance fingerprints, read off the emitted source. Kernel parameter lists span multiple
# lines, so this cannot be done with a line-oriented grep -- `[^)]*` hits the newline before it
# ever sees the closing paren and every pattern silently matches nothing. Parse with re.S, as
# gpt2_kernel_harness.py does.
cmd_finger() {
  local out; out=$(cell_dir "$1" "$2")
  case ${out:-} in /?*) ;; *) return 2;; esac
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
# Part 1's five-kernel table and Part 5's attention line items.
g1=lambda ns: any(re.fullmatch(r'l\d+_ffn_w1', n) for n in ns)
show("FFN GEMM1 + gelu", g1)
show("q/k/v projections", lambda ns: any(re.fullmatch(r'w_[qkv]_l\d+', n) for n in ns))
show("out projections", lambda ns: any(re.fullmatch(r'w_o_l\d+', n) for n in ns))
lm=lambda ns: "wte" in ns and any(n.endswith("_layer_norm") for n in ns)
five=[i for i in sigs if g1(sigs[i]) or lm(sigs[i])]
tot=sum(ms.values())
print(f"the five kernels (4x FFN GEMM1 + lm_head): {sum(ms.get(i,0) for i in five):.3f} ms "
      f"= {100*sum(ms.get(i,0) for i in five)/tot:.1f}% of {tot:.3f} ms")
# Part 1's launch-geometry census, aggregated by RESIDENT BLOCK COUNT -- the quantity that made the
# gh-569 story legible and the one Part 5 turns on.
geo={}
for m in re.finditer(r'seg (\d+)/(\d+) grid=\[(\d+);(\d+);(\d+)\] block=\[(\d+);(\d+);(\d+)\]',
                     open(os.path.join(out,"launches.err")).read()):
    if int(m.group(2))!=len(sigs): continue
    g=[int(m.group(i)) for i in (3,4,5)]; b=[int(m.group(i)) for i in (6,7,8)]
    geo[int(m.group(1))]=(g[0]*g[1]*g[2], b[0]*b[1]*b[2])
agg=collections.defaultdict(lambda:[0,0.0])
for i in sigs:
    if i in geo: agg[geo[i][0]][0]+=1; agg[geo[i][0]][1]+=ms.get(i,0.0)
print("blocks | kernels | ms | share")
for blk,(n,t) in sorted(agg.items()):
    print(f"  {blk:5} | {n:3} | {t:6.2f} | {100*t/tot:4.1f}%")
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

# Cross-cell signature-set diff: groups kernels by their SIGNATURE (the sorted parameter-name tuple,
# which is the identity that survives renumbering between two different fissions) and reports what
# exists only in one cell. This is what pins a mechanism to named kernels rather than to a bucket
# total -- Part 2's 14-vs-32 signatures and Part 3's 16-vs-17 are this subcommand's output.
cmd_diff() {
  local a; a=$(cell_dir "$1" "$2"); case ${a:-} in /?*) ;; *) return 2;; esac
  local b; b=$(cell_dir "$3" "$4"); case ${b:-} in /?*) ;; *) return 2;; esac
  python3 - "$1/r$2" "$a" "$3/r$4" "$b" <<'EOF' || return 1
import re,sys,csv,glob,os,statistics,collections
LOADED={}
import hashlib
SIG=re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+__seg(\d+))\s*\(([^)]*)\)\s*\{', re.S)
def load(name, out):
    try: src=open(open(os.path.join(out,"armA.path")).read().strip()).read()
    except OSError as e: sys.exit(f"{name}: {e} -- run `snap` first")
    sigs={}; bodies={}; params=set()
    for m in SIG.finditer(src):
        i=m.end(); d=1; j=i
        while d:
            if src[j]=="{": d+=1
            elif src[j]=="}": d-=1
            j+=1
        # Canonicalized body: whitespace collapsed, then each generated loop symbol ALPHA-RENAMED to
        # a stable token in first-appearance order. Collapsing them all to one token instead would
        # erase the relationships between indices -- a[i1][i2] and a[i2][i1] would hash alike -- so
        # the renaming has to be injective. Compile-order numbering is still normalized away.
        raw=re.sub(r"\s+"," ",src[i:j-1]).strip()
        seen={}
        def _rn(mo):
            k=mo.group(0)
            if k not in seen: seen[k]=f"v{len(seen)}"
            return seen[k]
        can=re.sub(r"\bi\d+\b",_rn,raw)
        sig=tuple(sorted(" ".join(q.split()).split()[-1].lstrip("*")
                         for q in m.group(3).split(",") if q.strip()))
        sigs[int(m.group(2))]=sig
        bodies[int(m.group(2))]=hashlib.md5(can.encode()).hexdigest()[:10]
        for q in m.group(3).split(","):
            q=" ".join(q.split())
            if q and "*" in q: params.add(q.split()[-1].lstrip("*"))
    LOADED[name]=(sigs,bodies,params)
    t=collections.defaultdict(list)
    # Timings are OPTIONAL here: "did the guard fire" is a question about the emitted kernel set, so
    # a structural diff must work off `snap` alone, without `profile`.
    csvs=sorted(glob.glob(os.path.join(out,"kernels-*.csv")))
    for f in csvs:
        for r in csv.DictReader(open(f)):
            t[int(r["Name"].rsplit("__seg",1)[1])].append(float(r["TotalDurationNs"])/1e6)
    ms={i:statistics.median(v) for i,v in t.items()}
    per=collections.defaultdict(float); n=collections.defaultdict(int)
    for i,sg in sigs.items(): per[sg]+=ms.get(i,0.0); n[sg]+=1
    return name, sigs, per, n, len(csvs)
(na,sa,pa,ca,ra)=load(sys.argv[1],sys.argv[2]); (nb,sb,pb,cb,rb)=load(sys.argv[3],sys.argv[4])
BODY_A=LOADED[na]; BODY_B=LOADED[nb]
def hdr(n,per,sg,r):
    t=f"{sum(per.values()):.3f} ms" if r else "(no timings: run `profile` for ms)"
    print(f"{n}: {t} / {len(sg)} kernels ({r} harness runs)")
hdr(na,pa,sa,ra); hdr(nb,pb,sb,rb)
onlya=[k for k in pa if k not in pb]; onlyb=[k for k in pb if k not in pa]
sh_a=sum(pa[k] for k in pa if k in pb); sh_b=sum(pb[k] for k in pb if k in pa)
print(f"\nsignatures only in {na}: {sum(ca[k] for k in onlya)} kernels, {sum(pa[k] for k in onlya):.3f} ms")
for k in sorted(onlya, key=lambda k:-pa[k])[:8]: print(f"  {pa[k]:7.3f} ms  {', '.join(k)[:120]}")
print(f"signatures only in {nb}: {sum(cb[k] for k in onlyb)} kernels, {sum(pb[k] for k in onlyb):.3f} ms")
for k in sorted(onlyb, key=lambda k:-pb[k])[:8]: print(f"  {pb[k]:7.3f} ms  {', '.join(k)[:120]}")
# MULTISET, not set: a signature present in both cells but a different NUMBER of times is not
# "shared". Without this, `only in` counts of 0/0 could still describe differing kernel multisets and
# the negative-control reading would be wrong while looking right.
remult=[(k,ca[k],cb[k]) for k in sorted(set(ca)&set(cb)) if ca[k]!=cb[k]]
if remult:
    print(f"\nshared signatures with DIFFERING multiplicity: {len(remult)} "
          f"({sum(x for _,x,_ in remult)} occurrences in {na} vs {sum(y for _,_,y in remult)} in {nb})")
    for k,x,y in sorted(remult, key=lambda t:-abs(t[1]-t[2]))[:8]:
        print(f"  x{x} -> x{y}  {', '.join(k)[:110]}")
elif onlya or onlyb:
    # Do NOT announce multiset agreement here: exclusive signatures on either side already mean the
    # kernel sets differ, and saying "the multisets agree" would contradict the counts just printed.
    print("\nshared signatures with differing multiplicity: 0 (the DIFFERENCE is entirely the "
          "exclusive signatures above)")
else:
    print("\nIDENTICAL kernel sets: no exclusive signatures on either side and no differing "
          "multiplicity\n-- the two cells emit the same kernel multiset. This is the negative-control "
          "reading.")
print(f"\nshared signatures: {sh_a:.3f} -> {sh_b:.3f} ms ({sh_b-sh_a:+.3f}); "
      f"NET {sum(pb.values())-sum(pa.values()):+.3f} ms")
# What a signature multiset does and does not settle. A kernel's POINTER PARAMETERS are exactly the
# materialized nodes it touches, so signature-multiset equality is the right invariant for the
# PLACEMENT question and is deliberately insensitive to the crowned tile. Bodies are not: they move
# with the tile too, so a body diff cannot be read as a placement change. Both are printed, and the
# newly-materialized NODE set is printed as well, because that -- not the signature count -- is the
# quantity that corresponds to guard firings (one materialized node can change several consumers'
# signatures, and several can be absorbed into one).
_,ba,pna=BODY_A; _,bb,pnb=BODY_B
mba=collections.Counter(ba.values()); mbb=collections.Counter(bb.values())
print(f"\ncanonicalized kernel BODIES differing: {sum((mba-mbb).values())} in {na}, "
      f"{sum((mbb-mba).values())} in {nb}")
print("  (bodies move with the crowned tile as well as with placement, so a body diff is NOT by "
      "itself\n   evidence that a placement decision changed -- check the crowned sketch labels too)")
newb=sorted(pnb-pna); newa=sorted(pna-pnb)
print(f"\nmaterialized NODES: {len(pna)} in {na}, {len(pnb)} in {nb}")
print(f"  only in {nb} (+{len(newb)}): {', '.join(newb) if newb else '-'}")
print(f"  only in {na} (+{len(newa)}): {', '.join(newa) if newa else '-'}")
print("  This node count is the closest available proxy for guard FIRINGS; exact counts would need a"
      "\n  provenance-41 placement log, which does not exist today.")
print("\nThe negative-control reading requires BOTH `only in` signature sides at 0 AND zero differing"
      "\nmultiplicities. It is a claim about PARAMETER-SIGNATURE multisets, i.e. about placement -- not"
      "\nabout byte-identical kernels.")
EOF
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
  diff)     cmd_diff     "$@" ;;
  sweep)    cmd_sweep    "$@" ;;
  roofline) cmd_roofline "$@" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
