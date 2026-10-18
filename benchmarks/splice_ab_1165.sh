#!/bin/bash
# gh-ocannl-1277 acceptance, measurement-only branch (never merged): the splice-segments A/B over
# gh-ocannl-1165's own caches, replayed on its treatment binary (182a9fa6 + the backported splice
# mode). Both arms replay the CONTROL's schedule for every segment but the q/k/v projections
# (structural digest 32d26aa17118, the twelve q/k/v kernels); the treatment arm takes those twelve
# from the treatment's crowned winner. The control's version-8 entries are adopted as version 10.
#
# Usage: splice_ab_1165.sh <out-dir> <rounds> [profiles...]   (profiles: exact approx; default both)
# Each round runs both arms, each a fresh bench_gpt process, in ABBA order across rounds.
set -u
[ $# -ge 2 ] || { echo "usage: $0 <out-dir> <rounds> [exact] [approx]" >&2; exit 2; }
OUT=$1
ROUNDS=$2
shift 2
PROFILES=("$@")
[ ${#PROFILES[@]} -gt 0 ] || PROFILES=(exact approx)
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd -P)
[ -z "$(ls -A "$OUT")" ] || { echo "splice_ab_1165: $OUT is not empty" >&2; exit 2; }
TREE=$(cd "$(dirname "$0")/.." && pwd -P) || exit 2
RUNS=${SPLICE_1165_RUNS:-$HOME/ocannl-staging-worktrees/1165-runs}
FIXTURE=${SPLICE_1165_FIXTURE:-$HOME/ocannl-bench-fixtures/gpt2_mini.safetensors}
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
(cd "$TREE" && dune build benchmarks/runners/ocannl/bench_gpt.exe) || { echo "splice_ab_1165: build failed" >&2; exit 1; }
GPT=$TREE/_build/default/benchmarks/runners/ocannl/bench_gpt.exe
{
  echo "splice_ab_1165: host=$(hostname) commit=$(git -C "$TREE" rev-parse HEAD) rounds=$ROUNDS profiles=${PROFILES[*]}"
  echo "fixture: $FIXTURE sha256 $(sha256sum "$FIXTURE" | cut -d' ' -f1)"
  echo "caches: $RUNS"
} | tee "$OUT/manifest.txt"
FAILED=0
run_arm() { # profile arm round
  local p=$1 arm=$2 r=$3 extra=() rc
  [ "$p" = approx ] && extra=(--ocannl_profile=approximate)
  (cd "$TREE/benchmarks" && env BENCH_FIXTURE="$FIXTURE" BENCH_TUNE=1 \
    BENCH_SPLICE_CONTROL="$RUNS/hip-ctrl-566de5e6/cache_$p" \
    BENCH_SPLICE_TREATMENT="$RUNS/hip-treat-182a9fa6/cache_$p" \
    BENCH_SPLICE_ADOPT_VERSION=8 BENCH_SPLICE_TARGET=32d26aa17118 BENCH_SPLICE_ARM="$arm" \
    "$GPT" --ocannl_backend=hip --ocannl_autotune_search=false "${extra[@]}") \
    >"$OUT/$r-$p-$arm.out" 2>"$OUT/$r-$p-$arm.err"
  rc=$?
  echo "$r $p $arm exit=$rc" | tee -a "$OUT/manifest.txt"
  [ "$rc" -eq 0 ] || FAILED=1
}
for ((r = 1; r <= ROUNDS; r++)); do
  for p in "${PROFILES[@]}"; do
    if ((r % 2)); then run_arm "$p" control "$r"; run_arm "$p" treatment "$r"
    else run_arm "$p" treatment "$r"; run_arm "$p" control "$r"; fi
  done
done
python3 - "$OUT" <<'EOF' | tee "$OUT/summary.txt"
import json, os, re, statistics, sys
out = sys.argv[1]
rows = {}
for f in sorted(os.listdir(out)):
    m = re.match(r"(\d+)-(\w+)-(control|treatment)\.out$", f)
    if not m:
        continue
    line = next((l for l in open(os.path.join(out, f)) if l.startswith("{")), None)
    if line is None:
        continue
    d = json.loads(line)
    assert d["searched"] is False, f
    rows.setdefault(m.group(2), {}).setdefault(m.group(3), []).append(
        (int(m.group(1)), d["step_ms"]["p50"], d["queued_step_ms"]))
for p, arms in rows.items():
    print(f"== {p}")
    for arm in ("control", "treatment"):
        xs = sorted(arms.get(arm, []))
        print(f"  {arm:9s} p50 " + " ".join(f"{x[1]:.4f}" for x in xs)
              + f"  mean {statistics.mean(x[1] for x in xs):.4f} stdev {statistics.pstdev([x[1] for x in xs]):.4f}"
              + f"  queued mean {statistics.mean(x[2] for x in xs):.4f}")
    c = dict((x[0], x[1]) for x in arms.get("control", []))
    t = dict((x[0], x[1]) for x in arms.get("treatment", []))
    d = [t[r] - c[r] for r in sorted(c) if r in t]
    if d:
        cm = statistics.mean(c.values())
        print(f"  per-round treatment-control (ms): " + " ".join(f"{x:+.4f}" for x in d))
        print(f"  mean delta {statistics.mean(d):+.4f} ms ({100*statistics.mean(d)/cm:+.2f}% of control), "
              f"min {min(d):+.4f} max {max(d):+.4f}")
EOF
echo "splice_ab_1165: done, failed=$FAILED"
exit $FAILED
