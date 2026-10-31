#!/bin/bash
# gh-ocannl-1557 measurement driver: does each placement arm's shipped step keep the time its
# search crowned it at? Per weight set -- the search's scratch lineage on the `Train.init_params`
# draw (BENCH_TUNE_SCRATCH_WEIGHTS=init, how the runners searched before gh-ocannl-1557) or on the
# fixture's weights (the default) -- one search pass fills a fresh schedule cache with both arms'
# crowns, then two fresh processes replay it with arm A and arm B forced (`tune_ship_arm`), and the
# summary prints each replay's crown (`best_ms`) against its timed steps' p10 and p50. The step
# times come from replay processes only, as the README's two-pass protocol requires: a searching
# process's launches are inflated by its accumulated modules. A replay that searched instead (the
# search pass was contended, so nothing was stored), or whose ship substituted the untuned default
# (`tune.ship_substituted`), has no pair and fails the run.
#
# Usage:
#   benchmarks/gh1557_cells.sh (--box BOX | --skip-box-check) FIXTURE OUT BUDGET [WEIGHTS...]
#
#   --box    before the first cell, benchmarks/check-box.sh checks BOX and the fixture, and a GAP
#            refuses the run (benchmarks/box-gate.sh; --skip-box-check is for a deliberate partial
#            run). The checkout is the one this script is in; bench_conv.exe must already be built
#            there (`dune build ./benchmarks/runners/ocannl/bench_conv.exe`): it never builds.
#   FIXTURE  cifar_resnet.safetensors (the gate checks it against fixtures/DIGESTS.txt).
#   OUT      results directory, created, refused unless empty: per process (pass: search, a, b)
#            OUT/<weights>-<pass>.out (the result line) and .err, and OUT/<weights>-<pass>/ the
#            cell's working directory with its cell.json record; OUT/cache-<weights>/ the schedule
#            caches, OUT/summary.txt.
#   BUDGET   autotune_budget_seconds for every search (each arm is its own search).
#   WEIGHTS  init and/or fixture (default: both, init first).
#
# Exit: 0 every cell produced a crown/step pair; 1 a cell failed or has none; 2 usage; 125 a cell's
# process group outlived its cleanup (the run stops there).
# Each process is a cell of the shared runner (benchmarks/cells.py `cell`, gh-ocannl-1607): no
# OCANNL_*, BENCH_* or OpenMP variable (OMP_*, GOMP_*, KMP_*) and no config file reaches it, the
# backend (cc) and the treatment are on argv and the runner modes stated, and its resolved
# configuration is read back -- a process whose forced arm, cache or budget did not take fails. The
# passes stay sequential by design: the replays read the cache the search pass wrote.
set -u
. "$(dirname "$0")/box-gate.sh"
box_gate_args "$@"; shift "$box_gate_shift"
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
[ $# -ge 3 ] || usage
fixture=$1 out=$2 budget=$3
shift 3
case $budget in '' | *[!0-9]*) echo "gh1557_cells: BUDGET must be a whole number of seconds" >&2; exit 2 ;; esac
weights=("$@")
[ ${#weights[@]} -gt 0 ] || weights=(init fixture)
for w in "${weights[@]}"; do
  case $w in init | fixture) ;; *) echo "gh1557_cells: unknown weights: $w" >&2; usage ;; esac
done
root=$(cd "$(dirname "$0")/.." && pwd -P) || exit 2
exe=$root/_build/default/benchmarks/runners/ocannl/bench_conv.exe
[ -x "$exe" ] || { echo "gh1557_cells: $exe is not built" >&2; exit 2; }
# The gate first: an absent fixture is the check's GAP, with its repair, not a usage error here.
fixture_dir=$(cd "$(dirname "$fixture")" 2>/dev/null && pwd -P) || fixture_dir=$(dirname "$fixture")
fixture=$fixture_dir/$(basename "$fixture")
box_gate gh1557_cells "$fixture_dir" "$(basename "$fixture" .safetensors)"
[ -f "$fixture" ] || { echo "gh1557_cells: no fixture at $fixture" >&2; exit 2; }
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) || exit 2
# A fresh directory only: a leftover cache would make a search a replay.
[ -z "$(ls -A "$out")" ] || { echo "gh1557_cells: $out is not empty" >&2; exit 2; }
rc=0
for w in "${weights[@]}"; do
  # The search pass ships arm A (any arm would do: forcing one skips neither arm's search).
  for pass in search a b; do
    arm=$pass
    [ "$pass" = search ] && arm=a
    echo "gh1557_cells: weights=$w pass=$pass start $(date +%T)" >&2
    python3 "$root/benchmarks/cells.py" cell --backend cc --dir "$out/$w-$pass" \
      --stdout "$out/$w-$pass.out" --stderr "$out/$w-$pass.err" \
      --expect tune_ship_arm="$arm" --expect autotune_cache_dir="$out/cache-$w" -- \
      BENCH_FIXTURE="$fixture" BENCH_TUNE=1 BENCH_TUNE_SCRATCH_WEIGHTS="$w" "$exe" \
      --ocannl_tune_ship_arm="$arm" --ocannl_autotune_cache_dir="$out/cache-$w" \
      --ocannl_autotune_budget_seconds="$budget"
    case $? in 0) ;; 125) echo "gh1557_cells: a process outlived its cleanup; stopping" >&2; exit 125 ;; *) rc=1 ;; esac
  done
done
python3 - "$root/benchmarks" "$out" "${weights[@]}" >"$out/summary.txt" <<'EOF' || rc=1
import math, os, sys
sys.path.insert(0, sys.argv[1])
import bench_output
out = sys.argv[2]
bad = 0
for w in sys.argv[3:]:
    for arm in "ab":
        path = os.path.join(out, f"{w}-{arm}.out")
        try:
            d = bench_output.result_record(open(path).read())
        except (OSError, ValueError) as e:
            print(f"weights={w} arm={arm.upper()}: no result line ({e})")
            bad += 1
            continue
        try:
            shipped = d["tune"]["shipped"]
            if shipped != arm.upper():
                raise ValueError(f"forced arm {arm.upper()} but shipped {shipped}")
            a = {x["arm"]: x for x in d["tune"]["arms"]}[shipped]
            s = d["step_ms"]
            if d["searched"] is not False or a["state"] != "cache-replay":
                raise ValueError(f"not a replay (searched={d['searched']}, state={a['state']})")
            # A shipping compile that fell back to the untuned default keeps the replay's report:
            # its steps are not the crown's (orchestrate.substitution_check reads the same field).
            # Only an explicit false is a measured ship; true, null or absent has no pair.
            if d["tune"].get("ship_substituted") is not False:
                raise ValueError(f"not a measured ship (ship_substituted="
                                 f"{d['tune'].get('ship_substituted')})")
            crown = float(a["best_ms"])
            float(s["p10"]), float(s["p50"])
            if not (math.isfinite(crown) and crown > 0):
                raise ValueError(f"crown {crown}")
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # No result line (None), a process that searched (its step times are a searching
            # process's), a substituted ship, or a search that crowned nothing (best_ms null).
            print(f"weights={w} arm={arm.upper()}: no crown/step pair ({e!r})")
            bad += 1
            continue
        print(f"weights={w} shipped={shipped} state={a['state']} crown={a['best_ms']:.2f} "
              f"p10={s['p10']:.2f} p50={s['p50']:.2f} p10/crown={s['p10'] / a['best_ms']:.3f} "
              f"p50/crown={s['p50'] / a['best_ms']:.3f} contended={a['timings_contended']} "
              f"label={a['best_label']}")
sys.exit(1 if bad else 0)
EOF
cat "$out/summary.txt"
exit $rc
