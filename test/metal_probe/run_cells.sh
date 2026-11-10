#!/bin/sh
# gh-ocannl-2082: builds metal_hang_probe and runs the cells named on the command line, one process
# each, in order (a cell is its words joined by `:`, e.g. `stall:1:120`). Stalls the GPU: run it only
# on a Mac with the GPU to itself. Prints the platform first; every cell's lines go to stdout.
set -u
cd "$(dirname "$0")/../.." || exit 2
dune build ./test/metal_probe/metal_hang_probe.exe || exit 2
P=./_build/default/test/metal_probe/metal_hang_probe.exe
echo "host=$(hostname -s) $(sw_vers -productVersion) $(sw_vers -buildVersion) $(sysctl -n machdep.cpu.brand_string)"
echo "uptime: $(uptime)"
echo "console user: $(stat -f %Su /dev/console)"
pmset -g assertions | grep -iE 'display|PreventUserIdleDisplaySleep' | head -5
system_profiler SPDisplaysDataType 2>/dev/null | grep -iE 'Chipset|Total Number of Cores|Display Type|Resolution|Online|Main Display' | sed 's/^ */  /'
for c in "$@"; do
  # shellcheck disable=SC2086
  set -- $(echo "$c" | tr ':' ' ')
  echo "--- cell $* at $(date -u +%H:%M:%S)"
  if [ "$1" = oob-validated ]; then shift; MTL_SHADER_VALIDATION=1 "$P" oob "$@"; else "$P" "$@"; fi
  echo "--- cell exit=$? at $(date -u +%H:%M:%S)"
done
