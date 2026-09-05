#!/usr/bin/env bash
# Where do the minutes of a CI run go?  Prints one line per job of a GitHub
# Actions run with the job's total wall-clock duration, and indented under it
# the steps that took more than ~30 seconds -- the question every CI-speed
# investigation starts with, previously answered by rewriting the same
# throwaway python each time (PRs #518/#519).
#
# Jobs and steps appear in the order GitHub lists them.  Pure `gh api` +
# python3; no OCANNL build involved.  A job's conclusion is appended when it
# is anything other than success (failure, cancelled, skipped...), and a job
# with no wall-clock duration to report says `(no time)` rather than a number
# (see the `secs` comment below).
#
# tools/test-ci-times.sh is the hermetic harness for the reporting below; it
# runs this exact file against a recorded `gh` fixture.
#
# Usage: tools/ci-times.sh [RUN_ID]
#   tools/ci-times.sh              # latest completed ci.yml run on the current repo
#   tools/ci-times.sh 17274043031  # that specific run
#
# The repo is whatever `gh` resolves for the current directory's remotes; run
# it from the checkout whose CI you are asking about.

set -euo pipefail

STEP_THRESHOLD=30 # seconds; steps at or under this stay unlisted

run_id="${1-}"
if [ -z "$run_id" ]; then
  run_id=$(gh run list --workflow ci.yml --status completed --limit 1 \
    --json databaseId --jq '.[0].databaseId')
  if [ -z "$run_id" ] || [ "$run_id" = "null" ]; then
    echo "ci-times.sh: no completed ci.yml run found on this repo" >&2
    exit 1
  fi
  echo "run $run_id (latest completed ci.yml)"
fi

# Fetch first, pipe second: on an HTTP error gh prints the response body to
# stdout, which would otherwise reach python as unparseable "JSON".
jobs=$(gh api --paginate "repos/{owner}/{repo}/actions/runs/${run_id}/jobs?per_page=100" \
  --jq '.jobs[]')

printf '%s\n' "$jobs" |
  python3 -c "
import json, sys
from datetime import datetime

threshold = int(sys.argv[1])

def secs(start, end):
    # None means 'this pair is not an interval', which covers two shapes.  A
    # missing endpoint is the obvious one.  The other is a pair that runs
    # BACKWARDS: a job GitHub never ran carries placeholder timestamps whose
    # completed_at precedes its started_at, and the plain subtraction reports
    # that as a duration -- '-1m59s' against a skipped job, seen while
    # measuring gh-ocannl-901 on staging#611.  Negative wall clock is not a
    # measurement, so it is unavailable rather than small.
    if not start or not end:
        return None
    fmt = '%Y-%m-%dT%H:%M:%SZ'
    d = (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds()
    return d if d >= 0 else None

def human(s):
    s = int(round(s))
    m, s = divmod(s, 60)
    return f'{m}m{s:02d}s' if m else f'{s}s'

def label_for(job, total):
    if total is not None:
        return human(total)
    status = job.get('status')
    # A queued or running job has no duration YET, and saying which is useful.
    # A completed one with no usable interval has none at all -- it was
    # skipped, or GitHub served no times -- and '(completed)' would read as an
    # answer to the question the column asks.
    if status and status != 'completed':
        return '(' + str(status) + ')'
    return '(no time)'

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    job = json.loads(line)
    total = secs(job.get('started_at'), job.get('completed_at'))
    label = label_for(job, total)
    tail = '' if job.get('conclusion') in (None, 'success') else '  [' + job['conclusion'] + ']'
    print(f'{label:>8}  {job[\"name\"]}{tail}')
    for step in job.get('steps') or []:
        d = secs(step.get('started_at'), step.get('completed_at'))
        if d is not None and d > threshold:
            print(f'{human(d):>12}    {step[\"name\"]}')
" "$STEP_THRESHOLD"
