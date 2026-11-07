#!/usr/bin/env bash
# Export the study's ratings from MongoDB into this folder and commit + push
# them to the private repo (github.com/mouryat9/cogbench-v2), so no rating
# lives only in the database. Run from cron every 10 minutes; it only commits
# when something changed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/hnagulap/Mourya/thesis_proposal_work/venv/bin/python
cd "$HERE"
"$PY" export_ratings.py >/dev/null
if [ -s ratings/ratings_long.csv ] && [ "$(wc -l < ratings/ratings_long.csv)" -gt 1 ]; then
  "$PY" analyze.py >/dev/null 2>&1 || true
fi
cd "$HERE/.."
git add human_study_3rater
if ! git diff --cached --quiet; then
  git commit -q -m "human_study_3rater: sync ratings $(date -u +%Y-%m-%dT%H:%MZ)"
  git push -q origin HEAD 2>&1 || echo "push failed; commit kept locally"
fi
