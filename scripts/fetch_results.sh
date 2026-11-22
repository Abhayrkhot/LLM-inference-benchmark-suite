#!/bin/bash
# Copy run summaries (not the per-request records) from the cluster into results/.
#   EXPLORER="ssh -i ~/.ssh/key user@login.explorer.northeastern.edu" scripts/fetch_results.sh
set -euo pipefail
REMOTE_USER_HOST=${REMOTE:-khot.ab@login.explorer.northeastern.edu}
SSH=${SSH_CMD:-ssh}
SRC=${SRC:-/scratch/khot.ab/llmbench/results}
cd "$(dirname "$0")/.."
rsync -az -e "$SSH" --prune-empty-dirs --exclude 'smoke*/' --exclude '_*/' \
  --include '*/' --include '*.json' --include 'node.txt' --include '*-head.log' --exclude '*' \
  "$REMOTE_USER_HOST:$SRC/" results/h200/
mkdir -p results/client_overhead
mv results/h200/client_overhead/*.json results/client_overhead/ 2>/dev/null || true
rmdir results/h200/client_overhead 2>/dev/null || true
du -sh results
