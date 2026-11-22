#!/bin/bash
# Client measurement-error experiment on a CPU node (no GPU needed): llmbench with 1-8 worker
# processes and vLLM's `vllm bench serve` against a fake server that streams on an exact schedule.
#   sbatch slurm/client-overhead.sbatch
#SBATCH --job-name=llmbench-client
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/llmbench/logs/%x-%j.out
set -euo pipefail
ROOT=${LLMBENCH_ROOT:-/scratch/$USER/llmbench}
source "$ROOT/repo/slurm/lib.sh"
mkdir -p "$ROOT/results/client_overhead"
lscpu | grep -E '^Model name|^CPU\(s\)'
"$ROOT/env/bin/python" "$ROOT/repo/experiments/client_overhead.py" \
  --concurrency "${CONC:-64,256,1024,2048}" --workers "${WORKERS:-1,2,4,8}" \
  --server-procs "${SERVER_PROCS:-16}" --tokens 200 --per-stream 3 \
  --vllm-bench "apptainer exec --bind /scratch/$USER $ROOT/vllm.sif vllm bench serve" \
  --out "$ROOT/results/client_overhead/explorer-${SLURM_JOB_ID}.json"
