#!/bin/bash
# One engine configuration on one H200: start the server, run the listed experiments, stop.
#
#   ENGINE=vllm   QUANT=bf16 EXPS="sweep cap_interactive cap_relaxed" sbatch slurm/bench.sbatch
#   ENGINE=sglang QUANT=fp8  EXPS="sweep cap_interactive"               sbatch slurm/bench.sbatch
#   ENGINE=vllm   EXPS="prefix"  sbatch slurm/bench.sbatch   # restarts the engine without caching
#
# Results land in $ROOT/results/<tag>/; copy them into the repo's results/ directory.
#SBATCH --job-name=llmbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/llmbench/logs/%x-%j.out
set -uo pipefail

ROOT=${LLMBENCH_ROOT:-/scratch/$USER/llmbench}
# shellcheck source=slurm/lib.sh
source "$ROOT/repo/slurm/lib.sh"
# shellcheck source=slurm/experiments.sh
source "$ROOT/repo/slurm/experiments.sh"

ENGINE=${ENGINE:?ENGINE=vllm|sglang}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
QUANT=${QUANT:-bf16}
EXPS=${EXPS:-sweep}
ENGINE_ARGS=${ENGINE_ARGS:-}
PORT=$((20000 + RANDOM % 20000))
JOB=${SLURM_JOB_ID:-local}  # engine logs are per job, so a rerun never overwrites an earlier log
SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/-instruct//')
TAG=${TAG:-${ENGINE}_${SHORT}_${QUANT}}
OUT=$ROOT/results/$TAG
mkdir -p "$OUT"

quant_args=()
if [ "$QUANT" = fp8 ]; then quant_args=(--quantization fp8); fi

{
  echo "host=$(hostname) job=${SLURM_JOB_ID:-} engine=$ENGINE model=$MODEL quant=$QUANT"
  echo "engine_version=$(engine_version "$ENGINE")"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E '^Model name|^CPU\(s\)'
} | tee -a "$OUT/node.txt"

engine_started() {  # record the engine's KV capacity and the head of its log
  echo "kv_tokens_${1}=$(kv_capacity "$ENGINE_LOG")" | tee -a "$OUT/node.txt"
  head -c 200000 "$ENGINE_LOG" > "${ENGINE_LOG%.log}-head.log"
}

run_exp() {
  local exp=$1
  echo "[$(date +%T)] === $exp"
  "exp_$exp" || echo "experiment $exp failed with status $?"
}

trap stop_engine EXIT
# shellcheck disable=SC2086  # ENGINE_ARGS is intentionally word-split
for exp in $EXPS; do
  case $exp in
    prefix)  # one engine launch per mode: prefix caching on, then off
      for mode in ${PREFIX_MODES:-on off}; do
        extra=()
        if [ "$mode" = off ]; then
          if [ "$ENGINE" = vllm ]; then extra=(--no-enable-prefix-caching); else extra=(--disable-radix-cache); fi
        fi
        start_engine "$ENGINE" "$MODEL" "$PORT" "$OUT/engine-prefix-$mode-$JOB.log" \
          "${quant_args[@]}" $ENGINE_ARGS $(ctx_len_args "$ENGINE" 20480) "${extra[@]}" || exit 1
        engine_started "prefix_$mode"
        CACHE=$mode run_exp prefix
        stop_engine
      done
      ;;
    *)
      if [ -z "$ENGINE_PID" ]; then
        start_engine "$ENGINE" "$MODEL" "$PORT" "$OUT/engine-$JOB.log" \
          "${quant_args[@]}" $ENGINE_ARGS $(ctx_len_args "$ENGINE" 8192) || exit 1
        engine_started main
      fi
      run_exp "$exp"
      ;;
  esac
done
echo "[$(date +%T)] done"
