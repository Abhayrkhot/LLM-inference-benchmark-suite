# Experiment definitions used by bench.sbatch. Each exp_* function benchmarks the engine that is
# already running on $PORT and writes into $OUT. Sourced, not executed.

LB=${LB:-$ROOT/env/bin/llmbench}
DATA=$ROOT/data
SHAREGPT=(--dataset sharegpt --dataset-path "$DATA/ShareGPT_V3_unfiltered_cleaned_split.json")
# MLPerf Inference-style latency tiers (p99 TTFT, p99 TPOT).
SLO_INTERACTIVE_REQ="ttft<=500ms,tpot<=30ms"
SLO_INTERACTIVE="ttft_p99<=500ms,tpot_p99<=30ms"
SLO_RELAXED="ttft_p99<=2s,tpot_p99<=100ms"
RATES=${RATES:-2,4,8,12,16,20,24,32,40,48,64,inf}

common() {
  echo --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$MODEL" \
    --gpu-monitor --workers 4 --tag "engine=$ENGINE" --tag "quant=$QUANT"
}

# Latency-throughput curve on ShareGPT, Poisson arrivals, ~60 s per point.
exp_sweep() {
  # shellcheck disable=SC2046
  "$LB" sweep $(common) "${SHAREGPT[@]}" --rates "$RATES" --point-duration 60 \
    --num-requests 2000 --warmup 32 --slo "$SLO_INTERACTIVE_REQ" --out "$OUT/sweep" --name "$TAG"
}

capacity() {  # capacity <name> <constraints>
  # shellcheck disable=SC2046
  "$LB" capacity $(common) "${SHAREGPT[@]}" --constraints "$2" --probe-duration 60 \
    --num-requests 2000 --warmup 32 --max-probes 9 --tolerance 0.05 \
    --slo "$SLO_INTERACTIVE_REQ" --out "$OUT/$1" --name "${TAG}_$1"
}
exp_cap_interactive() { capacity cap_interactive "$SLO_INTERACTIVE"; }
exp_cap_relaxed() { capacity cap_relaxed "$SLO_RELAXED"; }

# Everything at once: 2,000 ShareGPT requests sent at t=0, to see how the scheduler copes.
exp_burst() {
  # shellcheck disable=SC2046
  "$LB" run $(common) "${SHAREGPT[@]}" -n 2000 --rate inf --warmup 32 --seed 1000 \
    --out "$OUT/burst" --name "${TAG}_burst"
}

# Run-to-run variance: the same load five times with different prompt samples and arrivals.
exp_repeat() {
  local rate=${REPEAT_RATE:-16}
  for i in ${REPEAT_IDS:-1 2 3 4 5}; do
    # shellcheck disable=SC2046
    "$LB" run $(common) "${SHAREGPT[@]}" --rate "$rate" -n $((rate * 60)) --seed $((100 + i)) \
      --warmup 16 --slo "$SLO_INTERACTIVE_REQ" --out "$OUT/repeat" \
      --name "${TAG}_rate${rate}_rep$i"
  done
}

# Cross-validation against vLLM's own client on an identical-shape workload.
exp_xval() {
  local rate
  for rate in ${XVAL_RATES:-4 16 32 inf}; do
    # shellcheck disable=SC2046
    "$LB" run $(common) --dataset random --input-len 1024 --output-len 256 -n 1000 \
      --rate "$rate" --warmup 16 --out "$OUT/xval" --name "llmbench_rate$rate"
    apptainer exec --bind "/scratch/$USER" "$ROOT/vllm.sif" vllm bench serve \
      --backend vllm --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
      --dataset-name random --random-input-len 1024 --random-output-len 256 \
      --random-range-ratio 0 --num-prompts 1000 --request-rate "$rate" --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "$OUT/xval" --result-filename "vllm_rate$rate.json" \
      2>&1 | tail -40
  done
}

# Prefix caching on a production trace with real prompt sharing (Mooncake tool-agent, Kimi).
exp_prefix() {
  local scale
  for scale in ${PREFIX_SCALES:-1 2}; do
    reset_cache
    # shellcheck disable=SC2046
    "$LB" run $(common) --dataset mooncake --dataset-path "$DATA/mooncake_toolagent_trace.jsonl" \
      -n "${PREFIX_N:-1500}" --max-input-len 16384 --arrival trace --time-scale "$scale" \
      --warmup 16 --out "$OUT/prefix" --name "${TAG}_cache${CACHE}_x$scale"
  done
}

reset_cache() {
  curl -sf -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null \
    || curl -sf -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null \
    || echo "warning: could not reset the prefix cache"
}

# Quick end-to-end check of an engine image before the long jobs.
exp_smoke() {
  # shellcheck disable=SC2046
  "$LB" run $(common) "${SHAREGPT[@]}" -n 60 --rate 8 --warmup 4 --slo "$SLO_INTERACTIVE_REQ" \
    --out "$OUT/smoke" --name "${TAG}_smoke_sharegpt" &&
  # shellcheck disable=SC2046
  "$LB" run $(common) --dataset random --input-len 512 --output-len 64 -n 40 --concurrency 8 \
    --out "$OUT/smoke" --name "${TAG}_smoke_random" &&
  # shellcheck disable=SC2046
  "$LB" run $(common) --backend chat "${SHAREGPT[@]}" -n 20 --concurrency 4 \
    --out "$OUT/smoke" --name "${TAG}_smoke_chat"
}
