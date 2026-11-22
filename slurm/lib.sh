# Helpers for running engines from Apptainer images on a Slurm GPU node. Source from a job.
#
# Engine images are built for CUDA 13; Explorer's H200 driver (570) is CUDA 12.8, so the image's
# forward-compatibility libraries (/usr/local/cuda/compat) go first on LD_LIBRARY_PATH.

ROOT=${LLMBENCH_ROOT:-/scratch/$USER/llmbench}
# The client venv was built with this module's Python and needs its libpython at run time.
module load python/3.13.5 2>/dev/null || true
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}
mkdir -p "$TRITON_CACHE_DIR"

ENGINE_PID=""
ENGINE_LOG=""

# start_engine <vllm|sglang> <model> <port> <log> [extra engine args...]
start_engine() {
  local engine=$1 model=$2 port=$3 log=$4
  shift 4
  local sif=$ROOT/$engine.sif
  local cmd
  if [ "$engine" = vllm ]; then
    cmd=(vllm serve "$model" --host 127.0.0.1 --port "$port" --served-model-name "$model" "$@")
  else
    cmd=(python3 -m sglang.launch_server --model-path "$model" --host 127.0.0.1 --port "$port"
         --served-model-name "$model" --enable-metrics "$@")
  fi
  echo "[$(date +%T)] starting $engine: ${cmd[*]}"
  ENGINE_LOG=$log
  # setsid: the engine and all its worker processes get their own process group, so
  # stop_engine can kill exactly them and nothing else on a shared node.
  VLLM_SERVER_DEV_MODE=1 setsid apptainer exec --nv --bind "/scratch/$USER" "$sif" bash -c '
    export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    exec "$@"' _ "${cmd[@]}" >"$log" 2>&1 &
  ENGINE_PID=$!
  local t0
  t0=$(date +%s)
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then
      echo "[$(date +%T)] $engine ready after $(( $(date +%s) - t0 ))s"
      return 0
    fi
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      echo "$engine exited during startup"; tail -60 "$log"; return 1
    fi
    sleep 5
  done
  echo "$engine not ready after 30 min"; tail -60 "$log"; return 1
}

stop_engine() {  # stop this job's engine: its process group only, never a node-wide pkill
  if [ -n "$ENGINE_PID" ]; then
    kill -TERM -- "-$ENGINE_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 -- "-$ENGINE_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$ENGINE_PID" 2>/dev/null
    wait "$ENGINE_PID" 2>/dev/null
    sleep 3  # let the driver release GPU memory before the next engine starts
  fi
  ENGINE_PID=""
}

engine_version() {  # engine_version <vllm|sglang>
  if [ "$1" = vllm ]; then
    apptainer exec "$ROOT/vllm.sif" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null
  else
    apptainer exec "$ROOT/sglang.sif" python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null
  fi
}

ctx_len_args() {  # ctx_len_args <vllm|sglang> <tokens>: the engine's max-context flag
  if [ "$1" = vllm ]; then echo "--max-model-len $2"; else echo "--context-length $2"; fi
}

kv_capacity() {  # kv_capacity <engine log>: KV-cache size in tokens as the engine reports it
  grep -oE 'GPU KV cache size: [0-9,]+ tokens|max_total_num_tokens=[0-9]+' "$1" | tail -1 \
    | grep -oE '[0-9,]+' | tr -d ,
}
