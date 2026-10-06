#!/bin/bash
# Runs bench_sets.py for each backend and precision in its own process, one
# after another, on the Mac mini. Skips a run (and says so) when the
# production inference service has models loaded, since sharing memory with
# it would distort the times. Changes no service.
#   nohup bash run-sets.sh <results dir> [run ...] > <results dir>/run.log 2>&1 &
# A run is backend:device:dtype[:batching], e.g. torch:mps:float16 or mlx:gpu:float16:unpadded.
# SETS picks another candidate-set directory (default fixtures/sets);
# CACHE_LIMIT_MB caps MLX's buffer cache as KM_MLX_CACHE_MB does in the service;
# MLX_ENABLE_TF32=1 keeps MLX's TF32 default, as the service does.
set -uo pipefail
out=$(realpath "$1"); shift
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
models=$HOME/knowva-inference/models/llama-nemotron-rerank-vl-1b-v2
sets=${SETS:-$root/fixtures/sets}
runs=("$@")
[ ${#runs[@]} -gt 0 ] || runs=(torch:mps:float16 mlx:gpu:float16 mlx:gpu:bfloat16 torch:mps:float32 mlx:gpu:float32)

sample() {
  while :; do
    printf '%s swap_used=%s pressure=%s ' "$(date +%T)" "$(sysctl -n vm.swapusage | awk '{print $6}')" \
      "$(sysctl -n kern.memorystatus_vm_pressure_level)"
    vm_stat | awk '/Pages free/ {f=$NF} /Pages wired/ {w=$NF} /occupied by compressor/ {c=$NF}
      END {printf "free=%.2fGiB wired=%.2fGiB compressed=%.2fGiB\n", f*16384/2^30, w*16384/2^30, c*16384/2^30}'
    sleep 2
  done
}

for run in "${runs[@]}"; do
  IFS=: read -r backend device dtype variant <<< "$run"
  name="$backend-$device-$dtype${variant:+-$variant}"
  health=$(curl -s 127.0.0.1:9500/health)
  echo "### $name $(date +%T) service: $health"
  if ! grep -q '"loaded":\[\]' <<< "$health"; then echo "SKIPPED $name: the inference service has models loaded"; continue; fi
  sample > "$out/memory-$name.log" 2>&1 &
  sampler=$!
  extra=()
  [ -n "$variant" ] && extra=(--batching "$variant")
  [ -n "${CACHE_LIMIT_MB:-}" ] && extra+=(--cache-limit-mb "$CACHE_LIMIT_MB")
  # TF32 is MLX's default on this GPU; float32 means float32 here unless asked.
  MLX_ENABLE_TF32=${MLX_ENABLE_TF32:-0} "$root/.venv/bin/python" -u "$here/bench_sets.py" "$sets" "$models" "$out/$name.json" \
    --backend "$backend" --device "$device" --dtype "$dtype" --tiles 1 ${extra[@]+"${extra[@]}"} > "$out/$name.log" 2>&1
  echo "exit $? $(tail -1 "$out/$name.log")"
  kill "$sampler" 2>/dev/null
  echo "after: $(curl -s 127.0.0.1:9500/health)"
done
echo SETS_DONE
