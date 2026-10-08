#!/bin/bash
# MLX buffer cache limit against rerank time and the process's physical
# footprint, each setting in its own process; torch for reference. Changes no
# service.
#   nohup bash cache-sweep.sh <results dir> > <results dir>/sweep.log 2>&1 &
set -uo pipefail
out=$(realpath "$1")
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
model=$HOME/knowva-inference/models/llama-nemotron-rerank-vl-1b-v2
run() {
  local name=$1; shift
  health=$(curl -s 127.0.0.1:9500/health)
  if ! grep -q '"loaded":\[\]' <<< "$health"; then echo "SKIPPED $name: the production service has models loaded"; return; fi
  echo "### $name $(date +%T) swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
  "$root/.venv/bin/python" -u "$here/bench_sets.py" "$root/fixtures/sets" "$model" "$out/$name.json" \
    --dtype float16 --tiles 1 --batching unpadded "$@" > "$out/$name.log" 2>&1
  echo "exit $? $(tail -1 "$out/$name.log") swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
  sleep 10
}
run mlx-cache-default --backend mlx
run mlx-cache-0 --backend mlx --cache-limit-mb 0
run mlx-cache-256 --backend mlx --cache-limit-mb 256
run mlx-cache-1024 --backend mlx --cache-limit-mb 1024
run torch-unpadded --backend torch --device mps
echo SWEEP_DONE
