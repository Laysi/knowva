#!/bin/bash
# Runs embed_vectors.py for each backend and precision in its own process, one
# after another, on the Mac mini, then compares them with PyTorch MPS float32
# and with PyTorch MPS float16 (what stored page vectors were made with).
# Skips a run (and says so) when the production inference service has models
# loaded. Changes no service.
#   nohup bash run-embed.sh <results dir> [run ...] > <results dir>/run.log 2>&1 &
# A run is backend:device:dtype, e.g. torch:mps:float16 or mlx:gpu:float16.
# CACHE_LIMIT_MB caps MLX's buffer cache as KM_MLX_CACHE_MB does in the service.
set -uo pipefail
out=$(realpath "$1"); shift
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
models=$HOME/knowva-inference/models/llama-nemotron-embed-vl-1b-v2
pages=${PAGES:-$HOME/knowva-inference/bench-pages}
questions=${QUESTIONS:-$HOME/knowva-inference/app/bench/realistic.json}
runs=("$@")
[ ${#runs[@]} -gt 0 ] || runs=(torch:mps:float32 torch:mps:float16 mlx:gpu:float16 mlx:gpu:float32)

for run in "${runs[@]}"; do
  IFS=: read -r backend device dtype <<< "$run"
  name="$backend-$device-$dtype"
  health=$(curl -s 127.0.0.1:9500/health)
  echo "### $name $(date +%T) service: $health"
  if ! grep -q '"loaded":\[\]' <<< "$health"; then echo "SKIPPED $name: the inference service has models loaded"; continue; fi
  extra=()
  [ -n "${CACHE_LIMIT_MB:-}" ] && extra+=(--cache-limit-mb "$CACHE_LIMIT_MB")
  # TF32 is MLX's default on this GPU; float32 means float32 here.
  MLX_ENABLE_TF32=0 "$root/.venv/bin/python" -u "$root/tools/embed_vectors.py" "$models" "$pages" "$questions" \
    "$out/$name.npz" --backend "$backend" --device "$device" --dtype "$dtype" --tiles 6 \
    ${extra[@]+"${extra[@]}"} > "$out/$name.log" 2>&1
  echo "exit $?"
  cat "$out/$name.log"
  echo "after: $(curl -s 127.0.0.1:9500/health)"
done
cd "$out" || exit 1
others=(*.npz)
for ref in torch-mps-float32.npz torch-mps-float16.npz; do
  [ -f "$ref" ] || continue
  echo "## reference $ref"
  "$root/.venv/bin/python" "$root/tools/compare_embed.py" "$ref" $(printf '%s\n' "${others[@]}" | grep -vx "$ref")
done
echo EMBED_DONE
