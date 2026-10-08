#!/bin/bash
# Runs a copy of Knowva's Mac inference service (server.py) on 127.0.0.1:9502,
# next to the production one on 9500, once per rerank backend. Each run embeds
# a query first, so the embed model is resident as it is while people ask
# questions, then replays the candidate sets through HTTP while sampling the
# process's physical footprint (top's MEM, which counts GPU memory; ps's RSS
# does not). Changes no service; each test server is stopped after its run.
#   bash server-test.sh <results dir> <dir holding server.py, pyproject.toml, uv.lock> [backend ...]
set -uo pipefail
export PATH="$HOME/.proto/shims:$HOME/.proto/bin:/usr/bin:/bin:/usr/sbin:/sbin"
out=$(realpath "$1")
app=$(realpath "$2")
shift 2
backends=("$@")
[ ${#backends[@]} -gt 0 ] || backends=(torch mlx)
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
models=$HOME/knowva-inference/models
port=9502
python=$HOME/.proto/tools/python/3.12.15/bin/python

server=
sampler=
stop() {
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  [ -n "$server" ] && kill "$server" 2>/dev/null && wait "$server" 2>/dev/null
  echo "### stopped $(date +%T)"
}
trap stop EXIT
trap 'exit 1' INT TERM HUP


cd "$app" || exit 1
proto exec uv -- uv sync --frozen --python "$python" || exit 1
proto exec uv -- uv pip install --python .venv/bin/python "$root" || exit 1
.venv/bin/python -c "import mlx.core as mx, nemotron_vl_mlx, torch; print('mlx', mx.__version__, 'torch', torch.__version__)"

footprint() { top -l 1 -pid "$server" -stats mem | tail -1 | tr -d ' '; }
sample() {
  while :; do
    printf '%s footprint=%s swap_used=%s pressure=%s\n' "$(date +%T)" "$(footprint)" \
      "$(sysctl -n vm.swapusage | awk '{print $6}')" "$(sysctl -n kern.memorystatus_vm_pressure_level)"
    sleep 2
  done
}
embed() {
  curl -sf 127.0.0.1:$port/v1/embeddings -H 'content-type: application/json' \
    -d "{\"model\":\"nvidia/llama-nemotron-embed-vl-1b-v2\",\"input\":[\"$1\"],\"input_type\":\"query\",\"modality\":[\"text\"]}" \
    | "$root/.venv/bin/python" -c "import json,sys; print('embedding dims', len(json.load(sys.stdin)['data'][0]['embedding']))"
}

for backend in "${backends[@]}"; do
  health=$(curl -s 127.0.0.1:9500/health)
  echo "### $backend $(date +%T) production: $health"
  if ! grep -q '"loaded":\[\]' <<< "$health"; then echo "SKIPPED $backend: the production service has models loaded"; continue; fi
  echo "before: swap_used=$(sysctl -n vm.swapusage | awk '{print $6}') pressure=$(sysctl -n kern.memorystatus_vm_pressure_level)"
  HF_HUB_OFFLINE=1 KM_PORT=$port KM_RERANK_BACKEND=$backend KM_DTYPE=float16 \
    KM_EMBED_PATH=$models/llama-nemotron-embed-vl-1b-v2 KM_RERANK_PATH=$models/llama-nemotron-rerank-vl-1b-v2 \
    KM_MAX_TILES=6 KM_RERANK_TILES=1 KM_RERANK_INPUT=both KM_IDLE_SECS=600 KM_CHAT_IMAGE_MAX_SIDE=1280 \
    .venv/bin/python server.py > "$out/server-$backend.log" 2>&1 &
  server=$!
  for _ in $(seq 120); do curl -sf 127.0.0.1:$port/health >/dev/null && break; sleep 0.5; done
  sample > "$out/memory-$backend.log" 2>&1 &
  sampler=$!
  embed "IoT4Industry D.1 文件的作者是誰？" || echo "embed failed"
  "$root/.venv/bin/python" -u "$here/bench_sets.py" "$root/fixtures/sets" "$models/llama-nemotron-rerank-vl-1b-v2" \
    "$out/http-$backend.json" --backend http --url http://127.0.0.1:$port > "$out/http-$backend.log" 2>&1
  echo "exit $? $(tail -1 "$out/http-$backend.log") footprint $(footprint)"
  embed "智慧機械產業推動方案" || echo "embed failed"
  curl -s 127.0.0.1:$port/health; echo
  grep -iE "error|traceback|exception" "$out/server-$backend.log" | head -5
  kill "$sampler" 2>/dev/null; sampler=
  kill "$server" 2>/dev/null; wait "$server" 2>/dev/null; server=
  echo "peak footprint $(awk '{print $2}' "$out/memory-$backend.log" | sort -t= -k2 -h | tail -1)," \
    "max swap $(awk '{print $3}' "$out/memory-$backend.log" | sort -t= -k2 -n | tail -1)," \
    "max pressure $(awk '{print $4}' "$out/memory-$backend.log" | sort | tail -1)"
  sleep 20
done
echo SERVER_TEST_DONE
