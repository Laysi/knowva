#!/bin/bash
# Runs #60's retrieval configurations one after another on an isolated Knowva
# instance (compose.yml: its own database and files, port 8081)
# against the production inference service, so production's .env, app and
# plist are never touched. Per configuration: unload Gemma for an empty prompt
# cache, bench/concurrency.ts at 1-4 users, then examples.json, realistic.json
# and identifiers.json one question at a time, sampling memory and the
# inference service's and Ollama's footprints throughout.
#
#   nohup bash run.sh <instance dir> <results dir> <dense-rerank pool size> \
#     [configurations] > <results dir>/e2e.log 2>&1 &
#
# [configurations] is a space-separated subset of "dense-6 dense-8
# dense-rerank-<pool> current", run in that order (default: all four).
# <instance dir> holds a copy of compose.yml and dense60.env (POSTGRES_PASSWORD,
# GEMMA_4_MODEL_ID, optionally KNOWVA_IMAGE). Each configuration's settings are appended to a copy of
# dense60.env; touch <results dir>/ABORT to stop before the next one. The log
# ends in E2E_DONE or E2E_INCOMPLETE.
set -uo pipefail
# variant.env copies the instance's database password.
umask 077
export PATH="$HOME/.proto/shims:$HOME/.proto/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
instance=$(cd "$1" && pwd)
out=$(cd "$2" && pwd)
pool=$3
selected=" ${4:-dense-6 dense-8 dense-rerank-$pool current} "
here=$(cd "$(dirname "$0")" && pwd)
app=$HOME/knowva-inference/app
url=http://127.0.0.1:8081
O=127.0.0.1:11434
ollama_log=$HOME/Library/Logs/knowva/ollama.log

compose() {
  docker compose -p knowva-dense60 -f "$instance/compose.yml" --env-file "$instance/variant.env" "$@"
}

footprint() {
  local pid
  pid=$(lsof -ti "tcp:$1" -sTCP:LISTEN | head -1)
  [ -n "$pid" ] && top -l 1 -pid "$pid" -stats mem | tail -1 | tr -d ' ' || echo -
}
sample() {
  while :; do
    printf '%s swap_used=%s pressure=%s inference=%s ollama=%s loaded=%s ' "$(date +%T)" \
      "$(sysctl -n vm.swapusage | awk '{print $6}')" "$(sysctl -n kern.memorystatus_vm_pressure_level)" \
      "$(footprint 9500)" "$(footprint 11434)" \
      "$(curl -s 127.0.0.1:9500/health | sed -E 's/.*"loaded":\[([^]]*)\].*/\1/; s/[" ]//g')"
    vm_stat | awk '/Pages free/ {f=$NF} /Pages wired/ {w=$NF} /occupied by compressor/ {c=$NF}
      END {printf "free=%.2fGiB wired=%.2fGiB compressed=%.2fGiB\n", f*16384/2^30, w*16384/2^30, c*16384/2^30}'
    sleep 3
  done
}

# The startup log text the settings must produce, then KEY=VALUE settings on
# top of the instance's own; compose recreates the app when they change.
set_variant() {
  local expected=$1
  shift
  { cat "$instance/dense60.env"; printf '%s\n' "$@"; } > "$instance/variant.env"
  compose up -d app || return 1
  # The new container is up once its log shows these settings; a run under any
  # other settings would be mislabelled, so not seeing them is a failure.
  for _ in $(seq 240); do
    docker logs knowva-dense60-app-1 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' \
      | grep "cloud model client configured" | tail -1 | grep -qF "$expected" \
      && curl -sf $url/api/health >/dev/null && {
        echo "$expected"
        return 0
      }
    sleep 0.5
  done
  echo "the app did not start with: $expected" >&2
  return 1
}

# Unloading ends the runner, so each configuration starts with an empty prompt cache.
unload_gemma() {
  curl -s $O/api/generate -d '{"model":"gemma4:12b-mlx","keep_alive":0}' >/dev/null
  sleep 5
}

# The dense-only configurations never load the reranker; wait until the
# service has released it (KM_IDLE_SECS after its last use), so their
# footprints are measured without it.
wait_rerank_released() {
  for _ in $(seq 900); do
    curl -s 127.0.0.1:9500/health | grep -q '"rerank"' || return 0
    sleep 1
  done
  echo "rerank still loaded" >&2
  return 1
}

sampler=
failed=
finish() {
  local code=$?
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  echo "### finishing $(date +%T)"
  compose stop app
  rm -f "$instance/variant.env"
  if [ "$code" != 0 ] || [ -n "$failed" ]; then echo "E2E_INCOMPLETE: exit $code, failed:${failed:- none}"; exit 1; fi
  echo E2E_DONE
}
trap finish EXIT
trap 'exit 1' INT TERM HUP

# name, the startup log text, then the settings appended to dense60.env.
variant() {
  local name=$1
  shift
  [[ "$selected" == *" $name "* ]] || return 0
  [ -e "$out/ABORT" ] && { failed="$failed aborted"; exit 1; }
  echo "### $name $(date +%T)"
  set_variant "$@" || exit 1
  unload_gemma
  curl -s 127.0.0.1:9500/health > "$out/health-before-$name.json"
  local since offset
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  offset=$(stat -f %z "$ollama_log")
  sample > "$out/memory-$name.log" 2>&1 &
  sampler=$!
  cd "$app" || exit 1
  bun bench/concurrency.ts --url $url --levels 1,2,3,4 --rounds 2 --label "$name" \
    --out "$out/concurrency-$name.json" || failed="$failed $name:concurrency"
  # The datasets have known failing cases, so their exit status is a result.
  bun bench/examples.ts --url $url --out "$out/examples-$name.json"; echo "examples exit $?"
  bun bench/examples.ts --url $url --dataset bench/realistic.json --out "$out/realistic-$name.json"
  echo "realistic exit $?"
  bun bench/examples.ts --url $url --dataset "$here/../identifiers.json" --out "$out/identifiers-$name.json"
  echo "identifiers exit $?"
  kill "$sampler" 2>/dev/null
  sampler=
  curl -s $O/api/ps > "$out/ps-$name.json"
  curl -s 127.0.0.1:9500/health > "$out/health-$name.json"
  tail -c +$((offset + 1)) "$ollama_log" > "$out/ollama-$name.log"
  docker logs --since "$since" knowva-dense60-app-1 > "$out/app-$name.log" 2>&1
}

wait_rerank_released || exit 1
variant dense-6 'retrieval_mode="dense" rerank_enabled=false rerank_candidate_top_k=50 rerank_top_n=6 ' \
  RETRIEVAL_MODE=dense RERANK_ENABLED=false RERANK_TOP_N=6
variant dense-8 'retrieval_mode="dense" rerank_enabled=false rerank_candidate_top_k=50 rerank_top_n=8 ' \
  RETRIEVAL_MODE=dense RERANK_ENABLED=false RERANK_TOP_N=8
variant "dense-rerank-$pool" \
  "retrieval_mode=\"dense\" rerank_enabled=true rerank_candidate_top_k=$pool rerank_top_n=6 " \
  RETRIEVAL_MODE=dense RERANK_ENABLED=true RERANK_CANDIDATE_TOP_K="$pool"
variant current 'retrieval_mode="hybrid" rerank_enabled=true rerank_candidate_top_k=50 rerank_top_n=6 ' \
  RETRIEVAL_MODE=hybrid RERANK_ENABLED=true
