#!/bin/bash
# Acceptance run for the Mac mini's current inference settings. Answers
# examples.json and realistic.json with the app's MODEL_RERANK_URL pointed at
# capture-proxy.ts through a compose override, which records every rerank
# request and response; then recreates the app without the override and runs
# bench/concurrency.ts. Samples memory throughout. ~/knowva/.env and the
# launchd jobs are not changed, and the override is removed on any exit.
#   nohup bash capture-e2e.sh <results dir> > <results dir>/e2e.log 2>&1 &
set -uo pipefail
export PATH="$HOME/.proto/shims:$HOME/.proto/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
out=$(realpath "$1")
here=$(cd "$(dirname "$0")" && pwd)
app=$HOME/knowva-inference/app
logs=$HOME/Library/Logs/knowva

cat > "$out/capture.override.yml" <<'YML'
services:
  app:
    environment:
      MODEL_RERANK_URL: http://host.docker.internal:9501/v1/ranking
YML

wait_app() {
  for _ in $(seq 240); do curl -sf 127.0.0.1:8080/api/health >/dev/null && return 0; sleep 0.5; done
  echo "app did not come up" >&2
  return 1
}
rerank_url() {
  docker inspect knowva-app-1 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^MODEL_RERANK_URL='
}
footprint() {
  local pid
  pid=$(lsof -ti "tcp:$1" -sTCP:LISTEN | head -1)
  [ -n "$pid" ] && top -l 1 -pid "$pid" -stats mem | tail -1 | tr -d ' ' || echo -
}
sample() {
  while :; do
    printf '%s swap_used=%s pressure=%s inference=%s ollama=%s ' "$(date +%T)" \
      "$(sysctl -n vm.swapusage | awk '{print $6}')" "$(sysctl -n kern.memorystatus_vm_pressure_level)" \
      "$(footprint 9500)" "$(footprint 11434)"
    vm_stat | awk '/Pages free/ {f=$NF} /Pages wired/ {w=$NF} /occupied by compressor/ {c=$NF}
      END {printf "free=%.2fGiB wired=%.2fGiB compressed=%.2fGiB\n", f*16384/2^30, w*16384/2^30, c*16384/2^30}'
    sleep 3
  done
}

proxy=
sampler=
failed=
direct() {
  (cd ~/knowva && docker compose -p knowva up -d) && wait_app || return 1
  if rerank_url; then echo "MODEL_RERANK_URL is still set" >&2; return 1; fi
}
finish() {
  local code=$?
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  echo "### finishing $(date +%T)"
  local ok=1
  rerank_url >/dev/null && { direct || ok=; }
  [ -n "$proxy" ] && kill "$proxy" 2>/dev/null
  if [ -z "$ok" ]; then echo RESTORE_FAILED; exit 1; fi
  if [ "$code" != 0 ] || [ -n "$failed" ]; then echo "E2E_INCOMPLETE: exit $code, failed:${failed:- none}"; exit 1; fi
  echo E2E_DONE
}
trap finish EXIT
trap 'exit 1' INT TERM HUP

curl -s 127.0.0.1:9500/health > "$out/health-before.json"; cat "$out/health-before.json"; echo
curl -s 127.0.0.1:11434/api/ps > "$out/ps-before.json"
inference_offset=$(stat -f %z "$logs/inference.log")
ollama_offset=$(stat -f %z "$logs/ollama.log")
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sample > "$out/memory.log" 2>&1 &
sampler=$!

bun "$here/capture-proxy.ts" --out "$out/ranking" > "$out/proxy.log" 2>&1 &
proxy=$!
for _ in $(seq 40); do curl -sf 127.0.0.1:9501/health >/dev/null && break; sleep 0.5; done
curl -sf 127.0.0.1:9501/health >/dev/null || { echo "proxy did not come up" >&2; exit 1; }
(cd ~/knowva && docker compose -p knowva -f docker-compose.yml -f "$out/capture.override.yml" up -d) && wait_app || exit 1
rerank_url || exit 1

cd "$app" || exit 1
echo "### examples $(date +%T)"
# Both datasets have known failing cases, so their exit status is a result.
bun bench/examples.ts --out "$out/examples.json"; echo "examples exit $?"
echo "### realistic $(date +%T)"
bun bench/examples.ts --dataset bench/realistic.json --out "$out/realistic.json"; echo "realistic exit $?"
echo "ranking requests captured: $(ls "$out/ranking" | wc -l)"

# Recreating the app drops this container's log.
docker logs --since "$since" knowva-app-1 > "$out/app-capture.log" 2>&1
echo "### direct routing $(date +%T)"
direct || exit 1
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
kill "$proxy" 2>/dev/null; proxy=
echo "### concurrency $(date +%T)"
bun bench/concurrency.ts --levels 1,2,3,4 --rounds 2 --label mlx-rerank --out "$out/concurrency.json" || failed="$failed concurrency"

curl -s 127.0.0.1:9500/health > "$out/health-after.json"
curl -s 127.0.0.1:11434/api/ps > "$out/ps-after.json"
tail -c +$((inference_offset + 1)) "$logs/inference.log" > "$out/inference.log"
tail -c +$((ollama_offset + 1)) "$logs/ollama.log" > "$out/ollama.log"
docker logs --since "$since" knowva-app-1 > "$out/app.log" 2>&1
