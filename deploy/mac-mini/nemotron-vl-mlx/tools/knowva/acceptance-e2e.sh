#!/bin/bash
# Acceptance run for the Mac mini's current inference settings, with nothing
# changed: answers examples.json and realistic.json, runs bench/concurrency.ts
# at 1-4 users, then uploads copies of two realistic files (stress-upload.ts,
# which deletes them again) while 4 users ask questions. Samples memory and
# the footprints of the inference service and Ollama throughout.
#   nohup bash acceptance-e2e.sh <results dir> <label> > <results dir>/e2e.log 2>&1 &
set -uo pipefail
export PATH="$HOME/.proto/shims:$HOME/.proto/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
out=$(realpath "$1")
label=$2
here=$(cd "$(dirname "$0")" && pwd)
app=$HOME/knowva-inference/app
files=$HOME/knowva-inference/realistic
logs=$HOME/Library/Logs/knowva
# Names this run's upload copies, so cleanup touches nothing else.
prefix="stress-$label-$(date +%Y%m%d%H%M%S)-"

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

sampler=
failed=
finish() {
  local code=$?
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  echo "### finishing $(date +%T)"
  (cd "$app" && bun "$here/stress-upload.ts" --cleanup --prefix "$prefix") || failed="$failed cleanup"
  curl -s 127.0.0.1:9500/health > "$out/health-after.json"
  curl -s 127.0.0.1:11434/api/ps > "$out/ps-after.json"
  tail -c +$((inference_offset + 1)) "$logs/inference.log" > "$out/inference.log"
  tail -c +$((ollama_offset + 1)) "$logs/ollama.log" > "$out/ollama.log"
  docker logs --since "$since" knowva-app-1 > "$out/app.log" 2>&1
  if [ "$code" != 0 ] || [ -n "$failed" ]; then echo "E2E_INCOMPLETE: exit $code, failed:${failed:- none}"; exit 1; fi
  echo E2E_DONE
}

curl -s 127.0.0.1:9500/health > "$out/health-before.json"; cat "$out/health-before.json"; echo
curl -s 127.0.0.1:11434/api/ps > "$out/ps-before.json"
inference_offset=$(stat -f %z "$logs/inference.log")
ollama_offset=$(stat -f %z "$logs/ollama.log")
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
trap finish EXIT
trap 'exit 1' INT TERM HUP
sample > "$out/memory.log" 2>&1 &
sampler=$!

# proto resolves bun's version from the app's .prototools.
cd "$app" || exit 1
echo "### examples $(date +%T)"
# Both datasets have known failing cases, so their exit status is a result.
bun bench/examples.ts --out "$out/examples.json"; echo "examples exit $?"
echo "### realistic $(date +%T)"
bun bench/examples.ts --dataset bench/realistic.json --out "$out/realistic.json"; echo "realistic exit $?"
echo "### concurrency $(date +%T)"
bun bench/concurrency.ts --levels 1,2,3,4 --rounds 2 --label "$label" --out "$out/concurrency.json" \
  || failed="$failed concurrency"

echo "### stress $(date +%T)"
bun "$here/stress-upload.ts" --prefix "$prefix" "$files/Machine Guarding.pptx" \
  "$files/OSHA 3080 Hand and Power Tools.pdf" > "$out/upload.jsonl" 2> "$out/upload.err" &
uploader=$!
bun bench/concurrency.ts --levels 4 --rounds 3 --label "$label-stress" --out "$out/concurrency-stress.json" \
  || failed="$failed stress:concurrency"
wait "$uploader" || failed="$failed stress:upload"
cat "$out/upload.jsonl"
