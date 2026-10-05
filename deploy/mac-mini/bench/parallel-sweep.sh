#!/bin/bash
# Runs bench/concurrency.ts with Ollama at 1, 2 and 3 parallel slots (context
# checkpoints 2 in all three), then reinstalls the Ollama daemon exactly as it
# was before the run (docs/measurements/README.md §8.12).
#
#   sudo bash ~/knowva-inference/app/bench/parallel-sweep.sh
#
# It detaches at once and keeps going if the SSH session drops. Progress is in
# the sweep.log of the results directory it prints. To stop early, create a
# file named ABORT in that directory and kill the running `bun` (owned by
# knowva); the original daemon is put back either way.
set -uo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

user=knowva
home=/Users/$user
label=io.laysi.knowva.ollama
installed=/Library/LaunchDaemons/$label.plist
app=$home/knowva-inference/app
ollama_log=$home/Library/Logs/knowva/ollama.log
user_path="$home/.proto/shims:$home/.proto/bin:$home/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if [ "${1:-}" != --detached ]; then
  out=$home/knowva-inference/bench-results/parallel-$(date +%Y%m%d-%H%M%S)
  install -d -o "$user" -g staff -m 755 "$out"
  # A new session without a controlling terminal: sudo runs this in its own
  # pty and hangs it up on exit, which would otherwise reach the bench.
  /usr/bin/perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' bash "$0" --detached "$out" \
    > "$out/sweep.log" 2>&1 < /dev/null &
  echo "started (pid $!). Progress: $out/sweep.log"
  exit 0
fi
out=$2

as_user() { sudo -u "$user" -H env PATH="$user_path" "$@"; }

reload() {
  launchctl bootout "system/$label" 2>/dev/null || true
  # bootout returns before launchd has removed the job, and bootstrapping it
  # again while it is still listed fails with "5: Input/output error".
  while launchctl print "system/$label" >/dev/null 2>&1; do sleep 0.5; done
  while lsof -ti tcp:11434 -sTCP:LISTEN >/dev/null; do sleep 0.5; done
  for attempt in 1 2 3 4 5; do
    launchctl bootstrap system "$installed" && break
    [ "$attempt" = 5 ] && return 1
    sleep 2
  done
  for _ in $(seq 240); do
    curl -sf 127.0.0.1:11434/api/version >/dev/null && return 0
    sleep 0.5
  done
  echo "Ollama did not come up within 2 minutes" >&2
  return 1
}

sample() {
  while :; do
    printf '%s swap_used=%s ' "$(date +%T)" "$(sysctl -n vm.swapusage | awk '{print $6}')"
    vm_stat | awk '/Pages free/ {f=$NF} /Pages wired/ {w=$NF} /occupied by compressor/ {c=$NF}
      END {printf "free=%.2fGiB wired=%.2fGiB compressed=%.2fGiB\n", f*16384/2^30, w*16384/2^30, c*16384/2^30}'
    sleep 2
  done
}

sampler=
failed=
restore() {
  local code=$?
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  echo "### restoring the original Ollama daemon $(date +%T)"
  if install -o root -g wheel -m 644 "$out/ollama.plist.orig" "$installed" && reload; then
    launchctl print "system/$label" | grep -E "^\s+(state|pid) ="
    if [ "$code" != 0 ] || [ -n "$failed" ]; then
      echo "SWEEP_INCOMPLETE: exit $code, failed:${failed:- none}"
      exit 1
    fi
    echo SWEEP_DONE
  else
    echo "RESTORE_FAILED: the original Ollama daemon is not serving; see $ollama_log" >&2
    exit 1
  fi
}

cp -p "$installed" "$out/ollama.plist.orig"
trap restore EXIT
trap 'exit 1' INT TERM HUP

for parallel in 1 2 3; do
  [ -e "$out/ABORT" ] && { echo "### ABORT found, stopping"; failed="$failed aborted"; break; }
  echo "### parallel=$parallel $(date +%T)"
  plist=$out/ollama-p$parallel.plist
  cp "$out/ollama.plist.orig" "$plist"
  plutil -replace EnvironmentVariables.OLLAMA_NUM_PARALLEL -string "$parallel" "$plist"
  plutil -replace EnvironmentVariables.LLAMA_ARG_CTX_CHECKPOINTS -string 2 "$plist"
  plutil -lint "$plist" >/dev/null || exit 1
  install -o root -g wheel -m 644 "$plist" "$installed"

  offset=$(stat -f %z "$ollama_log")
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  reload || exit 1
  sample > "$out/memory-p$parallel.log" 2>&1 &
  sampler=$!
  as_user bash -c "cd '$app' && bun bench/concurrency.ts --label p$parallel --out '$out/p$parallel.json'" 2>&1
  status=$?
  echo "bench exit $status"
  # Errors under load are results too, so the later runs still go ahead.
  [ "$status" = 0 ] || failed="$failed p$parallel"
  kill "$sampler" 2>/dev/null
  sampler=

  curl -s 127.0.0.1:11434/api/ps > "$out/ps-p$parallel.json"
  tail -c +$((offset + 1)) "$ollama_log" > "$out/ollama-p$parallel.log"
  as_user docker logs --since "$since" knowva-app-1 > "$out/app-p$parallel.log" 2>&1
  chown "$user":staff "$out"/*
done
