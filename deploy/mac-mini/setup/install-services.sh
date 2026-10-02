#!/bin/bash
# Installs the three LaunchDaemons (Ollama, the inference service, Colima)
# so the appliance comes back on boot without anyone logging in.
#
#   sudo bash ~/knowva-inference/app/setup/install-services.sh
#
# Re-running replaces the daemons with the current plists, which is how a
# changed setting is applied.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

user=knowva
home=/Users/$user
src=$home/knowva-inference/app/launchd
labels=(io.laysi.knowva.ollama io.laysi.knowva.inference io.laysi.knowva.colima)

if grep -l "__[A-Z_]*__" "$src"/*.plist; then
  echo "a plist above still has an unfilled setting" >&2; exit 1
fi
for label in "${labels[@]}"; do
  plutil -lint "$src/$label.plist" >/dev/null
done

install -d -o "$user" -g staff -m 755 "$home/Library/Logs/knowva"

for label in "${labels[@]}"; do
  launchctl bootout "system/$label" 2>/dev/null || true
done
# A server started by hand during setup would hold the port and leave the
# daemon restarting every ThrottleInterval; a VM started by hand would make
# `colima start --foreground` exit at once for the same effect.
for port in 11434 9500; do
  pids=$(lsof -ti "tcp:$port" -sTCP:LISTEN || true)
  if [ -n "$pids" ]; then
    kill $pids
    while lsof -ti "tcp:$port" -sTCP:LISTEN >/dev/null; do sleep 0.5; done
    echo "stopped the process on port $port"
  fi
done
sudo -u "$user" -H env PATH="$home/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  colima stop 2>/dev/null || true

for label in "${labels[@]}"; do
  # launchd opens the log as root before switching user; creating it first
  # keeps it owned by the account that writes to it.
  log=$(plutil -extract StandardOutPath raw "$src/$label.plist")
  [ -e "$log" ] || install -o "$user" -g staff -m 644 /dev/null "$log"

  install -o root -g wheel -m 644 "$src/$label.plist" "/Library/LaunchDaemons/$label.plist"
  launchctl bootstrap system "/Library/LaunchDaemons/$label.plist"
  echo "loaded $label"
done

sleep 5
for label in "${labels[@]}"; do
  launchctl print "system/$label" | grep -E "^\s+(state|pid|last exit code) =" | sed "s/^/$label: /"
done
