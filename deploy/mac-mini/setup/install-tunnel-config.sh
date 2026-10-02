#!/bin/bash
# Installs cloudflared/config.yml as the tunnel's configuration and restarts
# the tunnel daemon. The file in this repository is the source of truth; the
# previous one is kept beside it as config.yml.previous.
#
#   sudo bash ~/knowva-inference/app/setup/install-tunnel-config.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

src=$(cd "$(dirname "$0")/.." && pwd)/cloudflared/config.yml
dest=/etc/cloudflared/config.yml

/usr/local/bin/cloudflared tunnel --config "$src" ingress validate
cp -p "$dest" "$dest.previous"
install -o root -g wheel -m 600 "$src" "$dest"
echo "installed $dest"

# Restarted in the background: when this runs over the tunnel itself, the
# restart cuts the session it was started from. Check afterwards with
#   grep 'Registered tunnel connection' /Library/Logs/cloudflared.log | tail -4
nohup sh -c 'sleep 2; launchctl kickstart -k system/com.cloudflare.cloudflared' >/dev/null 2>&1 &
echo "tunnel restarting; an SSH session through it drops for a few seconds"
