#!/bin/bash
# Installs the Xcode Command Line Tools headlessly. proto's uv and bun plugins
# resolve versions with `git ls-remote`, and /usr/bin/git is only a stub until
# these are installed. Run: sudo bash ~/knowva-inference/setup/install-clt.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
if xcode-select -p >/dev/null 2>&1; then
  echo "already installed: $(xcode-select -p)"; exit 0
fi
marker=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
touch "$marker"
trap 'rm -f "$marker"' EXIT
label=$(softwareupdate -l 2>/dev/null | sed -n 's/^\* Label: \(Command Line Tools for Xcode.*\)$/\1/p' | sort -V | tail -1)
[ -n "$label" ] || { echo "no Command Line Tools label offered" >&2; exit 1; }
echo "installing: $label"
softwareupdate -i "$label" --verbose
xcode-select -p
/usr/bin/git --version
