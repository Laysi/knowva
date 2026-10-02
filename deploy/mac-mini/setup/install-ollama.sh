#!/bin/bash
# Installs the pinned Ollama release under ~/knowva-inference/ollama and points
# `current` at it, which is the path the LaunchDaemon runs. The previous
# version stays beside it, so rolling back is re-pointing the link.
set -euo pipefail

VERSION=0.35.0
SHA256=2608dbb0a0f0136a198db9d48b4f74ece55f452314a39452fca35b7cf20c2589

root=$HOME/knowva-inference/ollama
mkdir -p "$root/v$VERSION"
archive=$(mktemp)
trap 'rm -f "$archive"' EXIT

curl -fsSL --retry 3 -o "$archive" \
  "https://github.com/ollama/ollama/releases/download/v$VERSION/ollama-darwin.tgz"
echo "$SHA256  $archive" | shasum -a 256 -c - >/dev/null || { echo "checksum mismatch" >&2; exit 1; }
tar -xzf "$archive" -C "$root/v$VERSION"
ln -sfn "v$VERSION" "$root/current"
"$root/current/ollama" --version 2>&1 | tail -1
