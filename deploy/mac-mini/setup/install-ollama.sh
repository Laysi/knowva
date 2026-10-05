#!/bin/bash
# Installs the pinned Ollama release under ~/knowva-inference/ollama, rebuilds
# its Go binary from the same commit with ollama/prefix-cache-budget.patch
# into v<version>-km1 (a copy of the release, so the MLX and llama.cpp
# libraries are the released ones), and points `current` at the rebuild,
# which is the path the LaunchDaemon runs. The plain release stays beside it,
# so rolling back is re-pointing the link. Restart the daemon afterwards.
#
# Needs the Command Line Tools (clang) and proto's Go from ../.prototools.
set -euo pipefail

VERSION=0.35.0
SHA256=2608dbb0a0f0136a198db9d48b4f74ece55f452314a39452fca35b7cf20c2589
# The release tag's commit; a moved tag fails the build instead of shipping
# different code.
COMMIT=cc4069396f3ad2c370c53eed2e4a42ac13adab84
BUILD=km1

app=$(cd "$(dirname "$0")/.." && pwd)
root=$HOME/knowva-inference/ollama
release=$root/v$VERSION
patched=$root/v$VERSION-$BUILD
mkdir -p "$root"
work=$(mktemp -d)
stage=$root/.v$VERSION.new
trap 'rm -rf "$work" "$stage" "$patched/.ollama.new"' EXIT

curl -fsSL --retry 3 -o "$work/ollama-darwin.tgz" \
  "https://github.com/ollama/ollama/releases/download/v$VERSION/ollama-darwin.tgz"
echo "$SHA256  $work/ollama-darwin.tgz" | shasum -a 256 -c - >/dev/null || { echo "checksum mismatch" >&2; exit 1; }
rm -rf "$stage"
mkdir "$stage"
tar -xzf "$work/ollama-darwin.tgz" -C "$stage"
# Swapped in by renames, never written over: a daemon still running the old
# copy keeps its files.
if [ -d "$release" ]; then
  rm -rf "$release.old"
  mv "$release" "$release.old"
fi
mv "$stage" "$release"
rm -rf "$release.old"

[ -d "$patched" ] || cp -cR "$release" "$patched"

git clone -q --depth 1 --branch "v$VERSION" https://github.com/ollama/ollama.git "$work/src"
[ "$(git -C "$work/src" rev-parse HEAD)" = "$COMMIT" ] || { echo "v$VERSION is not $COMMIT" >&2; exit 1; }
git -C "$work/src" apply "$app/ollama/prefix-cache-budget.patch"
# Run from $app so proto picks Go from its .prototools; -C builds the clone.
# The flags are upstream's darwin build (cmake/local.cmake, scripts/build_darwin.sh).
(cd "$app" && CGO_ENABLED=1 \
  CGO_CFLAGS="-O3 -mmacosx-version-min=14.0" \
  CGO_LDFLAGS="-lc++ -framework Metal -framework Foundation -framework Accelerate -mmacosx-version-min=14.0" \
  proto exec go -- go build -C "$work/src" -trimpath \
    -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$VERSION-$BUILD -X=github.com/ollama/ollama/server.mode=release" \
    -o "$patched/.ollama.new" .)
mv -f "$patched/.ollama.new" "$patched/ollama"
ln -sfn "v$VERSION-$BUILD" "$root/current"
"$root/current/ollama" --version 2>&1 | tail -1
