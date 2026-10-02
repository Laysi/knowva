#!/bin/bash
# Installs Colima, Lima, the Docker CLI and the Compose plugin from their
# release binaries, each checked against a pinned SHA-256. Nothing goes through
# Homebrew and nothing needs root: everything lands under ~/.local.
#
# The Docker CLI is the only one without a published checksum; its digest was
# pinned from two independent downloads of the same release.
set -euo pipefail

COLIMA_VERSION=0.10.3
COLIMA_SHA256=980ad8bf61a4ca370243f4cb41401a61276dcd2c2502bee7b9b86f9250169f34
LIMA_VERSION=2.2.0
LIMA_SHA256=bbdef91774885a0d05f7b048c4eb89ae2bcf3a0c252ae7ca7934e63df76d93c3
DOCKER_VERSION=29.8.2
DOCKER_SHA256=5ffa2bafbabe073470f0333f4a129599230b73d1170bb61146b3d9b08e1d1325
COMPOSE_VERSION=5.5.1
COMPOSE_SHA256=998735c9b6fe68a4f05895e6ea73d71ad06f9fc7046383ad89e47346781b6af5

opt=$HOME/.local/opt
bin=$HOME/.local/bin
mkdir -p "$opt" "$bin" "$HOME/.docker/cli-plugins"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fetch() { # url sha256 dest
  curl -fsSL --retry 3 -o "$3" "$1"
  echo "$2  $3" | shasum -a 256 -c - >/dev/null || { echo "checksum mismatch: $1" >&2; exit 1; }
}

fetch "https://github.com/lima-vm/lima/releases/download/v$LIMA_VERSION/lima-$LIMA_VERSION-Darwin-arm64.tar.gz" \
  "$LIMA_SHA256" "$work/lima.tgz"
rm -rf "$opt/lima-$LIMA_VERSION" && mkdir -p "$opt/lima-$LIMA_VERSION"
tar -xzf "$work/lima.tgz" -C "$opt/lima-$LIMA_VERSION"
# Colima's dependency check looks for the `lima` wrapper, not only limactl.
ln -sfn "$opt/lima-$LIMA_VERSION/bin/limactl" "$bin/limactl"
ln -sfn "$opt/lima-$LIMA_VERSION/bin/lima" "$bin/lima"

fetch "https://github.com/abiosoft/colima/releases/download/v$COLIMA_VERSION/colima-Darwin-arm64" \
  "$COLIMA_SHA256" "$work/colima"
mkdir -p "$opt/colima-$COLIMA_VERSION"
install -m 755 "$work/colima" "$opt/colima-$COLIMA_VERSION/colima"
ln -sfn "$opt/colima-$COLIMA_VERSION/colima" "$bin/colima"

fetch "https://download.docker.com/mac/static/stable/aarch64/docker-$DOCKER_VERSION.tgz" \
  "$DOCKER_SHA256" "$work/docker.tgz"
mkdir -p "$opt/docker-$DOCKER_VERSION"
tar -xzf "$work/docker.tgz" -C "$work"
install -m 755 "$work/docker/docker" "$opt/docker-$DOCKER_VERSION/docker"
ln -sfn "$opt/docker-$DOCKER_VERSION/docker" "$bin/docker"

fetch "https://github.com/docker/compose/releases/download/v$COMPOSE_VERSION/docker-compose-darwin-aarch64" \
  "$COMPOSE_SHA256" "$work/docker-compose"
mkdir -p "$opt/compose-$COMPOSE_VERSION"
install -m 755 "$work/docker-compose" "$opt/compose-$COMPOSE_VERSION/docker-compose"
ln -sfn "$opt/compose-$COMPOSE_VERSION/docker-compose" "$HOME/.docker/cli-plugins/docker-compose"

grep -q '.local/bin' "$HOME/.zshenv" 2>/dev/null ||
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshenv"

"$bin/colima" version | head -1
"$bin/limactl" --version
"$bin/docker" --version
"$bin/docker" compose version
