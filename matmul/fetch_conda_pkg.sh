#!/usr/bin/env bash
# Fetch a conda-format .conda/.tar.bz2 package and extract it into $2.
# Used to pull individual CUDA *tools* out of NVIDIA's conda channel without
# installing conda. We only need tiny packages (cuobjdump, nvdisasm).
set -euo pipefail
pkg_url="$1"; dest="$2"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sL "$pkg_url" -o "$tmp/$(basename "$pkg_url")"
mkdir -p "$tmp/x"
pkg="$tmp/$(basename "$pkg_url")"
case "$pkg" in
  *.conda)
    ( cd "$tmp/x" && unzip -q "$pkg" )
    inner=$(ls "$tmp/x"/pkg-*.tar.zst 2>/dev/null | head -1 || true)
    [ -n "$inner" ] && zstd -dq -c "$inner" | tar -x -C "$tmp/x"
    ;;
  *.tar.bz2) tar -xjf "$tmp/pkg" -C "$tmp/x" ;;
  *) echo "unknown package format: $pkg_url" >&2; exit 1 ;;
esac
mkdir -p "$dest"
cp -a "$tmp/x"/. "$dest"/
