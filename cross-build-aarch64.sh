#!/usr/bin/env bash
# Cross-compile lightwhisper for 64-bit ARM Linux (aarch64) from an x86_64 host.
# Produces a portable, fully static binary in dist/.
#
# Prereq:  apt install g++-aarch64-linux-gnu
# Test:    apt install qemu-user  &&  qemu-aarch64 ./dist/lw-whisper-aarch64 ...
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    echo "error: aarch64-linux-gnu-g++ not found. Install it, e.g.:" >&2
    echo "  sudo apt install g++-aarch64-linux-gnu" >&2
    exit 1
fi

STATIC="${LW_STATIC:-ON}"
PORTABLE="${LW_PORTABLE:-ON}"      # aarch64: NEON is baseline, this is about ggml flags

echo "==> cross-compiling lw-whisper for aarch64 (static=${STATIC} portable=${PORTABLE})"
cmake -S . -B build-aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.toolchain.cmake \
    -DLW_STATIC="${STATIC}" \
    -DLW_PORTABLE="${PORTABLE}" \
    >/dev/null
cmake --build build-aarch64 -j"$(nproc)"

mkdir -p dist
cp build-aarch64/lw-whisper dist/lw-whisper-aarch64
echo "==> done: $(pwd)/dist/lw-whisper-aarch64"
file dist/lw-whisper-aarch64
