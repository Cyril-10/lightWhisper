#!/usr/bin/env bash
# Build lightwhisper for the current machine (x86_64 or aarch64 Linux).
# Produces a portable, fully static binary in dist/.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="$(uname -m)"
BUILD_DIR="build-${ARCH}"
STATIC="${LW_STATIC:-ON}"          # LW_STATIC=OFF -> dynamically linked
PORTABLE="${LW_PORTABLE:-ON}"      # LW_PORTABLE=OFF -> -march=native, faster

echo "==> building lw-whisper for ${ARCH} (static=${STATIC} portable=${PORTABLE})"
cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLW_STATIC="${STATIC}" \
    -DLW_PORTABLE="${PORTABLE}" \
    >/dev/null
cmake --build "${BUILD_DIR}" -j"$(nproc)"

mkdir -p dist
cp "${BUILD_DIR}/lw-whisper" "dist/lw-whisper-${ARCH}"
echo "==> done: $(pwd)/dist/lw-whisper-${ARCH}"
file "dist/lw-whisper-${ARCH}"
