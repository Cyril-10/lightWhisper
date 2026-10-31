#!/usr/bin/env bash
# Download a whisper.cpp ggml model manually (the program also does this
# automatically on first run). Default: tiny (~75 MB, 39M params).
#
# Usage: scripts/fetch-model.sh [tiny|tiny.en|base|base.en|small|small.en|...]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-tiny}"
case "${MODEL}" in
    auto|tiny)            FILE="ggml-tiny.bin"            ;;
    tiny.en)              FILE="ggml-tiny.en.bin"         ;;
    base)                 FILE="ggml-base.bin"            ;;
    base.en)              FILE="ggml-base.en.bin"         ;;
    small)                FILE="ggml-small.bin"           ;;
    small.en)             FILE="ggml-small.en.bin"        ;;
    medium)               FILE="ggml-medium.bin"          ;;
    medium.en)            FILE="ggml-medium.en.bin"       ;;
    large-v3)             FILE="ggml-large-v3.bin"        ;;
    large-v3-turbo)       FILE="ggml-large-v3-turbo.bin"  ;;
    *) echo "unknown model: ${MODEL}" >&2; exit 1 ;;
esac

URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${FILE}"
mkdir -p models
echo "==> downloading ${FILE} -> models/${FILE}"
echo "    ${URL}"
if command -v curl >/dev/null 2>&1; then
    curl -L --progress-bar -C - -o "models/${FILE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "models/${FILE}" "${URL}"
else
    echo "error: need curl or wget" >&2
    exit 1
fi
echo "==> done: models/${FILE}"
