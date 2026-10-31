# lightwhisper

A tiny, self-contained **C++ terminal program** that streams a live transcript
from any audio file using [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
(the C/C++ port of OpenAI's Whisper). Runs 100% locally, on CPU, with the
smallest useful model (whisper `tiny`, 39M params) — fast enough for real-time
use on a Raspberry Pi or an ARM laptop.

```
$ ./lw-whisper meeting.mp3
> loading audio: meeting.mp3
> audio: 45.2 s, 723200 samples @16 kHz
> model: ./models/ggml-tiny.bin | VAD utterances | 8 threads | beam 5 | lang auto

Okay, so let's kick off the design review.
First, the API contract needs to settle the retry policy.
...
> done in 12.3 s (3.67x realtime)
> detected language: en
> transcript: meeting.srt (timed) | meeting.txt (plain)
```

The plain text streams to stdout line by line as segments are decoded; the
*timed* `.srt` and *plain* `.txt` versions are written incrementally at the
same moment — so `tail -f meeting.srt` shows the captions appearing live.
Both formats come from one pass; there are no duplicate or dropped segments.

## Features

- **Any audio file** — mp3, wav, m4a, flac, ogg, opus, aac, ... anything
  ffmpeg can read. 16 kHz audio required by Whisper is handled automatically
  (ffmpeg resamples/remixes). If ffmpeg is *not* installed, plain 16 kHz
  16-bit PCM WAV files still work via a built-in parser.
- **Live streaming output** — the audio is split into utterances at natural
  pauses with the built-in silero VAD, and every utterance is written the
  instant it finishes decoding (whisper's `new_segment_callback`). Voiced
  segments are decoded exactly once: no window-overlap artifacts.
- **Both formats, always in sync** — SRT with timestamps + plain text.
- **Tiny local model by default** — `ggml-tiny.bin` (~75 MB, 39M params) is
  downloaded automatically on first run. Pick any other size with `--model`.
- **Portable binaries** — builds for **x86-64** and **ARM64 (aarch64)** Linux;
  fully statically linked, so the ARM binary runs on your Raspberry Pi or a
  Windows-on-ARM Ubuntu WSL2 box with zero dependencies.
- **Ctrl-C safe** — interrupting mid-file stops decoding cleanly and keeps
  whatever was already streamed.

## Requirements at runtime

| what            | needed for                                                    |
|-----------------|---------------------------------------------------------------|
| ffmpeg          | decoding non-WAV inputs (convert/resample to 16 kHz).         |
|                 | *Optional*: plain 16 kHz 16-bit WAV works without it.         |
| curl or wget    | auto-downloading the model / VAD on first use.                |

Everything else — model, VAD, decoding — is on-board. Nothing is uploaded
anywhere; the binary never phones home after the optional one-time model
download.

## Install / build

You need a C++17 compiler and CMake ≥ 3.16. On Debian/Ubuntu:

```sh
sudo apt install build-essential cmake git ffmpeg
git clone <this repo> lightwhisper && cd lightwhisper
./build.sh                 # -> dist/lw-whisper-<arch>, static & portable
```

First run downloads the model automatically:

```sh
./dist/lw-whisper-x86_64 my_recording.mp3
# or pre-fetch:  ./scripts/fetch-model.sh tiny
```

### Build for ARM64 (aarch64) Linux (e.g. Windows-on-ARM WSL2)

On any x86-64 Linux you can cross-compile the ARM binary:

```sh
sudo apt install g++-aarch64-linux-gnu
./cross-build-aarch64.sh   # -> dist/lw-whisper-aarch64 (static)
```

Copy `dist/lw-whisper-aarch64` to the ARM machine; it needs no libraries
(besides ffmpeg for non-WAV files and curl/wget for first-run downloads).
You can verify it locally with `apt install qemu-user` and
`qemu-aarch64 ./dist/lw-whisper-aarch64 ...`.

To build *natively* on the ARM machine itself, just run `./build.sh` there.

### Optional flags

```
-m, --model <path|alias>   auto | tiny | tiny.en | base | base.en | small | small.en |
                           medium | medium.en | large-v3 | large-v3-turbo | /path/to/model.bin
                           (default: auto = tiny; english-only models end in .en)
    --model-dir <dir>      where auto-downloaded models go (default: ./models or next to the binary)
-l, --language <lang>      en, de, zh, ja, ... (default: auto-detect)
-f, --output <base>        .srt/.txt base name (default: same name as the input)
    --no-vad               use whisper's plain segmentation instead of silero VAD
    --srt-stdout           print SRT blocks to stdout instead of plain text
    --no-files             don't write .srt/.txt files
    --no-download          fail instead of downloading a missing model
-t, --threads <n>          decode threads (default: all cores)
-b, --beam-size <n>        beam size, 1 = greedy (default: 5)
-h, --help
```

## How the streaming works

1. The whole file is decoded to 16 kHz mono floats (ffmpeg subprocess, or the
   built-in WAV parser as a fallback).
2. A single `whisper_full()` pass is run. With VAD (default), silero splits
   the audio into utterance-sized chunks; without VAD, whisper segments on its
   own. Either way there is **one decode pass** — no overlapping sliding
   windows — so nothing is transcribed twice and nothing is dropped.
3. whisper calls `new_segment_callback` the moment each segment is decoded;
   we write it to stdout + `.srt` + `.txt` immediately, then keep going.

Latency is one utterance (typical caption behavior): with pauses ≈ 0.5 s or
more you see each line as it finishes; for a wall of unbroken speech it comes
out in one block.

## Models

| alias     | params  | ggml size | realtime speed (CPU, 4+ threads) |
|-----------|---------|-----------|----------------------------------|
| **tiny** (default) | 39M | ~75 MB | ~5–15x |
| base      | 74M     | ~142 MB   | ~3–8x   |
| small     | 244M    | ~466 MB   | ~1–3x   |
| medium    | 769M    | ~1.5 GB   | ~0.5–1.5x |
| large-v3  | 1550M   | ~2.9 GB   | memory-heavy |

`tiny.en` / `base.en` / `small.en` are English-only variants of the same size
that are a bit more accurate and faster for English audio.

## Project layout

```
CMakeLists.txt              build config (portable/static knobs)
build.sh                    native build -> dist/lw-whisper-<arch>
cross-build-aarch64.sh      ARM64 cross-compile -> dist/lw-whisper-aarch64
cmake/aarch64-linux-gnu.toolchain.cmake
scripts/fetch-model.sh      manual model download
src/main.cpp                CLI + streaming core
src/audio.cpp/.h            ffmpeg decode + fallback WAV parser
src/download.cpp/.h         model & VAD resolution / auto-download
third_party/whisper.cpp     vendored whisper.cpp (MIT)
```

## Notes & limits

- Works best on speech; music/effects may produce garbage (same as Whisper).
- The whole file is held in memory as 16 kHz floats (~1.9 MB/minute).
- Timestamps are per-utterance (0.3–10+ s), which is ideal for SRT captions.
- License: the wrapper here is public domain (Unlicense); whisper.cpp is MIT.
  See `third_party/whisper.cpp/LICENSE`.
