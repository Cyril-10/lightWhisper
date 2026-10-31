// lightwhisper - audio input
// Loads any audio file whisper can accept as 16 kHz mono f32 PCM.
//
// Strategy:
//   1. If ffmpeg is available on PATH: decode via
//        ffmpeg -v error -nostdin -i <file> -f f32le -ac 1 -ar 16000 -
//      which handles every container/codec/bit-depth/sample-rate/channel count.
//   2. Otherwise fall back to a built-in minimal WAV parser that accepts
//      16-bit PCM, 16000 Hz, mono or stereo WAV files (the classic
//      whisper.cpp sample format). Anything else is rejected with a hint.
#pragma once

#include <string>
#include <vector>

namespace lw {

// Load <path> into <pcm> as 16 kHz mono float samples in [-1, 1].
// Returns true on success; fills <err> with a human readable reason on failure.
bool load_audio(const std::string & path, std::vector<float> & pcm, std::string & err);

} // namespace lw
