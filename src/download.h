// lightwhisper - model resolution & download
//
// The program "chooses the model that runs locally": by default it uses the
// tiny whisper model (ggml-tiny.bin, ~75 MB, 39M params), fetched
// automatically from the ggml-org Hugging Face repo on first use. Larger
// models can be selected explicitly with --model.
#pragma once

#include <string>

namespace lw {

struct ModelInfo {
    bool        found      = false; // resolved to an existing file?
    bool        downloaded = false; // had to download it?
    std::string path;      // absolute path of the model file
    std::string alias;     // the alias that was used (e.g. "tiny")
    std::string message;   // human-readable note
};

// Resolve a model. <model_arg> is either:
//   - a filesystem path (anything with '/' or ending in .bin/.ggml/.gguf)
//   - an alias: auto, tiny, tiny.en, base, base.en, small, small.en,
//     medium, medium.en, large-v3, large-v3-turbo
// <exe_dir> is the directory of the running binary (for models/ next to it).
// When the file does not exist locally it is downloaded from
// https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<file>
// unless <allow_download> is false.
ModelInfo resolve_model(const std::string & model_arg,
                        const std::string & exe_dir,
                        const std::string & model_dir,
                        bool                allow_download);

// Resolve the silero VAD model (ggml-silero-v6.2.0.bin, ~3 MB), which whisper
// uses to split audio into utterances so the transcript can stream out with
// clean per-utterance segments. Downloaded automatically on first use from
// https://huggingface.co/ggml-org/whisper-vad
// Returns "" on failure (with <err> filled).
std::string resolve_vad_model(const std::string & exe_dir,
                              const std::string & model_dir,
                              bool                allow_download,
                              std::string &       err);

} // namespace lw
