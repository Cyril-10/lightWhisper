// lightwhisper - lightweight local whisper transcriber
//
// Usage: lw-whisper [options] <audio-file>
//
// Streams a transcript from any audio file whisper can accept:
//   * plain text -> stdout, one segment per line, as soon as it is decoded
//   * SRT        -> <output>.srt, written incrementally with timestamps
//   * plain text -> <output>.txt, written incrementally too
//
// The default model is whisper "tiny" (39M params, ~75 MB ggml file), which is
// downloaded automatically on first use -- small enough to run locally on a
// Raspberry Pi class machine or an ARM laptop, at many times realtime.
//
// How "live" streaming works: the audio is decoded once with whisper's built-in
// silero VAD, which splits it into utterances at natural pauses. Every time an
// utterance finishes decoding, whisper fires a callback and we print/write it
// immediately -- so the transcript streams out progressively, in caption-sized
// chunks, with clean per-utterance timestamps and no duplicated text. Because
// each segment is decoded exactly once (no sliding windows / overlap), the SRT
// output is lossless and definitive. Drop `--no-vad` to use whisper's own
// segmentation instead (still streamed segment-by-segment via the callback).
#include "audio.h"
#include "download.h"
#include "whisper.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace lw; // audio.h / download.h helpers

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------
struct Options {
    std::string model      = "auto";       // alias or path
    std::string language   = "auto";       // "auto" = detect
    std::string input;                     // audio file
    std::string output_base;               // for .srt / .txt ("" = derive)
    std::string model_dir;                 // where to put downloaded models

    int n_threads = std::max(1, (int) std::thread::hardware_concurrency());
    int beam_size = 5;                     // 1 = greedy

    bool vad          = true;              // split into utterances with silero VAD
    bool srt_stdout   = false;             // also print SRT blocks to stdout
    bool no_files     = false;             // don't write .srt/.txt files
    bool no_download  = false;             // fail instead of downloading models
};

static void usage(const char * argv0, FILE * out = stderr) {
    fprintf(out,
        "\n"
        "usage: %s [options] <audio-file>\n"
        "\n"
        "Streams a transcript from any audio file (mp3, wav, m4a, flac, ogg, ...)\n"
        "that ffmpeg can decode. Plain text streams to stdout; <output>.srt (timed)\n"
        "and <output>.txt (plain) are written incrementally as segments finalize.\n"
        "\n"
        "options:\n"
        "  -m, --model <path|alias>   model: 'auto' (default), tiny, tiny.en, base,\n"
        "                             base.en, small, small.en, medium, medium.en,\n"
        "                             large-v3, large-v3-turbo, or a path to a .bin\n"
        "                             Downloads ggml-tiny.bin automatically if missing.\n"
        "      --model-dir <dir>      directory for auto-downloaded models\n"
        "                             (default: ./models or next to the binary)\n"
        "  -l, --language <lang>      spoken language code, e.g. en, de, zh, ja\n"
        "                             (default: auto-detect)\n"
        "  -f, --output <base>        output base for .srt/.txt (default: input name)\n"
        "      --no-vad               use whisper's plain segmentation instead of\n"
        "                             silero VAD utterance splitting\n"
        "      --srt-stdout           print SRT blocks to stdout instead of plain text\n"
        "      --no-files             do not write .srt/.txt files\n"
        "      --no-download          do not download a missing model\n"
        "  -t, --threads <n>          decode threads (default: %d)\n"
        "  -b, --beam-size <n>        beam size, 1 = greedy (default %d)\n"
        "  -h, --help                 show this help\n"
        "\n",
        argv0, (int) std::thread::hardware_concurrency(), 5);
}

static bool parse_args(int argc, char ** argv, Options & opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", what);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help")            { usage(argv[0], stdout); exit(0); }
        else if (a == "-m" || a == "--model")      { opt.model = next("--model"); }
        else if (a == "--model-dir")               { opt.model_dir = next("--model-dir"); }
        else if (a == "-l" || a == "--language")   { opt.language = next("--language"); }
        else if (a == "-f" || a == "--output")     { opt.output_base = next("--output"); }
        else if (a == "--no-vad")                  { opt.vad = false; }
        else if (a == "--srt-stdout")              { opt.srt_stdout = true; }
        else if (a == "--no-files")                { opt.no_files = true; }
        else if (a == "--no-download")             { opt.no_download = true; }
        else if (a == "-t" || a == "--threads")    { opt.n_threads = std::stoi(next("--threads")); }
        else if (a == "-b" || a == "--beam-size")  { opt.beam_size = std::stoi(next("--beam-size")); }
        else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            usage(argv[0]);
            exit(1);
        }
        else {
            if (!opt.input.empty()) {
                fprintf(stderr, "error: multiple input files given ('%s' and '%s')\n", opt.input.c_str(), a.c_str());
                exit(1);
            }
            opt.input = a;
        }
    }
    if (opt.input.empty()) {
        usage(argv[0]);
        exit(1);
    }
    if (opt.n_threads < 1) opt.n_threads = 1;
    if (opt.beam_size < 1) opt.beam_size = 1;
    return true;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::string dir_of(const std::string & path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static std::string exe_dir() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = '\0';
    return dir_of(buf);
}

static std::string strip_ext(const std::string & path) {
    const size_t pos = path.find_last_of('.');
    const size_t sep = path.find_last_of('/');
    if (pos != std::string::npos && (sep == std::string::npos || pos > sep)) {
        return path.substr(0, pos);
    }
    return path;
}

static std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && isspace((unsigned char) s[b])) ++b;
    while (e > b && isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

// SRT timestamp: HH:MM:SS,mmm
static std::string srt_time(int64_t ms) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d",
             (int) (ms / 3600000), (int) ((ms / 60000) % 60),
             (int) ((ms / 1000) % 60), (int) (ms % 1000));
    return buf;
}

// ---------------------------------------------------------------------------
// global state for Ctrl-C handling
// ---------------------------------------------------------------------------
static std::atomic<bool> g_abort{false};

static void on_sigint(int) {
    // first Ctrl-C: stop cleanly and flush what we have; second one: die now
    signal(SIGINT, SIG_DFL);
    g_abort.store(true);
}

// ggml contract: return true when the computation should be aborted
static bool abort_check(void *) {
    return g_abort.load();
}

// ---------------------------------------------------------------------------
// streaming output
// ---------------------------------------------------------------------------
struct Output {
    std::ofstream srt;  // <base>.srt
    std::ofstream txt;  // <base>.txt
    int srt_index = 0;

    bool srt_stdout = false;

    // Emit one finished segment. Times are absolute milliseconds.
    void emit(const std::string & text, int64_t t0, int64_t t1) {
        const std::string line = trim(text);
        if (line.empty()) return;

        ++srt_index;

        if (srt.is_open() || srt_stdout) {
            const std::string block =
                std::to_string(srt_index) + "\n" +
                srt_time(t0) + " --> " + srt_time(t1) + "\n" +
                line + "\n\n";
            if (srt.is_open()) srt << block << std::flush;
            if (srt_stdout) { printf("%s", block.c_str()); fflush(stdout); }
        }

        if (txt.is_open()) { txt << line << "\n" << std::flush; }

        if (!srt_stdout) { printf("%s\n", line.c_str()); fflush(stdout); }
    }
};

// whisper calls this each time a new segment finishes decoding; we read every
// segment produced since the previous call and stream it out immediately.
struct StreamCtx {
    Output * out   = nullptr;
    int prev_segs  = 0;
};

static void on_new_segment(struct whisper_context *, struct whisper_state * state,
                           int /*n_new*/, void * user) {
    auto * sc = (StreamCtx *) user;
    if (!sc->out) return;

    const int n = whisper_full_n_segments_from_state(state);
    for (int i = sc->prev_segs; i < n; ++i) {
        // whisper.cpp reports segment times in 10 ms units
        sc->out->emit(whisper_full_get_segment_text_from_state(state, i),
                      10 * whisper_full_get_segment_t0_from_state(state, i),
                      10 * whisper_full_get_segment_t1_from_state(state, i));
    }
    sc->prev_segs = n;
}

// ---------------------------------------------------------------------------
// progress
// ---------------------------------------------------------------------------
struct ProgressCtx {
    int64_t total = 0; // total samples
};

static void progress_cb(struct whisper_context *, struct whisper_state *, int p, void * user) {
    auto * pc = (ProgressCtx *) user;
    const int64_t done = (int64_t) (p / 100.0 * pc->total);
    fprintf(stderr, "\r\33[2Ktranscribing... %3d%%  (%lld s / %lld s)",
            p, (long long) (done / WHISPER_SAMPLE_RATE),
            (long long) (pc->total / WHISPER_SAMPLE_RATE));
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) return 1;

    const std::string exe = exe_dir();

    // ---- model resolution / download --------------------------------------
    const ModelInfo model = resolve_model(opt.model, exe, opt.model_dir, !opt.no_download);
    if (!model.found) {
        fprintf(stderr, "error: %s\n", model.message.c_str());
        return 1;
    }
    if (model.downloaded) fprintf(stderr, "%s\n", model.message.c_str());

    std::string vad_model;
    if (opt.vad) {
        std::string err;
        vad_model = resolve_vad_model(exe, opt.model_dir, !opt.no_download, err);
        if (vad_model.empty()) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
    }

    // ---- load audio --------------------------------------------------------
    std::vector<float> pcm;
    {
        std::string err;
        fprintf(stderr, "> loading audio: %s\n", opt.input.c_str());
        if (!load_audio(opt.input, pcm, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "> audio: %.1f s, %zu samples @16 kHz\n",
                (double) pcm.size() / WHISPER_SAMPLE_RATE, pcm.size());
    }

    // ---- init whisper ------------------------------------------------------
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // local CPU only - stays tiny & portable

    whisper_context * ctx = whisper_init_from_file_with_params(model.path.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", model.path.c_str());
        return 2;
    }

    // multilingual model check
    if (!whisper_is_multilingual(ctx) && opt.language != "en" && opt.language != "auto") {
        fprintf(stderr, "> warning: model is English-only; forcing language=en\n");
        opt.language = "en";
    }

    // ---- validate language ------------------------------------------------
    if (opt.language != "auto" && whisper_lang_id(opt.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", opt.language.c_str());
        whisper_free(ctx);
        return 1;
    }

    // ---- output files ------------------------------------------------------
    const std::string base = opt.output_base.empty() ? strip_ext(opt.input) : opt.output_base;
    Output out;
    out.srt_stdout = opt.srt_stdout;
    if (!opt.no_files) {
        out.srt.open(base + ".srt", std::ios::out | std::ios::trunc);
        out.txt.open(base + ".txt", std::ios::out | std::ios::trunc);
        if (!out.srt.is_open()) {
            fprintf(stderr, "error: cannot open '%s.srt' for writing\n", base.c_str());
            whisper_free(ctx);
            return 1;
        }
        if (!out.txt.is_open()) {
            fprintf(stderr, "error: cannot open '%s.txt' for writing\n", base.c_str());
            whisper_free(ctx);
            return 1;
        }
        fprintf(stderr, "> writing: %s.srt | %s.txt\n", base.c_str(), base.c_str());
    }

    // ---- decoding params ---------------------------------------------------
    whisper_full_params wparams = whisper_full_default_params(
        opt.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;   // we draw our own progress line
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.translate        = false;
    wparams.no_context       = false;   // whisper keeps context across the file
    wparams.single_segment   = false;
    wparams.language         = opt.language.c_str();
    wparams.n_threads        = opt.n_threads;
    wparams.beam_search.beam_size = opt.beam_size;
    wparams.audio_ctx        = 0;       // use full audio context

    wparams.vad              = opt.vad;
    wparams.vad_model_path   = opt.vad ? vad_model.c_str() : nullptr;
    wparams.vad_params       = whisper_vad_default_params();

    wparams.new_segment_callback       = on_new_segment;
    wparams.new_segment_callback_user_data = nullptr;
    wparams.progress_callback          = progress_cb;
    wparams.progress_callback_user_data = nullptr;
    wparams.abort_callback             = abort_check;
    wparams.abort_callback_user_data   = nullptr;

    StreamCtx sc;
    sc.out = &out;
    wparams.new_segment_callback_user_data = &sc;

    ProgressCtx pc;
    pc.total = (int64_t) pcm.size();
    wparams.progress_callback_user_data = &pc;

    signal(SIGINT, on_sigint);

    fprintf(stderr,
            "> model: %s | %s | %d threads | beam %d | lang %s\n\n",
            model.path.c_str(), opt.vad ? "VAD utterances" : "whisper segmentation",
            opt.n_threads, opt.beam_size, opt.language.c_str());

    // ---- transcribe (segments stream out via on_new_segment) ---------------
    const auto t_start = std::chrono::steady_clock::now();
    const int rc = whisper_full(ctx, wparams, pcm.data(), (int) pcm.size());

    // whisper may assemble final segments after the callback; emit any leftover
    {
        const int n = whisper_full_n_segments(ctx);
        for (int i = sc.prev_segs; i < n; ++i) {
            out.emit(whisper_full_get_segment_text(ctx, i),
                     10 * whisper_full_get_segment_t0(ctx, i),
                     10 * whisper_full_get_segment_t1(ctx, i));
        }
        sc.prev_segs = n;
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t_end - t_start).count();
    const double audio_secs = (double) pcm.size() / WHISPER_SAMPLE_RATE;
    const double speed     = secs > 0 ? audio_secs / secs : 0.0;

    fprintf(stderr, "\n> done in %.1f s (%.2fx realtime)\n", secs, speed);
    if (g_abort.load()) {
        fprintf(stderr, "> interrupted.\n");
    } else if (rc != 0) {
        fprintf(stderr, "> transcription failed (code %d)\n", rc);
    } else {
        const char * lang = whisper_full_lang_id(ctx) >= 0 ? whisper_lang_str(whisper_full_lang_id(ctx)) : "?";
        fprintf(stderr, "> detected language: %s\n", lang);
        if (!opt.no_files) {
            fprintf(stderr, "> transcript: %s.srt (timed) | %s.txt (plain)\n", base.c_str(), base.c_str());
        }
    }

    whisper_free(ctx);
    return 0;
}
