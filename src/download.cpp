#include "download.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace lw {

namespace {

constexpr const char * HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && 0 == strcmp(s.c_str() + s.size() - n, suffix);
}

// quote a path for a POSIX shell command line
std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

bool dir_exists(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool file_exists(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool make_dir(const std::string & p) {
    return mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
}

// Shell out to curl or wget (whichever exists) to download <url> to <dest>.
bool download_file(const std::string & url, const std::string & dest, std::string & err) {
    const std::string qurl  = shell_quote(url);
    const std::string qdest = shell_quote(dest);

    if (system(nullptr) == 0) {
        err = "no shell available for download";
        return false;
    }

    // curl first (more common on modern distros, supports resume with -C -)
    int rc = system(("curl -L --fail --progress-bar -C - -o " + qdest + " " + qurl).c_str());
    if (rc == 0) return true;
    // some systems have wget only
    if (WIFEXITED(rc) && WEXITSTATUS(rc) == 127) {
        rc = system(("wget -O " + qdest + " " + qurl).c_str());
        if (rc == 0) return true;
    }

    err = "download failed (curl/wget returned " + std::to_string(rc) + ")\n"
          "You can download it manually:\n  " + url + "\n  -> " + dest;
    return false;
}

} // namespace

ModelInfo resolve_model(const std::string & model_arg,
                        const std::string & exe_dir,
                        const std::string & model_dir,
                        bool                allow_download) {
    ModelInfo info;

    const bool looks_like_path =
        model_arg.find('/') != std::string::npos ||
        (model_arg.size() > 4 &&
         (ends_with(model_arg, ".bin") || ends_with(model_arg, ".ggml") || ends_with(model_arg, ".gguf")));

    std::string file;
    if (looks_like_path) {
        info.alias = model_arg;
        if (file_exists(model_arg)) {
            info.path  = model_arg;
            info.found = true;
            return info;
        }
        info.message = "model file not found: '" + model_arg + "'";
        return info;
    }

    // alias -> file name
    static const struct { const char * alias; const char * file; } kAliases[] = {
        { "auto",            "ggml-tiny.bin"            },
        { "tiny",            "ggml-tiny.bin"            },
        { "tiny.en",         "ggml-tiny.en.bin"         },
        { "base",            "ggml-base.bin"            },
        { "base.en",         "ggml-base.en.bin"         },
        { "small",           "ggml-small.bin"           },
        { "small.en",        "ggml-small.en.bin"        },
        { "medium",          "ggml-medium.bin"          },
        { "medium.en",       "ggml-medium.en.bin"       },
        { "large-v3",        "ggml-large-v3.bin"        },
        { "large-v3-turbo",  "ggml-large-v3-turbo.bin"  },
    };

    const char * file_name = nullptr;
    for (const auto & a : kAliases) {
        if (model_arg == a.alias) {
            file_name = a.file;
            info.alias = a.alias;
            break;
        }
    }
    if (!file_name) {
        info.message = "unknown model '" + model_arg + "' (use a path, or one of: " +
                       "auto, tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v3, large-v3-turbo)";
        return info;
    }
    file = file_name;

    // look for an existing copy: <model_dir> first, then <exe_dir>/models
    std::vector<std::string> candidates;
    if (!model_dir.empty()) candidates.push_back(model_dir + "/" + file);
    if (!exe_dir.empty())   candidates.push_back(exe_dir + "/models/" + file);
    candidates.push_back("./models/" + file);

    for (const auto & c : candidates) {
        if (file_exists(c)) {
            info.path  = c;
            info.found = true;
            return info;
        }
    }

    if (!allow_download) {
        info.message = "model '" + file + "' not found (searched: " +
                       [&] {
                           std::string s;
                           for (const auto & c : candidates) { if (!s.empty()) s += ", "; s += c; }
                           return s;
                       }() +
                       "). Re-run without --no-download to fetch it automatically.";
        return info;
    }

    // download into the first writable candidate directory (default models/)
    std::string dest_dir = model_dir;
    if (dest_dir.empty() || !dir_exists(dest_dir)) dest_dir = "./models";
    if (!make_dir(dest_dir)) dest_dir = exe_dir + "/models";

    const std::string dest = dest_dir + "/" + file;
    fprintf(stderr, "> downloading %s model (%s) ...\n", file.c_str(), file.c_str());
    fprintf(stderr, "  %s%s\n", HF_BASE, file.c_str());

    std::string err;
    if (!download_file(std::string(HF_BASE) + file, dest, err)) {
        info.message = err;
        return info;
    }

    info.path       = dest;
    info.downloaded = true;
    info.found      = true;
    info.message    = "downloaded to " + dest;
    return info;
}

std::string resolve_vad_model(const std::string & exe_dir,
                              const std::string & model_dir,
                              bool                allow_download,
                              std::string &       err) {
    constexpr const char * VAD_REPO = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/";
    constexpr const char * FILE     = "ggml-silero-v6.2.0.bin";

    // candidates: explicit model dir, next to the binary, cwd
    std::vector<std::string> candidates;
    if (!model_dir.empty()) candidates.push_back(model_dir + "/" + FILE);
    if (!exe_dir.empty())   candidates.push_back(exe_dir + "/models/" + FILE);
    candidates.push_back(std::string("./models/") + FILE);

    for (const auto & c : candidates) {
        if (file_exists(c)) return c;
    }

    if (!allow_download) {
        err = std::string("VAD model '") + FILE + "' not found (use --no-download only if it exists locally)";
        return "";
    }

    std::string dest_dir = model_dir;
    if (dest_dir.empty() || !dir_exists(dest_dir)) dest_dir = "./models";
    if (!make_dir(dest_dir)) dest_dir = exe_dir + "/models";

    const std::string dest = dest_dir + "/" + FILE;
    fprintf(stderr, "> downloading VAD model (silero-v6.2.0, ~3 MB) ...\n");
    fprintf(stderr, "  %s%s\n", VAD_REPO, FILE);

    if (!download_file(std::string(VAD_REPO) + FILE, dest, err)) {
        return "";
    }
    fprintf(stderr, "> VAD model: %s\n", dest.c_str());
    return dest;
}

} // namespace lw
