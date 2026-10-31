#include "audio.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <strings.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <vector>

namespace lw {

namespace {

// Find an executable on $PATH. Returns its full path, or "" when absent.
std::string find_in_path(const std::string & name) {
    const char * path_env = getenv("PATH");
    if (!path_env) return "";

    std::string path(path_env);
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find(':', start);
        std::string dir = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (dir.empty()) dir = ".";
        std::string candidate = dir + "/" + name;
        if (access(candidate.c_str(), X_OK) == 0) return candidate;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return "";
}

// Run an argv array, capturing stdout into <out>. Child stderr is inherited so
// the program's own error messages reach the user. Returns the exit status.
int run_and_capture(const std::vector<std::string> & args, std::vector<char> & out) {
    int fds[2];
    if (pipe(fds) != 0) return -1;

    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return -1;
    }

    if (pid == 0) {
        // child
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);

        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto & a : args) argv.push_back(const_cast<char *>(a.c_str()));
        argv.push_back(nullptr);

        execvp(argv[0], argv.data());
        _exit(127); // exec failed (e.g. binary missing)
    }

    close(fds[1]);

    char buf[65536];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
        out.insert(out.end(), buf, buf + n);
    }
    close(fds[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (!WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

bool has_extension(const std::string & path, const char * ext) {
    return path.size() >= strlen(ext) &&
           0 == strcasecmp(path.c_str() + path.size() - strlen(ext), ext);
}

// ---------------------------------------------------------------------------
// Minimal WAV parser: 16-bit PCM, any sample rate, mono/stereo.
// ---------------------------------------------------------------------------
uint32_t rd_u32(const unsigned char * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

uint16_t rd_u16(const unsigned char * p) {
    return (uint16_t)((uint16_t) p[0] | ((uint16_t) p[1] << 8));
}

// Returns 0 on success.
int parse_wav(const std::vector<char> & file, std::vector<float> & pcm, std::string & err) {
    if (file.size() < 12 || memcmp(file.data(), "RIFF", 4) != 0 || memcmp(file.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return 1;
    }

    const unsigned char * d = (const unsigned char *) file.data();
    const size_t size = file.size();

    uint16_t audio_format  = 0;
    uint16_t n_channels    = 0;
    uint32_t sample_rate   = 0;
    uint16_t bits_per_samp = 0;
    const unsigned char * data = nullptr;
    size_t data_bytes = 0;

    size_t off = 12;
    while (off + 8 <= size) {
        const char id[5] = { (char) d[off], (char) d[off + 1], (char) d[off + 2], (char) d[off + 3], '\0' };
        const uint32_t chunk_size = rd_u32(d + off + 4);
        const size_t payload = off + 8;
        if (payload + chunk_size > size) {
            err = "truncated WAV chunk '" + std::string(id) + "'";
            return 1;
        }
        if (strcmp(id, "fmt ") == 0 && chunk_size >= 16) {
            audio_format  = rd_u16(d + payload);
            n_channels    = rd_u16(d + payload + 2);
            sample_rate   = rd_u32(d + payload + 4);
            bits_per_samp = rd_u16(d + payload + 14);
        } else if (strcmp(id, "data") == 0) {
            data = d + payload;
            data_bytes = chunk_size;
        }
        off = payload + chunk_size + (chunk_size & 1); // chunks are word aligned
    }

    if (audio_format != 1) {
        err = "WAV is not uncompressed PCM (format id " + std::to_string(audio_format) +
              "); install ffmpeg or convert to 16-bit PCM";
        return 1;
    }
    if (bits_per_samp != 16) {
        err = "WAV bit depth " + std::to_string(bits_per_samp) + " not supported without ffmpeg; use 16-bit PCM";
        return 1;
    }
    if (n_channels < 1 || n_channels > 2) {
        err = "WAV channel count " + std::to_string(n_channels) + " not supported without ffmpeg; use mono or stereo";
        return 1;
    }
    if (sample_rate != 16000) {
        err = "WAV sample rate " + std::to_string(sample_rate) +
              " Hz not supported without ffmpeg; resample to 16 kHz (ffmpeg -i in -ar 16000 -ac 1 out.wav)";
        return 1;
    }
    if (!data || data_bytes == 0) {
        err = "WAV contains no audio data";
        return 1;
    }

    const size_t n_frames = data_bytes / 2 / n_channels;
    if (n_frames == 0) {
        err = "WAV contains no audio samples";
        return 1;
    }

    pcm.clear();
    pcm.reserve(n_frames);
    for (size_t f = 0; f < n_frames; ++f) {
        int32_t sum = 0;
        for (uint16_t c = 0; c < n_channels; ++c) {
            const size_t i = (f * n_channels + c) * 2;
            int16_t s = (int16_t)((uint16_t) data[i] | ((uint16_t) data[i + 1] << 8));
            sum += s;
        }
        pcm.push_back((float) sum / (n_channels * 32768.0f));
    }
    return 0;
}

} // namespace

bool load_audio(const std::string & path, std::vector<float> & pcm, std::string & err) {
    pcm.clear();

    // --- preferred path: ffmpeg decodes anything ---
    if (!find_in_path("ffmpeg").empty()) {
        std::vector<char> raw;
        const int rc = run_and_capture(
            {"ffmpeg", "-v", "error", "-nostdin", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"},
            raw);

        if (rc == 0) {
            if (raw.size() % sizeof(float) != 0) {
                err = "ffmpeg produced a truncated PCM stream";
                return false;
            }
            pcm.resize(raw.size() / sizeof(float));
            memcpy(pcm.data(), raw.data(), raw.size());
            if (pcm.empty()) {
                err = "audio decoded to 0 samples (empty or silent file?)";
                return false;
            }
            return true;
        }
        if (rc == 127) {
            // exec failed; fall through to the WAV path below
        } else {
            err = "ffmpeg failed to decode '" + path + "' (exit code " + std::to_string(rc) + ")";
            return false;
        }
    }

    // --- fallback: built-in WAV parser (16-bit PCM, 16 kHz, mono/stereo) ---
    if (has_extension(path, ".wav")) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) {
            err = "cannot open '" + path + "': " + std::strerror(errno);
            return false;
        }
        std::vector<char> file;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) file.insert(file.end(), buf, buf + n);
        fclose(f);

        if (parse_wav(file, pcm, err) != 0) {
            err = "failed to read '" + path + "': " + err;
            return false;
        }
        return true;
    }

    err = "cannot decode '" + path + "': ffmpeg is not installed and the file is not a "
          "16 kHz 16-bit PCM WAV. Install ffmpeg (apt install ffmpeg) or convert the file first.";
    return false;
}

} // namespace lw
