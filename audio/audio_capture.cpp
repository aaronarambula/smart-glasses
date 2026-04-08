// ─── audio_capture.cpp ───────────────────────────────────────────────────────
// Audio capture implementation using ALSA on Linux.

#include "audio/audio_capture.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <sysexits.h>

#ifdef __linux__
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <signal.h>
#endif

namespace audio {

// ─── WAV file helper (reserved for future use) ──────────────────────────────

// struct WAVHeader { ... };  // Defined but not currently used in stub implementation

// ─── Construction / Destruction ──────────────────────────────────────────────

AudioCapture::AudioCapture(const std::string& device,
                           uint32_t sample_rate,
                           uint16_t channels,
                           uint16_t bit_depth)
    : device_(device)
    , sample_rate_(sample_rate)
    , channels_(channels)
    , bit_depth_(bit_depth)
    , recording_(false)
    , error_("")
{
}

AudioCapture::~AudioCapture() {
    if (recording_) {
        stop_recording();
    }
}

// ─── Recording Control ───────────────────────────────────────────────────────

bool AudioCapture::start_recording(float duration_seconds) {
    if (recording_) {
        error_ = "Recording already in progress";
        return false;
    }

#ifdef __linux__
    return start_recording_linux(duration_seconds);
#else
    return start_recording_stub(duration_seconds);
#endif
}

AudioBuffer AudioCapture::stop_recording() {
    if (!recording_) {
        return AudioBuffer();
    }

    recording_ = false;

#ifdef __linux__
    return read_audio_linux();
#else
    return read_audio_stub();
#endif
}

AudioBuffer AudioCapture::get_current_buffer() const {
    // Not implemented in this simple version.
    // Would require access to the temporary file being written.
    return AudioBuffer();
}

std::string AudioCapture::error_message() const {
    return error_;
}

// ─── Linux Implementation ────────────────────────────────────────────────────

#ifdef __linux__

bool AudioCapture::start_recording_linux(float duration_seconds) {
    // Generate temporary filename.
    static int counter = 0;
    std::ostringstream oss;
    oss << "/tmp/audio_capture_" << getpid() << "_" << (counter++) << ".wav";
    temp_file_ = oss.str();

    // Build arecord command.
    // arecord: command-line ALSA recorder
    // -d N: duration in seconds
    // -r N: sample rate
    // -c N: channels
    // -f S16_LE: 16-bit signed little-endian
    // -t wav: WAV output format
    std::ostringstream cmd;
    cmd << "arecord"
        << " -D " << device_
        << " -r " << sample_rate_
        << " -c " << channels_
        << " -f S16_LE"
        << " -t wav"
        << " -d " << static_cast<int>(duration_seconds)
        << " '" << temp_file_ << "'"
        << " 2>/dev/null";

    // Fork and execute arecord in background.
    pid_t pid = fork();
    if (pid < 0) {
        error_ = "fork() failed";
        return false;
    }

    if (pid == 0) {
        // Child process: execute arecord
        execlp("/bin/sh", "sh", "-c", cmd.str().c_str(), nullptr);
        _exit(EX_UNAVAILABLE);
    }

    // Parent process: return success, background process will record.
    recording_ = true;
    return true;
}

AudioBuffer AudioCapture::read_audio_linux() {
    // Wait for arecord to finish (it will stop when duration expires).
    // In a real implementation, we'd track the child PID and waitpid().
    // For now, assume the temporary file exists and is complete.

    AudioBuffer buffer;
    buffer.sample_rate = sample_rate_;
    buffer.channels = channels_;
    buffer.bit_depth = bit_depth_;

    std::ifstream file(temp_file_, std::ios::binary);
    if (!file.is_open()) {
        error_ = "Failed to open recorded audio file: " + temp_file_;
        return buffer;
    }

    // Read raw PCM data (skip 44-byte WAV header).
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(44, std::ios::beg);  // Back to after header
    std::streamsize data_size = file_size - 44;

    if (data_size > 0) {
        buffer.pcm_data.resize(data_size);
        file.read(reinterpret_cast<char*>(buffer.pcm_data.data()), data_size);
    }

    file.close();

    // Clean up temporary file.
    unlink(temp_file_.c_str());
    temp_file_ = "";

    return buffer;
}

#else

// ─── Non-Linux Stub Implementation ───────────────────────────────────────────

bool AudioCapture::start_recording_stub(float /* duration_seconds */) {
    // Non-Linux: pretend to record; return a silent buffer.
    recording_ = true;
    return true;
}

AudioBuffer AudioCapture::read_audio_stub() {
    AudioBuffer buffer;
    buffer.sample_rate = sample_rate_;
    buffer.channels = channels_;
    buffer.bit_depth = bit_depth_;

    // Create silent buffer (16-bit PCM zeros).
    size_t samples = sample_rate_ * channels_;  // 1 second of silence
    size_t bytes_per_sample = bit_depth_ / 8;
    buffer.pcm_data.resize(samples * bytes_per_sample, 0);

    return buffer;
}

#endif

} // namespace audio
