#pragma once

// ─── audio_capture.h ─────────────────────────────────────────────────────────
// Microphone audio capture for voice input.
//
// Records raw PCM audio during button listen activation.
// Saves to a file or passes directly to transcription service.
//
// Typical usage:
//   AudioCapture recorder;
//   recorder.start_recording("user_input.wav");
//   // Button held for 2 seconds...
//   auto audio_data = recorder.stop_recording();
//   // Send to OpenAI Whisper API

#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace audio {

// ─── AudioBuffer ─────────────────────────────────────────────────────────────
//
// In-memory audio data holder.

struct AudioBuffer {
    std::vector<uint8_t> pcm_data;
    uint32_t sample_rate;  // Hz, typically 16000
    uint16_t channels;     // 1 = mono, 2 = stereo
    uint16_t bit_depth;    // 16 = 16-bit samples
    
    // Return total duration in seconds
    float duration_seconds() const {
        if (sample_rate == 0 || channels == 0 || bit_depth == 0) {
            return 0.0f;
        }
        size_t bytes_per_sample = (bit_depth + 7) / 8;
        size_t total_samples = pcm_data.size() / (bytes_per_sample * channels);
        return static_cast<float>(total_samples) / sample_rate;
    }
};

// ─── AudioCapture ────────────────────────────────────────────────────────────
//
// Records microphone input to WAV file or raw buffer.
//
// On Raspberry Pi:
//   Requires ALSA (Advanced Linux Sound Architecture).
//   Device: "default" or "hw:0,0" (varies by setup).
//   Command-based recording (arecord via subprocess) for simplicity.
//
// On non-Pi:
//   Stubbed; returns a silent buffer.

class AudioCapture {
public:
    // Construct with audio device name and parameters.
    // device: ALSA device name, e.g., "default" or "hw:0,0"
    // sample_rate: Hz, typically 16000 for speech recognition
    // channels: 1 = mono, 2 = stereo
    // bit_depth: 16 = 16-bit
    AudioCapture(const std::string& device = "default",
                 uint32_t sample_rate = 16000,
                 uint16_t channels = 1,
                 uint16_t bit_depth = 16);
    ~AudioCapture();

    // Non-copyable.
    AudioCapture(const AudioCapture&)            = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    // Start recording to a temporary file or buffer.
    // duration_seconds: approximate recording duration before auto-stop.
    // Returns true on success, false on error.
    bool start_recording(float duration_seconds = 5.0f);

    // Stop recording and return the captured audio.
    // Returns an AudioBuffer; check size() to verify data was captured.
    AudioBuffer stop_recording();

    // Get the last recorded audio without stopping.
    AudioBuffer get_current_buffer() const;

    // Status.
    bool        is_recording() const { return recording_; }
    std::string error_message() const;

    // Configuration getters.
    const std::string& device() const { return device_; }
    uint32_t sample_rate() const { return sample_rate_; }
    uint16_t channels() const { return channels_; }
    uint16_t bit_depth() const { return bit_depth_; }

private:
    // Platform-specific implementation.
#ifdef __linux__
    bool start_recording_linux(float duration_seconds);
    AudioBuffer read_audio_linux();
#else
    bool start_recording_stub(float duration_seconds);
    AudioBuffer read_audio_stub();
#endif

    std::string device_;
    uint32_t sample_rate_;
    uint16_t channels_;
    uint16_t bit_depth_;
    bool recording_;
    std::string temp_file_;
    std::string error_;
};

} // namespace audio
