#pragma once

// ─── voice_agent.h ───────────────────────────────────────────────────────────
// Extensions to AgentLoop for voice input handling.
//
// Provides:
//   - Voice query submission (via button handler)
//   - Integration with audio capture module
//   - Transcription (placeholder for Whisper API integration)
//
// Typical usage:
//   auto voice_agent = std::make_unique<VoiceAgent>(agent_loop, audio_capture);
//   voice_agent->on_user_question("What should I do?");
//   // → sends to GPT with user question in context
//   // → response spoken via TTS

#include "agent_loop.h"
#include "audio/audio_capture.h"
#include "audio/audio.h"

#include <string>
#include <memory>
#include <functional>

namespace agent {

// ─── VoiceAgent ──────────────────────────────────────────────────────────────
//
// Wraps AgentLoop and AudioCapture to handle voice input queries from the button.

class VoiceAgent {
public:
    explicit VoiceAgent(AgentLoop& agent_loop,
                       audio::AudioCapture& audio_capture);
    ~VoiceAgent();

    // Non-copyable.
    VoiceAgent(const VoiceAgent&)            = delete;
    VoiceAgent& operator=(const VoiceAgent&) = delete;

    // Submit a transcribed user question for processing.
    // The question will be included in the next GPT query along with the
    // current scene context, overriding the default scene-only prompt.
    void on_user_question(const std::string& transcribed_text);

    // Handle raw audio data and transcribe to text.
    // This would call OpenAI Whisper API or a local transcriber.
    // For now, it's a placeholder that accepts raw audio.
    void transcribe_audio(const audio::AudioBuffer& audio);

    // Clear any pending user question (e.g., on emergency stop).
    void clear_pending_question();

    // Get the current user question (if any).
    std::string pending_question() const;

    // Callback: called when transcription is complete.
    using OnTranscriptionComplete = std::function<void(const std::string&)>;
    void set_transcription_callback(OnTranscriptionComplete cb) {
        on_transcription_complete_ = cb;
    }

private:
    AgentLoop&                      agent_loop_;
    audio::AudioCapture&            audio_capture_;
    mutable std::string             pending_question_;
    OnTranscriptionComplete         on_transcription_complete_;

    // Placeholder for actual Whisper API integration
    std::string transcribe_with_whisper(const audio::AudioBuffer& audio);
};

} // namespace agent
