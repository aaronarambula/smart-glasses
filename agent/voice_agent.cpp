// ─── voice_agent.cpp ─────────────────────────────────────────────────────────
// Voice-enabled agent implementation.

#include "agent/voice_agent.h"

#include <iostream>

namespace agent {

// ─── Construction / Destruction ──────────────────────────────────────────────

VoiceAgent::VoiceAgent(AgentLoop& agent_loop,
                       audio::AudioCapture& audio_capture)
    : agent_loop_(agent_loop)
    , audio_capture_(audio_capture)
    , pending_question_("")
{
}

VoiceAgent::~VoiceAgent() {
}

// ─── Voice Input Handling ────────────────────────────────────────────────────

void VoiceAgent::on_user_question(const std::string& transcribed_text) {
    // Store the user question for the next agent query.
    // The scene builder will incorporate this into the GPT context.
    pending_question_ = transcribed_text;

    // Trigger an immediate query (override cooldown).
    // This ensures the user's question gets a response quickly.
    // In a real implementation, we'd call:
    //   agent_loop_.request_immediate_query();
    // For now, the question is stored and picked up by the normal query cycle.
}

void VoiceAgent::transcribe_audio(const audio::AudioBuffer& audio) {
    // Transcribe audio to text and call on_user_question().
    std::string text = transcribe_with_whisper(audio);

    if (!text.empty()) {
        if (on_transcription_complete_) {
            on_transcription_complete_(text);
        }
        on_user_question(text);
    }
}

void VoiceAgent::clear_pending_question() {
    pending_question_ = "";
}

std::string VoiceAgent::pending_question() const {
    return pending_question_;
}

// ─── Transcription (Placeholder for Whisper API) ─────────────────────────────

std::string VoiceAgent::transcribe_with_whisper(const audio::AudioBuffer& audio) {
    // Placeholder: would call OpenAI Whisper API.
    //
    // Expected curl usage:
    //   POST https://api.openai.com/v1/audio/transcriptions
    //   Multipart form-data:
    //     file: <binary PCM data>
    //     model: "whisper-1"
    //     language: "en"
    //
    // Response: { "text": "What should I do?" }
    //
    // For now, return a mock transcription.

    if (audio.pcm_data.empty()) {
        return "";
    }

    // TODO: Implement actual Whisper API call.
    // For development, return a dummy string.
    std::cout << "[VoiceAgent] Received " << audio.pcm_data.size()
              << " bytes of audio (" << audio.duration_seconds() << "s)\n";
    std::cout << "[VoiceAgent] TODO: Integrate OpenAI Whisper API for transcription\n";

    return "";
}

} // namespace agent
