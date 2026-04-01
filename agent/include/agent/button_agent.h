// ─── button_agent.h ──────────────────────────────────────────────────────────
// GPIO button input handler for voice-activated agent queries.
//
// Monitors a GPIO button pin. When held for 2+ seconds:
//   1. Signals "query mode" (ready for input)
//   2. Accepts text query from stdin
//   3. Passes to agent system
//   4. Speaks response via TTS
//
// Thread-safe: runs on separate background thread, signals via atomic flags.
//

#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <string>

namespace sensors {
    class DigitalInput;
}

namespace audio {
    class AudioSystem;
}

namespace agent {

class AgentSystem;

class ButtonAgent {
public:
    ButtonAgent(int gpio_pin = 17,
                AgentSystem* agent_sys = nullptr,
                audio::AudioSystem* audio_sys = nullptr);
    ~ButtonAgent();

    // Start monitoring button in background thread
    void start();

    // Stop monitoring and clean up
    void stop();

    // Check if query mode is active (button held 2+ seconds)
    bool is_query_mode_active() const { return query_mode_.load(); }

    // (Internal) Background thread entry point
    void monitor_button();

private:
    int gpio_pin_;
    AgentSystem* agent_;
    audio::AudioSystem* audio_;

    std::atomic<bool> running_{false};
    std::atomic<bool> query_mode_{false};
    std::unique_ptr<std::thread> monitor_thread_;
    std::unique_ptr<sensors::DigitalInput> button_input_;
};

} // namespace agent
