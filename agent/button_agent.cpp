// ─── button_agent.cpp ────────────────────────────────────────────────────────
// GPIO button monitoring and voice-activated agent queries.

#include "agent/button_agent.h"
#include "agent/agent.h"
#include "audio/audio.h"
#include "agent/openai_client.h"
#include "sensors/gpio.h"

#include <iostream>
#include <chrono>
#include <memory>

namespace agent {

// Use the full audio namespace
using audio::AudioSystem;

ButtonAgent::ButtonAgent(int gpio_pin,
                         AgentSystem* agent_sys,
                         AudioSystem* audio_sys)
    : gpio_pin_(gpio_pin), agent_(agent_sys), audio_(audio_sys)
{
}

ButtonAgent::~ButtonAgent()
{
    stop();
}

void ButtonAgent::start()
{
    if (running_.load()) return;

    // Initialize GPIO input using the proper abstraction (pull-up mode: LOW = pressed)
    button_input_ = std::make_unique<sensors::DigitalInput>(gpio_pin_, true);
    
    if (!button_input_->open()) {
        std::cerr << "⚠ Failed to open GPIO pin " << gpio_pin_ << ": "
                  << button_input_->error_message() << "\n";
        button_input_.reset();
        return;
    }

    std::cout << "  ✓ Button input on GPIO " << gpio_pin_ << " (2-sec hold to activate)\n";

    running_.store(true);
    monitor_thread_ = std::make_unique<std::thread>([this] { monitor_button(); });
}

void ButtonAgent::stop()
{
    if (!running_.load()) return;

    running_.store(false);
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

    if (button_input_) {
        button_input_->close();
        button_input_.reset();
    }
}

void ButtonAgent::monitor_button()
{
    std::cout << "[ButtonAgent] Monitor thread started (GPIO " << gpio_pin_ << ")\n";

    if (!button_input_) {
        std::cerr << "[ButtonAgent] Button input not initialized\n";
        return;
    }

    const int PRESS_THRESHOLD_MS = 2000;  // 2 seconds
    const int DEBOUNCE_MS = 50;           // 50 ms debounce
    const int POLL_INTERVAL_MS = 100;     // Check GPIO every 100 ms when unpressed

    while (running_.load()) {
        // DigitalInput::read() returns true if the pin reads HIGH
        // In pull-up mode: LOW (pressed) returns false, HIGH (not pressed) returns true
        bool is_pressed = !button_input_->read();

        if (is_pressed) {
            // Button pressed — measure hold duration
            auto press_start = std::chrono::steady_clock::now();
            bool long_press_announced = false;

            while (running_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(DEBOUNCE_MS));
                is_pressed = !button_input_->read();

                // Button released?
                if (!is_pressed) break;

                // Check hold duration
                auto now = std::chrono::steady_clock::now();
                auto held_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - press_start).count();

                // At 2 seconds: activate query mode
                if (held_ms >= PRESS_THRESHOLD_MS && !long_press_announced) {
                    long_press_announced = true;
                    query_mode_.store(true);

                    if (audio_) {
                        // Speak: "Listening for query"
                        audio_->deliver_agent_advice("Listening. What is your question?");
                    }

                    std::cout << "\n[ButtonAgent] ▶ QUERY MODE ACTIVE\n";
                    std::cout << "[ButtonAgent] Type your question and press Enter:\n> ";
                    std::cout.flush();

                    // Read query from stdin
                    std::string query;
                    if (std::getline(std::cin, query) && !query.empty()) {
                        std::cout << "[ButtonAgent] Processing: \"" << query << "\"\n";

                        if (agent_ && agent_->is_enabled()) {
                            // Send direct query to GPT-4o
                            std::cout << "[ButtonAgent] Querying GPT-4o...\n";

                            try {
                                auto response = agent_->client().query_direct(query);
                                std::cout << "[ButtonAgent] Response: " << response << "\n";

                                if (audio_) {
                                    audio_->deliver_agent_advice(response);
                                }
                            } catch (const std::exception& e) {
                                std::cerr << "[ButtonAgent] Query failed: " << e.what() << "\n";
                                if (audio_) {
                                    audio_->deliver_agent_advice("Query failed. Please try again.");
                                }
                            }
                        } else {
                            std::cerr << "[ButtonAgent] Agent not available\n";
                            if (audio_) {
                                audio_->deliver_agent_advice("Agent not available. Check your API key.");
                            }
                        }
                    }

                    query_mode_.store(false);
                    std::cout << "[ButtonAgent] Query mode deactivated\n";
                    break;
                }
            }
        } else {
            // Button not pressed, sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }
    }

    std::cout << "[ButtonAgent] Monitor thread stopped\n";
}

} // namespace agent
