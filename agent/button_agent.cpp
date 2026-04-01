// ─── button_agent.cpp ────────────────────────────────────────────────────────
// GPIO button monitoring and voice-activated agent queries.

#include "agent/button_agent.h"
#include "agent/agent.h"
#include "audio/audio.h"
#include "agent/openai_client.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

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

    if (!gpio_exists()) {
        std::cerr << "⚠ GPIO pin " << gpio_pin_ << " not available (button mode disabled)\n";
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

    if (sysfs_fd_ >= 0) {
        close(sysfs_fd_);
        sysfs_fd_ = -1;
    }
}

bool ButtonAgent::gpio_exists()
{
    // Check GPIO pin is in valid range
    if (gpio_pin_ < 0 || gpio_pin_ > 31) {
        return false;
    }
    // On Raspberry Pi, we'll assume GPIO is available
    // In production, could check /sys/class/gpio or /dev/gpiochip0
    return true;
}

int ButtonAgent::read_gpio_value()
{
    // Try sysfs: /sys/class/gpio/gpio{N}/value
    std::string value_path = "/sys/class/gpio/gpio" + std::to_string(gpio_pin_) + "/value";
    std::ifstream value_file(value_path);
    if (value_file.is_open()) {
        int val = 0;
        value_file >> val;
        return val;
    }

    // Fallback: return "not pressed" (for testing without actual GPIO)
    return 1;  // HIGH = not pressed (pull-up mode)
}

void ButtonAgent::monitor_button()
{
    std::cout << "[ButtonAgent] Monitor thread started (GPIO " << gpio_pin_ << ")\n";

    const int PRESS_THRESHOLD_MS = 2000;  // 2 seconds
    const int DEBOUNCE_MS = 50;           // 50 ms debounce
    const int POLL_INTERVAL_MS = 100;     // Check GPIO every 100 ms when unpressed

    while (running_.load()) {
        int current = read_gpio_value();

        // In pull-up mode: 0 = pressed, 1 = not pressed
        if (current == 0) {
            // Button pressed — measure hold duration
            auto press_start = std::chrono::steady_clock::now();
            bool long_press_announced = false;

            while (running_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(DEBOUNCE_MS));
                current = read_gpio_value();

                // Button released?
                if (current != 0) break;

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
