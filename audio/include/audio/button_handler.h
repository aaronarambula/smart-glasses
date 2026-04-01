#pragma once

// ─── button_handler.h ────────────────────────────────────────────────────────
// Physical button input handler with multi-state FSM.
//
// Button states:
//   - Released (idle)
//   - Held 0–2 seconds (charging for listen mode)
//   - Held 2–5 seconds (listening — recording audio input)
//   - Held >5 seconds (emergency stop)
//
// Events:
//   - on_button_press() → FSM transitions to "charging"
//   - on_button_release() → FSM checks hold duration and triggers action
//   - State checks occur periodically (10 ms polling or interrupt-driven)
//
// Callbacks registered by the caller:
//   - on_ready_to_listen() — fires at 2 seconds (play audio cue, start recording)
//   - on_listen_complete() — fires on release after 2-second hold (stop recording, send to agent)
//   - on_emergency_stop() — fires if held >5 seconds (shutdown or safe state)

#include "sensors/gpio.h"

#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <string>

namespace audio {

// ─── ButtonState (internal FSM) ───────────────────────────────────────────────

enum class ButtonState {
    Idle,              // Button not held
    Charging,          // Button held, < 2 seconds
    ListeningReady,    // Button held, >= 2 seconds, waiting for more or release
    EmergencyTriggered // Button held >5 seconds
};

// ─── ButtonHandler ────────────────────────────────────────────────────────────
//
// Manages button input and invokes callbacks on state transitions.

class ButtonHandler {
public:
    // Callback types.
    using OnReadyCallback = std::function<void()>;       // 2-second hold reached
    using OnListenCompleteCallback = std::function<void()>; // Released after 2-second hold
    using OnEmergencyCallback = std::function<void()>;    // 5-second hold reached

    explicit ButtonHandler(uint32_t button_gpio_pin);
    ~ButtonHandler();

    // Non-copyable.
    ButtonHandler(const ButtonHandler&)            = delete;
    ButtonHandler& operator=(const ButtonHandler&) = delete;

    // Lifecycle.
    bool open();   // Initialize GPIO
    bool start();  // Begin polling thread
    void stop();   // Stop polling
    void close();  // Release GPIO

    // Register callbacks.
    void set_on_ready(OnReadyCallback cb) {
        on_ready_ = cb;
    }

    void set_on_listen_complete(OnListenCompleteCallback cb) {
        on_listen_complete_ = cb;
    }

    void set_on_emergency(OnEmergencyCallback cb) {
        on_emergency_ = cb;
    }

    // Status.
    bool        is_open()        const { return is_open_; }
    bool        is_running()      const { return is_running_; }
    ButtonState current_state()  const;
    uint32_t    gpio_pin()        const { return button_pin_; }
    std::string error_message()   const;

    // For testing: manually trigger state transitions.
    void simulate_press();
    void simulate_release();

private:
    void polling_loop();
    void handle_state_transition(ButtonState new_state,
                                  std::chrono::milliseconds hold_duration);

    uint32_t button_pin_;
    bool     is_open_;
    std::atomic<bool> is_running_;

    std::unique_ptr<sensors::DigitalInput> button_;

    std::thread polling_thread_;
    std::atomic<ButtonState> state_;

    std::chrono::steady_clock::time_point press_time_;

    OnReadyCallback on_ready_;
    OnListenCompleteCallback on_listen_complete_;
    OnEmergencyCallback on_emergency_;

    mutable std::string error_;

    // Debounce: ignore presses that occur within this time of last release
    static constexpr std::chrono::milliseconds DEBOUNCE_MS{ 50 };
    std::chrono::steady_clock::time_point last_release_time_;
};

} // namespace audio
