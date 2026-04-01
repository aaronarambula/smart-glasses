// ─── button_handler.cpp ──────────────────────────────────────────────────────
// Button handler implementation with FSM and debounce logic.

#include "audio/button_handler.h"

#include <iostream>
#include <thread>

namespace audio {

// ─── Construction / Destruction ──────────────────────────────────────────────

ButtonHandler::ButtonHandler(uint32_t button_gpio_pin)
    : button_pin_(button_gpio_pin)
    , is_open_(false)
    , is_running_(false)
    , state_(ButtonState::Idle)
    , last_release_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10))
{
    button_ = std::make_unique<sensors::DigitalInput>(button_gpio_pin, true);
}

ButtonHandler::~ButtonHandler() {
    stop();
    close();
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

bool ButtonHandler::open() {
    if (is_open_) return true;

    if (!button_ || !button_->open()) {
        error_ = "Failed to open button GPIO " + std::to_string(button_pin_);
        return false;
    }

    is_open_ = true;
    return true;
}

bool ButtonHandler::start() {
    if (!is_open_) {
        error_ = "Cannot start; button not open";
        return false;
    }

    if (is_running_) return true;

    is_running_ = true;
    polling_thread_ = std::thread([this] { polling_loop(); });
    return true;
}

void ButtonHandler::stop() {
    if (!is_running_) return;

    is_running_ = false;
    if (polling_thread_.joinable()) {
        polling_thread_.join();
    }
}

void ButtonHandler::close() {
    stop();

    if (button_) button_->close();

    is_open_ = false;
}

// ─── Status ──────────────────────────────────────────────────────────────────

ButtonState ButtonHandler::current_state() const {
    return state_.load();
}

std::string ButtonHandler::error_message() const {
    return error_;
}

// ─── Polling Loop ─────────────────────────────────────────────────────────────
//
// Polls the button GPIO pin every 10 ms and manages FSM state transitions.

void ButtonHandler::polling_loop() {
    bool was_pressed = false;

    while (is_running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (!button_) continue;

        bool is_pressed = button_->read();

        // Detect press (rising edge) and debounce
        if (is_pressed && !was_pressed) {
            auto now = std::chrono::steady_clock::now();
            auto time_since_release = now - last_release_time_;

            // Debounce: ignore if too soon after last release
            if (time_since_release > DEBOUNCE_MS) {
                state_ = ButtonState::Charging;
                press_time_ = now;
            }
        }

        // Detect release (falling edge)
        if (!is_pressed && was_pressed) {
            auto now = std::chrono::steady_clock::now();
            auto hold_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - press_time_);

            last_release_time_ = now;

            ButtonState old_state = state_.exchange(ButtonState::Idle);

            if (old_state == ButtonState::EmergencyTriggered) {
                // Already at emergency; just transition back to idle
                if (on_emergency_) {
                    on_emergency_();
                }
            } else if (hold_ms.count() >= 2000) {
                // At least 2 seconds held → listened mode triggered
                if (on_listen_complete_) {
                    on_listen_complete_();
                }
            }
        }

        // Check for emergency (5+ seconds)
        was_pressed = is_pressed;
        if (is_pressed) {
            auto now = std::chrono::steady_clock::now();
            auto hold_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - press_time_);

            ButtonState current = state_.load();

            // Transition to ready at 2 seconds
            if (current == ButtonState::Charging && hold_ms.count() >= 2000) {
                state_ = ButtonState::ListeningReady;
                if (on_ready_) {
                    on_ready_();
                }
            }

            // Transition to emergency at 5 seconds
            if ((current == ButtonState::Charging || 
                 current == ButtonState::ListeningReady) &&
                hold_ms.count() >= 5000) {
                state_ = ButtonState::EmergencyTriggered;
                if (on_emergency_) {
                    on_emergency_();
                }
            }
        }
    }
}

// ─── State Transition Handler ─────────────────────────────────────────────────

void ButtonHandler::handle_state_transition(ButtonState new_state,
                                             std::chrono::milliseconds hold_duration)
{
    switch (new_state) {
        case ButtonState::ListeningReady:
            if (on_ready_) {
                on_ready_();
            }
            break;

        case ButtonState::Idle:
            // Check what state we're transitioning FROM based on hold duration
            if (hold_duration.count() >= 5000) {
                if (on_emergency_) {
                    on_emergency_();
                }
            } else if (hold_duration.count() >= 2000) {
                if (on_listen_complete_) {
                    on_listen_complete_();
                }
            }
            break;

        case ButtonState::Charging:
        case ButtonState::EmergencyTriggered:
            // No special action on entry to these states
            break;
    }
}

// ─── Testing / Simulation ────────────────────────────────────────────────────

void ButtonHandler::simulate_press() {
    state_ = ButtonState::Charging;
    press_time_ = std::chrono::steady_clock::now();
}

void ButtonHandler::simulate_release() {
    auto hold_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - press_time_);

    ButtonState old_state = state_.exchange(ButtonState::Idle);

    if (old_state == ButtonState::EmergencyTriggered) {
        // Already at emergency; just transition back to idle
    } else if (hold_duration.count() >= 2000) {
        // At least 2 seconds held → listened mode triggered
    }

    handle_state_transition(ButtonState::Idle, hold_duration);
}

} // namespace audio
