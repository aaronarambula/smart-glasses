// ─── vibrator.cpp ────────────────────────────────────────────────────────────
// Vibrator controller implementation.

#include "audio/vibrator.h"

#include <iostream>
#include <thread>

namespace audio {

// ─── Construction / Destruction ──────────────────────────────────────────────

VibrationController::VibrationController(uint32_t gpio_pin, bool pwm_capable)
    : gpio_pin_(gpio_pin)
    , pwm_capable_(pwm_capable)
    , is_open_(false)
    , is_running_(false)
    , stop_requested_(false)
{
    if (pwm_capable_) {
        pwm_ = std::make_unique<sensors::PWMOutput>(gpio_pin, 1000, 255);
    } else {
        relay_ = std::make_unique<sensors::DigitalOutput>(gpio_pin, false);
    }
}

VibrationController::~VibrationController() {
    stop();
    close();
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

bool VibrationController::open() {
    if (is_open_) return true;

    bool success = false;
    if (pwm_capable_) {
        success = pwm_ && pwm_->open();
    } else {
        success = relay_ && relay_->open();
    }

    if (!success) {
        error_ = "Failed to open GPIO " + std::to_string(gpio_pin_);
        return false;
    }

    is_open_ = true;
    return true;
}

bool VibrationController::start() {
    if (!is_open_) {
        error_ = "Cannot start; vibrator not open";
        return false;
    }

    if (is_running_) return true;

    is_running_ = true;
    stop_requested_ = false;
    worker_ = std::thread([this] { worker_loop(); });
    return true;
}

void VibrationController::stop() {
    if (!is_running_) return;

    stop_requested_ = true;
    if (worker_.joinable()) {
        worker_.join();
    }
    is_running_ = false;
    stop_immediately();
}

void VibrationController::close() {
    stop();

    if (pwm_) pwm_->close();
    if (relay_) relay_->close();

    is_open_ = false;
}

// ─── Risk Level Handling ─────────────────────────────────────────────────────

void VibrationController::handle_risk_level(prediction::RiskLevel level) {
    auto now = std::chrono::steady_clock::now();
    VibrationRequest req;
    req.deadline = now + std::chrono::milliseconds(5000);

    {
        std::lock_guard<std::mutex> lock(cooldown_mutex_);

        switch (level) {
            case prediction::RiskLevel::CLEAR:
                // No vibration
                return;

            case prediction::RiskLevel::CAUTION:
                // Single pulse, no cooldown (informational)
                req.pattern = VibratorPattern::SinglePulse;
                break;

            case prediction::RiskLevel::WARNING:
                // Double pulse with cooldown
                if (now - last_warning_time_ < WARNING_COOLDOWN) {
                    return;  // Still in cooldown
                }
                last_warning_time_ = now;
                req.pattern = VibratorPattern::DoublePulse;
                break;

            case prediction::RiskLevel::DANGER:
                // Continuous vibration with minimal cooldown
                if (now - last_danger_time_ < DANGER_COOLDOWN) {
                    return;  // Still in cooldown
                }
                last_danger_time_ = now;
                req.pattern = VibratorPattern::Continuous;
                break;

            default:
                return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pattern_queue_.push(req);
    }
}

// ─── Pattern Queueing ────────────────────────────────────────────────────────

void VibrationController::enqueue_pattern(VibratorPattern pattern,
                                           std::chrono::milliseconds duration_ms)
{
    VibrationRequest req;
    req.pattern = pattern;
    req.deadline = std::chrono::steady_clock::now() + duration_ms;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pattern_queue_.push(req);
    }
}

// ─── Emergency Stop ──────────────────────────────────────────────────────────

void VibrationController::stop_immediately() {
    if (pwm_) {
        pwm_->set_duty(0);
    } else if (relay_) {
        relay_->write(false);
    }
}

// ─── Status ──────────────────────────────────────────────────────────────────

std::string VibrationController::error_message() const {
    return error_;
}

// ─── Pattern Application ─────────────────────────────────────────────────────

void VibrationController::apply_pattern(VibratorPattern pattern) {
    switch (pattern) {
        case VibratorPattern::Off:
            stop_immediately();
            break;

        case VibratorPattern::SinglePulse:
            pulse(std::chrono::milliseconds(200), 200);
            break;

        case VibratorPattern::DoublePulse:
            pulse(std::chrono::milliseconds(150), 200);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            pulse(std::chrono::milliseconds(150), 200);
            break;

        case VibratorPattern::Continuous:
            if (pwm_) {
                pwm_->set_percentage(80);
            } else if (relay_) {
                relay_->write(true);
            }
            break;
    }
}

void VibrationController::pulse(std::chrono::milliseconds duration_ms,
                                  uint32_t intensity)
{
    if (pwm_) {
        pwm_->set_duty(intensity);
    } else if (relay_) {
        relay_->write(true);
    }

    std::this_thread::sleep_for(duration_ms);

    stop_immediately();
}

// ─── Worker Thread ───────────────────────────────────────────────────────────

void VibrationController::worker_loop() {
    while (!stop_requested_) {
        VibrationRequest req;
        bool have_request = false;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!pattern_queue_.empty()) {
                req = pattern_queue_.front();
                pattern_queue_.pop();
                have_request = true;
            }
        }

        if (have_request) {
            // Check if deadline has passed (stale request).
            auto now = std::chrono::steady_clock::now();
            if (now < req.deadline) {
                apply_pattern(req.pattern);
            }
        } else {
            // No request; sleep briefly to avoid busy-wait.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

} // namespace audio
