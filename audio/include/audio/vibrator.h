#pragma once

// ─── vibrator.h ──────────────────────────────────────────────────────────────
// Vibrator / haptic feedback controller via GPIO PWM or relay.
//
// Provides vibration patterns triggered by risk level events:
//   - WARNING: Short double-pulse to alert without alarming
//   - DANGER:  Continuous vibration to demand immediate attention
//
// Can control:
//   - Simple relay output (GPIO HIGH/LOW): on/off vibrator motor
//   - PWM output: variable intensity vibration
//
// Thread safety
// ─────────────
// Vibration requests are enqueued and processed in a background worker thread
// to avoid blocking the main perception/prediction loop.

#include "sensors/gpio.h"
#include "prediction/prediction.h"

#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <chrono>
#include <cstdint>
#include <string>

namespace audio {

// ─── VibratorPattern ─────────────────────────────────────────────────────────
//
// Predefined vibration patterns triggered by risk levels.

enum class VibratorPattern {
    Off,          // Stop vibration
    SinglePulse,  // One short pulse (~200 ms)
    DoublePulse,  // Two pulses for WARNING level
    Continuous,   // Steady vibration for DANGER level
};

// ─── VibrationRequest ────────────────────────────────────────────────────────
//
// Internal queue item for the worker thread.

struct VibrationRequest {
    VibratorPattern pattern;
    std::chrono::steady_clock::time_point deadline;
};

// ─── VibrationController ─────────────────────────────────────────────────────
//
// Manages vibrator motor output in response to risk level changes.
//
// Typical usage:
//   VibrationController vibrator(13);  // GPIO pin
//   vibrator.open();
//   vibrator.start();
//   
//   // Periodically call with the current risk level:
//   vibrator.handle_risk_level(prediction::RiskLevel::WARNING);
//   
//   vibrator.stop();
//   vibrator.close();

class VibrationController {
public:
    // Construct with GPIO pin (BCM numbering).
    // pwm_capable: if true, uses PWM for variable intensity; else simple relay.
    explicit VibrationController(uint32_t gpio_pin, bool pwm_capable = false);
    ~VibrationController();

    // Non-copyable.
    VibrationController(const VibrationController&)            = delete;
    VibrationController& operator=(const VibrationController&) = delete;

    // Lifecycle.
    bool open();  // Initialize GPIO
    bool start(); // Begin worker thread
    void stop();  // Stop worker thread gracefully
    void close(); // Release GPIO

    // Handle risk level change — enqueues appropriate vibration pattern.
    // Called from main perception/prediction loop.
    void handle_risk_level(prediction::RiskLevel level);

    // Directly enqueue a vibration request (advanced use).
    void enqueue_pattern(VibratorPattern pattern,
                         std::chrono::milliseconds duration_ms = 
                         std::chrono::milliseconds(0));

    // Immediately stop vibration (emergency).
    void stop_immediately();

    // Status.
    bool        is_open()        const { return is_open_; }
    bool        is_running()      const { return is_running_; }
    uint32_t    gpio_pin()        const { return gpio_pin_; }
    std::string error_message()   const;

private:
    void worker_loop();
    void apply_pattern(VibratorPattern pattern);
    void pulse(std::chrono::milliseconds duration_ms, uint32_t intensity = 255);

    uint32_t gpio_pin_;
    bool     pwm_capable_;
    bool     is_open_;
    std::atomic<bool> is_running_;

    std::unique_ptr<sensors::DigitalOutput> relay_;
    std::unique_ptr<sensors::PWMOutput>     pwm_;

    std::thread                       worker_;
    std::mutex                        queue_mutex_;
    std::queue<VibrationRequest>      pattern_queue_;
    std::atomic<bool>                 stop_requested_;

    mutable std::string               error_;

    // Cooldowns per risk level to avoid vibration spam
    std::mutex                                         cooldown_mutex_;
    std::chrono::steady_clock::time_point              last_warning_time_;
    std::chrono::steady_clock::time_point              last_danger_time_;
    static constexpr std::chrono::milliseconds WARNING_COOLDOWN{ 500 };
    static constexpr std::chrono::milliseconds DANGER_COOLDOWN{ 200 };
};

} // namespace audio
