#pragma once

// ─── gpio.h ──────────────────────────────────────────────────────────────────
// GPIO abstraction layer for Raspberry Pi hardware (BCM pin numbering).
//
// Provides:
//   - DigitalInput:  Read discrete GPIO pins (buttons, echo signals)
//   - DigitalOutput: Control discrete GPIO pins (vibrator relay, trigger pulses)
//   - PWMOutput:     Control PWM-capable pins (vibrator intensity/duration)
//
// On non-Pi systems (dev machines), the interface is stubbed to always succeed
// but do nothing (GPIO writes silently ignored, reads return default values).
//
// Thread safety
// ─────────────
// All GPIO operations are thread-safe as they use pigpio library (pigpiod daemon)
// or sysfs (which serializes at kernel level).  However, typically all GPIO
// operations happen on the main thread (not concurrent with sensor reads).

#include <string>
#include <cstdint>
#include <memory>

namespace sensors {

// ─── DigitalInput ────────────────────────────────────────────────────────────
//
// Read-only GPIO input (pulled high/low by hardware).
// Typical uses: button pins, echo timing pins.
//
//   bcm_pin : Raspberry Pi GPIO number (BCM numbering), e.g. 12, 15, 39
//   active_high : if true, HIGH=pressed; if false, LOW=pressed (default: true)

class DigitalInput {
public:
    explicit DigitalInput(uint32_t bcm_pin, bool active_high = true);
    ~DigitalInput();

    // Non-copyable.
    DigitalInput(const DigitalInput&)            = delete;
    DigitalInput& operator=(const DigitalInput&) = delete;

    // Movable.
    DigitalInput(DigitalInput&&) noexcept;
    DigitalInput& operator=(DigitalInput&&) noexcept;

    // Initialize the pin for input (configure pull resistor if needed).
    // Returns true on success, false on error (check error_message()).
    bool open();

    // Release the pin and return it to default state.
    void close();

    // Read the current state: true = HIGH/pressed, false = LOW/released.
    // Only valid after open() succeeds.
    bool read() const;

    // Status.
    bool        is_open()        const { return is_open_; }
    uint32_t    bcm_pin()        const { return bcm_pin_; }
    std::string error_message()  const { return error_; }

private:
    uint32_t bcm_pin_;
    bool     active_high_;
    bool     is_open_;
    mutable std::string error_;

    void set_error(const std::string& msg) { error_ = msg; }
};

// ─── DigitalOutput ───────────────────────────────────────────────────────────
//
// Write-only GPIO output (drives a relay, LED, or trigger pin).
// Typical uses: vibrator relay, ultrasonic trigger pulse.

class DigitalOutput {
public:
    explicit DigitalOutput(uint32_t bcm_pin, bool initial_state = false);
    ~DigitalOutput();

    // Non-copyable.
    DigitalOutput(const DigitalOutput&)            = delete;
    DigitalOutput& operator=(const DigitalOutput&) = delete;

    // Movable.
    DigitalOutput(DigitalOutput&&) noexcept;
    DigitalOutput& operator=(DigitalOutput&&) noexcept;

    // Initialize the pin for output.
    // initial_state: true = HIGH, false = LOW.
    // Returns true on success, false on error.
    bool open();

    // Release the pin.
    void close();

    // Write the pin: true = HIGH, false = LOW.
    // Only valid after open() succeeds.
    void write(bool state);

    // Get the last written state (may differ from hardware if overridden).
    bool read_state() const { return state_; }

    // Status.
    bool        is_open()        const { return is_open_; }
    uint32_t    bcm_pin()        const { return bcm_pin_; }
    std::string error_message()  const { return error_; }

private:
    uint32_t bcm_pin_;
    bool     state_;
    bool     is_open_;
    mutable std::string error_;

    void set_error(const std::string& msg) { error_ = msg; }
};

// ─── PWMOutput ───────────────────────────────────────────────────────────────
//
// PWM output for variable intensity/frequency control.
// Typical uses: vibrator motor speed/intensity, LED brightness.
//
// Supported only on pins with hardware PWM support (BCM 12, 13, 18, 19, etc.).
// Unsupported pins on non-Pi systems silently succeed but produce no output.
//
//   bcm_pin     : GPIO pin number (BCM numbering)
//   frequency   : PWM frequency in Hz (default 1000 Hz)
//   range       : max duty cycle value (default 255, so 0–255 for 0–100%)

class PWMOutput {
public:
    explicit PWMOutput(uint32_t bcm_pin, uint32_t frequency = 1000, 
                       uint32_t range = 255);
    ~PWMOutput();

    // Non-copyable.
    PWMOutput(const PWMOutput&)            = delete;
    PWMOutput& operator=(const PWMOutput&) = delete;

    // Movable.
    PWMOutput(PWMOutput&&) noexcept;
    PWMOutput& operator=(PWMOutput&&) noexcept;

    // Initialize the pin for PWM output.
    // Returns true on success, false on error.
    bool open();

    // Release the pin and stop PWM.
    void close();

    // Set PWM duty cycle. duty must be in [0, range_].
    // duty == 0: 0% (fully off)
    // duty == range_: 100% (fully on)
    void set_duty(uint32_t duty);

    // Convenience: set duty as a percentage [0, 100].
    void set_percentage(uint32_t percent) {
        if (percent > 100) percent = 100;
        set_duty((percent * range_) / 100);
    }

    // Get the last set duty cycle.
    uint32_t duty() const { return duty_; }

    // Status.
    bool        is_open()        const { return is_open_; }
    uint32_t    bcm_pin()        const { return bcm_pin_; }
    uint32_t    frequency()      const { return frequency_; }
    uint32_t    range()          const { return range_; }
    std::string error_message()  const { return error_; }

private:
    uint32_t bcm_pin_;
    uint32_t frequency_;
    uint32_t range_;
    uint32_t duty_;
    bool     is_open_;
    mutable std::string error_;

    void set_error(const std::string& msg) { error_ = msg; }
};

} // namespace sensors
