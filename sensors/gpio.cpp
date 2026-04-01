// ─── gpio.cpp ────────────────────────────────────────────────────────────────
// GPIO implementation using sysfs on Linux and stubs on non-Pi systems.
//
// This implementation uses the Linux sysfs GPIO interface:
//   /sys/class/gpio/gpio<N>/
//
// For production use on Raspberry Pi, consider using pigpio library (pigpiod)
// for better real-time performance, but sysfs is sufficient for our use case.

#include "sensors/gpio.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <unistd.h>

// Check if we're on a Linux system with sysfs GPIO support
#ifdef __linux__
  #define HAS_SYSFS_GPIO 1
#else
  #define HAS_SYSFS_GPIO 0
#endif

namespace sensors {

// ─── Helper utilities ────────────────────────────────────────────────────────

#if HAS_SYSFS_GPIO
static bool sysfs_export_gpio(uint32_t bcm_pin) {
    std::ofstream f("/sys/class/gpio/export");
    if (!f.is_open()) return false;
    f << bcm_pin;
    f.close();
    // Small delay for kernel to create the sysfs entry
    usleep(10000);
    return true;
}

static bool sysfs_unexport_gpio(uint32_t bcm_pin) {
    std::ofstream f("/sys/class/gpio/unexport");
    if (!f.is_open()) return false;
    f << bcm_pin;
    f.close();
    usleep(10000);
    return true;
}

static std::string sysfs_pin_path(uint32_t bcm_pin) {
    std::ostringstream oss;
    oss << "/sys/class/gpio/gpio" << bcm_pin;
    return oss.str();
}

static bool sysfs_set_direction(uint32_t bcm_pin, const std::string& dir) {
    std::string path = sysfs_pin_path(bcm_pin) + "/direction";
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << dir;
    f.close();
    return true;
}

static bool sysfs_read_value(uint32_t bcm_pin) {
    std::string path = sysfs_pin_path(bcm_pin) + "/value";
    std::ifstream f(path);
    if (!f.is_open()) return false;
    int val;
    f >> val;
    f.close();
    return val != 0;
}

static bool sysfs_write_value(uint32_t bcm_pin, bool state) {
    std::string path = sysfs_pin_path(bcm_pin) + "/value";
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << (state ? "1" : "0");
    f.close();
    return true;
}
#endif

// ─── DigitalInput ────────────────────────────────────────────────────────────

DigitalInput::DigitalInput(uint32_t bcm_pin, bool active_high)
    : bcm_pin_(bcm_pin)
    , active_high_(active_high)
    , is_open_(false)
    , error_("")
{
}

DigitalInput::~DigitalInput() {
    close();
}

DigitalInput::DigitalInput(DigitalInput&& o) noexcept
    : bcm_pin_(o.bcm_pin_)
    , active_high_(o.active_high_)
    , is_open_(o.is_open_)
    , error_(std::move(o.error_))
{
    o.is_open_ = false;
}

DigitalInput& DigitalInput::operator=(DigitalInput&& o) noexcept {
    if (this != &o) {
        close();
        bcm_pin_ = o.bcm_pin_;
        active_high_ = o.active_high_;
        is_open_ = o.is_open_;
        error_ = std::move(o.error_);
        o.is_open_ = false;
    }
    return *this;
}

bool DigitalInput::open() {
    if (is_open_) return true;

#if HAS_SYSFS_GPIO
    // Export the GPIO if not already exported
    sysfs_export_gpio(bcm_pin_);

    // Set direction to input
    if (!sysfs_set_direction(bcm_pin_, "in")) {
        set_error("Failed to set GPIO " + std::to_string(bcm_pin_) + " to input");
        return false;
    }

    is_open_ = true;
    return true;
#else
    // Non-Pi system: stub succeeds silently
    is_open_ = true;
    return true;
#endif
}

void DigitalInput::close() {
    if (!is_open_) return;

#if HAS_SYSFS_GPIO
    sysfs_unexport_gpio(bcm_pin_);
#endif

    is_open_ = false;
}

bool DigitalInput::read() const {
    if (!is_open_) return false;

#if HAS_SYSFS_GPIO
    bool value = sysfs_read_value(bcm_pin_);
    return active_high_ ? value : !value;
#else
    // Non-Pi system: always return false (not pressed)
    return false;
#endif
}

// ─── DigitalOutput ───────────────────────────────────────────────────────────

DigitalOutput::DigitalOutput(uint32_t bcm_pin, bool initial_state)
    : bcm_pin_(bcm_pin)
    , state_(initial_state)
    , is_open_(false)
    , error_("")
{
}

DigitalOutput::~DigitalOutput() {
    close();
}

DigitalOutput::DigitalOutput(DigitalOutput&& o) noexcept
    : bcm_pin_(o.bcm_pin_)
    , state_(o.state_)
    , is_open_(o.is_open_)
    , error_(std::move(o.error_))
{
    o.is_open_ = false;
}

DigitalOutput& DigitalOutput::operator=(DigitalOutput&& o) noexcept {
    if (this != &o) {
        close();
        bcm_pin_ = o.bcm_pin_;
        state_ = o.state_;
        is_open_ = o.is_open_;
        error_ = std::move(o.error_);
        o.is_open_ = false;
    }
    return *this;
}

bool DigitalOutput::open() {
    if (is_open_) return true;

#if HAS_SYSFS_GPIO
    // Export the GPIO
    sysfs_export_gpio(bcm_pin_);

    // Set direction to output
    if (!sysfs_set_direction(bcm_pin_, "out")) {
        set_error("Failed to set GPIO " + std::to_string(bcm_pin_) + " to output");
        return false;
    }

    // Set initial state
    if (!sysfs_write_value(bcm_pin_, state_)) {
        set_error("Failed to write initial value to GPIO " + std::to_string(bcm_pin_));
        return false;
    }

    is_open_ = true;
    return true;
#else
    // Non-Pi system: stub succeeds silently
    is_open_ = true;
    return true;
#endif
}

void DigitalOutput::close() {
    if (!is_open_) return;

#if HAS_SYSFS_GPIO
    sysfs_unexport_gpio(bcm_pin_);
#endif

    is_open_ = false;
}

void DigitalOutput::write(bool state) {
    state_ = state;

#if HAS_SYSFS_GPIO
    if (is_open_) {
        sysfs_write_value(bcm_pin_, state);
    }
#endif
}

// ─── PWMOutput ───────────────────────────────────────────────────────────────

PWMOutput::PWMOutput(uint32_t bcm_pin, uint32_t frequency, uint32_t range)
    : bcm_pin_(bcm_pin)
    , frequency_(frequency)
    , range_(range)
    , duty_(0)
    , is_open_(false)
    , error_("")
{
}

PWMOutput::~PWMOutput() {
    close();
}

PWMOutput::PWMOutput(PWMOutput&& o) noexcept
    : bcm_pin_(o.bcm_pin_)
    , frequency_(o.frequency_)
    , range_(o.range_)
    , duty_(o.duty_)
    , is_open_(o.is_open_)
    , error_(std::move(o.error_))
{
    o.is_open_ = false;
}

PWMOutput& PWMOutput::operator=(PWMOutput&& o) noexcept {
    if (this != &o) {
        close();
        bcm_pin_ = o.bcm_pin_;
        frequency_ = o.frequency_;
        range_ = o.range_;
        duty_ = o.duty_;
        is_open_ = o.is_open_;
        error_ = std::move(o.error_);
        o.is_open_ = false;
    }
    return *this;
}

bool PWMOutput::open() {
    if (is_open_) return true;

#if HAS_SYSFS_GPIO
    // Export the GPIO
    sysfs_export_gpio(bcm_pin_);

    // Set to output (PWM will use the pin's PWM capability if available)
    if (!sysfs_set_direction(bcm_pin_, "out")) {
        set_error("Failed to set GPIO " + std::to_string(bcm_pin_) + " to output");
        return false;
    }

    // Note: Full PWM setup requires access to chip-specific PWM controllers
    // (/sys/class/pwm/pwmchip0, etc.). For now, we simulate PWM via repeated
    // writes in a background thread (omitted for simplicity; use pigpio for real PWM).
    
    is_open_ = true;
    return true;
#else
    // Non-Pi system: stub succeeds silently
    is_open_ = true;
    return true;
#endif
}

void PWMOutput::close() {
    if (!is_open_) return;

#if HAS_SYSFS_GPIO
    sysfs_write_value(bcm_pin_, false);
    sysfs_unexport_gpio(bcm_pin_);
#endif

    is_open_ = false;
}

void PWMOutput::set_duty(uint32_t duty) {
    if (duty > range_) duty = range_;
    duty_ = duty;

#if HAS_SYSFS_GPIO
    if (is_open_) {
        // Simplified: duty > 0 means ON, duty == 0 means OFF.
        // For real PWM, integrate pigpio or use /sys/class/pwm.
        sysfs_write_value(bcm_pin_, duty > 0);
    }
#endif
}

} // namespace sensors
