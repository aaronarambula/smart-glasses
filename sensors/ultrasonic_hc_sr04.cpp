// ─── ultrasonic_hc_sr04.cpp ──────────────────────────────────────────────────
// HC-SR04 ultrasonic sensor implementation using GPIO (trigger/echo pins).
//
// Measurement timing:
//   - Trigger pulse: HIGH for 10 µs minimum
//   - Echo pulse: HIGH for a duration proportional to distance
//   - Conversion: distance_mm ≈ echo_time_us * 0.343 / 2
//                            ≈ echo_time_us / 5.8
//
// Read loop:
//   1. Set TRIG high
//   2. Sleep 10 µs
//   3. Set TRIG low
//   4. Wait for ECHO to go high (timeout 100 µs for safety)
//   5. Measure ECHO pulse width using busy-wait (no interrupts in this impl)
//   6. Convert to distance
//   7. Publish frame
//   8. Sleep until next measurement interval (1000 ms / scan_hz)

#include "sensors/ultrasonic_hc_sr04.h"

#include <chrono>
#include <thread>
#include <cmath>
#include <sstream>

namespace sensors {

// ─── Construction / Destruction ──────────────────────────────────────────────

UltrasonicHCSR04::UltrasonicHCSR04(uint32_t trigger_pin, uint32_t echo_pin,
                                   float scan_hz)
    : LidarBase("HC-SR04 ultrasonic")
    , trigger_pin_(trigger_pin)
    , echo_pin_(echo_pin)
    , scan_hz_(scan_hz > 0.0f ? scan_hz : 10.0f)
{
    // Initialize GPIO objects (but don't open yet).
    trigger_ = std::make_unique<DigitalOutput>(trigger_pin, false);
    echo_ = std::make_unique<DigitalInput>(echo_pin, true);
}

UltrasonicHCSR04::~UltrasonicHCSR04() {
    stop();
    close();
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

bool UltrasonicHCSR04::open() {
    if (open_) return true;

    // Open GPIO pins.
    if (!trigger_ || !trigger_->open()) {
        set_error("Failed to open trigger pin GPIO " + std::to_string(trigger_pin_));
        return false;
    }

    if (!echo_ || !echo_->open()) {
        set_error("Failed to open echo pin GPIO " + std::to_string(echo_pin_));
        trigger_->close();
        return false;
    }

    open_ = true;
    return true;
}

bool UltrasonicHCSR04::start() {
    if (!open_) {
        set_error("Cannot start; sensor not open");
        return false;
    }

    if (running_) return true;

    running_ = true;
    read_thread_ = std::thread([this] { read_loop(); });
    return true;
}

void UltrasonicHCSR04::stop() {
    if (!running_) return;

    running_ = false;
    if (read_thread_.joinable()) {
        read_thread_.join();
    }
}

void UltrasonicHCSR04::close() {
    stop();

    if (trigger_) trigger_->close();
    if (echo_) echo_->close();

    open_ = false;
}

// ─── Data Retrieval ──────────────────────────────────────────────────────────

ScanFrame UltrasonicHCSR04::get_latest_frame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

void UltrasonicHCSR04::set_frame_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lock(frame_cb_mutex_);
    frame_callback_ = cb;
}

// ─── Status ──────────────────────────────────────────────────────────────────

bool UltrasonicHCSR04::is_open() const {
    return open_;
}

bool UltrasonicHCSR04::is_running() const {
    return running_;
}

std::string UltrasonicHCSR04::error_message() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
}

// ─── Frame Creation ──────────────────────────────────────────────────────────

ScanFrame UltrasonicHCSR04::make_frame(float distance_mm) const {
    ScanFrame frame;
    frame.frame_id = frame_counter_;
    frame.timestamp = std::chrono::steady_clock::now();
    frame.sensor_rpm = 0.0f;  // N/A for ultrasonic

    // Create a single "forward-looking" point (0° = forward).
    // Quality: 255 if valid distance, 0 if out of range.
    ScanPoint point;
    point.angle_deg = 0.0f;
    point.distance_mm = distance_mm;
    point.quality = (distance_mm > 0 && distance_mm <= max_distance_mm_) ? 255 : 0;
    point.is_new_scan = true;

    frame.points.push_back(point);
    return frame;
}

// ─── Distance Measurement ────────────────────────────────────────────────────

float UltrasonicHCSR04::read_distance_mm() {
    // Pulse sequence:
    // 1. Ensure TRIG is LOW before starting
    trigger_->write(false);
    std::this_thread::sleep_for(std::chrono::microseconds(5));

    // 2. Set TRIG HIGH for ≥10 µs
    trigger_->write(true);
    std::this_thread::sleep_for(std::chrono::microseconds(10));

    // 3. Set TRIG LOW
    trigger_->write(false);

    // 4. Wait for ECHO to go HIGH (timeout 100 µs for safety)
    auto echo_start_time = std::chrono::steady_clock::now();
    while (!echo_->read()) {
        auto elapsed = std::chrono::steady_clock::now() - echo_start_time;
        if (elapsed > std::chrono::microseconds(100)) {
            set_error("Echo pin timeout waiting for HIGH");
            return 0.0f;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    // 5. Measure ECHO pulse width using busy-wait
    // (For production, use interrupts or libgpiod events for better accuracy.)
    auto pulse_start_time = std::chrono::steady_clock::now();
    while (echo_->read()) {
        auto elapsed = std::chrono::steady_clock::now() - pulse_start_time;
        if (elapsed > std::chrono::milliseconds(100)) {
            // Sanity check: max echo time is ~58 ms for 10 m (max range)
            set_error("Echo pulse timeout");
            return 0.0f;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    auto pulse_end_time = std::chrono::steady_clock::now();

    // 6. Convert pulse width to distance
    // distance_mm = pulse_time_us * sound_speed_mm_per_us / 2
    //            = pulse_time_us * 0.343 / 2  (343 m/s at 20°C)
    //            ≈ pulse_time_us / 5.8
    auto pulse_us = std::chrono::duration_cast<std::chrono::microseconds>(
        pulse_end_time - pulse_start_time).count();
    float distance_mm = static_cast<float>(pulse_us) / 5.8f;

    // Clamp to valid range.
    if (distance_mm < 0) distance_mm = 0.0f;
    if (distance_mm > max_distance_mm_) distance_mm = 0.0f;  // Invalid

    return distance_mm;
}

// ─── Read Loop ───────────────────────────────────────────────────────────────

void UltrasonicHCSR04::read_loop() {
    const auto interval = std::chrono::milliseconds(
        static_cast<int>(1000.0f / scan_hz_));

    while (running_) {
        auto start = std::chrono::steady_clock::now();

        // Measure distance.
        float distance_mm = read_distance_mm();

        // Create and publish frame.
        {
            ScanFrame frame = make_frame(distance_mm);
            frame_counter_++;

            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                latest_frame_ = frame;
            }

            // Invoke callback if registered.
            {
                std::lock_guard<std::mutex> lock(frame_cb_mutex_);
                if (frame_callback_) {
                    frame_callback_(frame);
                }
            }
        }

        // Sleep to maintain scan rate.
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < interval) {
            std::this_thread::sleep_for(interval - elapsed);
        }
    }
}

// ─── Error Handling ──────────────────────────────────────────────────────────

void UltrasonicHCSR04::set_error(const std::string& msg) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_message_ = msg;
}

} // namespace sensors
