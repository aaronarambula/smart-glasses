#pragma once

// ─── ultrasonic_hc_sr04.h ────────────────────────────────────────────────────
// HC-SR04 ultrasonic distance sensor via GPIO (trigger + echo pins).
//
// Inherits from LidarBase so it can be used interchangeably with LD06/RPLidar.
// Synthesizes a narrow forward-looking "sector" to fit into the LiDAR interface.
//
// Pinout:
//   Vcc   : +5V (power)
//   GND   : Ground
//   Trig  : GPIO pin (BCM numbering) for trigger pulse (LOW before measurement)
//   Echo  : GPIO pin (BCM numbering) for echo pulse (goes HIGH during measurement)
//
// Measurement principle:
//   1. Set TRIG high for ≥10 µs to trigger measurement
//   2. Sensor sets ECHO high, then low after sound return
//   3. Measure ECHO pulse width: time_us = width / 58 millimetres
//      (sound speed ~343 m/s → 0.343 mm/µs → 1 mm = 5.8 µs)

#include "lidar_base.h"
#include "gpio.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

namespace sensors {

class UltrasonicHCSR04 : public LidarBase {
public:
    // Construct with GPIO pin numbers (BCM numbering).
    // scan_hz: measurement rate in Hz (default 10 Hz for ~100 ms between reads).
    explicit UltrasonicHCSR04(uint32_t trigger_pin, uint32_t echo_pin,
                               float scan_hz = 10.0f);
    ~UltrasonicHCSR04() override;

    // Lifecycle.
    bool open() override;
    bool start() override;
    void stop() override;
    void close() override;

    // Data retrieval.
    ScanFrame get_latest_frame() const override;
    void set_frame_callback(FrameCallback cb) override;

    // Status.
    bool        is_open() const override;
    bool        is_running() const override;
    std::string error_message() const override;
    std::string model_name() const override { return "HC-SR04 Ultrasonic"; }

    // Configuration.
    uint32_t trigger_pin() const { return trigger_pin_; }
    uint32_t echo_pin() const { return echo_pin_; }
    float scan_hz() const { return scan_hz_; }
    float max_distance_mm() const { return max_distance_mm_; }

private:
    void read_loop();
    float read_distance_mm();
    void publish_frame(ScanFrame frame);
    void set_error(const std::string& msg);

    // Create a synthetic ScanFrame with a single forward-looking "beam"
    // at the measured distance.
    ScanFrame make_frame(float distance_mm) const;

    uint32_t trigger_pin_;
    uint32_t echo_pin_;
    float    scan_hz_;
    float    max_distance_mm_ = 4000.0f;  // HC-SR04 max ~4 m

    std::atomic<bool> open_{ false };
    std::atomic<bool> running_{ false };
    std::thread       read_thread_;

    std::unique_ptr<DigitalOutput> trigger_;
    std::unique_ptr<DigitalInput>  echo_;

    mutable std::mutex frame_mutex_;
    ScanFrame          latest_frame_;
    uint64_t           frame_counter_ = 0;

    mutable std::mutex frame_cb_mutex_;
    FrameCallback      frame_callback_;

    mutable std::mutex error_mutex_;
    std::string        error_message_;
};

} // namespace sensors
