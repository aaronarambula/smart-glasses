#pragma once

// ─── tfluna.h ─────────────────────────────────────────────────────────────────
// Driver for the Benewake TF-Luna (V1.3) single-point ToF LiDAR.
//
// The TF-Luna is a fixed, forward-facing Time-of-Flight sensor — it measures
// one distance at a time, not a 360° sweep.  It is wired into the LidarBase
// interface by emitting a ScanFrame containing exactly one ScanPoint at
// angle_deg = 0.0° (forward) for each 9-byte measurement packet.
//
// Protocol (UART, 115200 baud 8N1, from TF-Luna product manual V1.3):
//
//   The sensor streams 9-byte frames continuously at 100 Hz (default):
//
//     byte 0   : 0x59  (header 1)
//     byte 1   : 0x59  (header 2)
//     byte 2   : Dist_L  — distance low byte  (cm, little-endian)
//     byte 3   : Dist_H  — distance high byte (cm, little-endian)
//     byte 4   : Amp_L   — signal strength low byte
//     byte 5   : Amp_H   — signal strength high byte
//     byte 6   : Temp_L  — chip temperature low byte (units: 0.01 °C)
//     byte 7   : Temp_H  — chip temperature high byte
//     byte 8   : Checksum = (sum of bytes 0-7) & 0xFF
//
//   distance_mm = dist_cm * 10
//   Valid range : 20 cm – 800 cm (20 mm – 8 000 mm).
//                 dist == 0  → too close / no target
//                 dist > 800 → out of range
//   Amp (signal strength): < 100 → measurement unreliable (quality = 0)
//
// Commands used by this driver:
//   Enable output  : 0x5A 0x05 0x07 0x01 0x67
//   Disable output : 0x5A 0x05 0x07 0x00 0x66
//
// Lifecycle:
//   TFLuna lidar("/dev/ttyAMA0");
//   lidar.open();    // open port, flush, send enable-output command
//   lidar.start();   // launch background read thread
//   // poll lidar.get_latest_frame() or use set_frame_callback()
//   lidar.stop();    // send disable-output, join thread
//   lidar.close();   // close serial port

#include "lidar_base.h"
#include "serial_port.h"

#include <cstdint>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

namespace sensors {

class TFLuna : public LidarBase {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // port_path : serial device, e.g. "/dev/ttyAMA0" or "/dev/ttyUSB0"
    // The TF-Luna V1.3 always runs at 115200 baud.
    explicit TFLuna(std::string port_path);

    // Destructor: stops streaming and closes the port if still open.
    ~TFLuna() override;

    // ── LidarBase interface ───────────────────────────────────────────────────

    // Opens the serial port, flushes stale data, and sends the enable-output
    // command.  Returns false if the port cannot be opened.
    bool open() override;

    // Starts the background read thread.  open() must have succeeded first.
    bool start() override;

    // Sends the disable-output command and joins the background thread.
    void stop() override;

    // Closes the serial port.  Calls stop() first if still running.
    void close() override;

    // Returns a copy of the most recently parsed ScanFrame (one point at 0°).
    // Thread-safe.
    ScanFrame get_latest_frame() const override;

    // Register / clear the per-frame callback.
    // Invoked from the background thread — keep it short.
    void set_frame_callback(FrameCallback cb) override;

    bool        is_open()      const override;
    bool        is_running()   const override;
    std::string error_message() const override;
    std::string model_name()   const override { return "TF-Luna"; }

    // ── Extended API ──────────────────────────────────────────────────────────

    // Returns the most recent chip temperature in degrees Celsius.
    // Returns 0 if no frame has been received yet.
    float last_temperature_c() const;

private:
    // ── Serial port ───────────────────────────────────────────────────────────

    SerialPort serial_;

    // ── Background thread ─────────────────────────────────────────────────────

    std::thread       read_thread_;
    std::atomic<bool> running_{ false };

    // ── Latest frame (guarded by frame_mutex_) ────────────────────────────────

    mutable std::mutex frame_mutex_;
    ScanFrame          latest_frame_;
    uint64_t           frame_counter_{ 0 };

    // ── Frame callback ────────────────────────────────────────────────────────

    mutable std::mutex frame_cb_mutex_;
    FrameCallback      frame_callback_;

    // ── Temperature ───────────────────────────────────────────────────────────

    mutable std::mutex temp_mutex_;
    float              temperature_c_{ 0.0f };

    // ── Error state ───────────────────────────────────────────────────────────

    mutable std::mutex error_mutex_;
    std::string        error_msg_;

    // ── Protocol constants ────────────────────────────────────────────────────

    static constexpr uint8_t HEADER         = 0x59;
    static constexpr size_t  FRAME_LEN      = 9;
    static constexpr uint16_t AMP_MIN_VALID = 100;   // below this → unreliable
    static constexpr uint16_t DIST_MAX_CM   = 800;   // 8 m max range

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Sends a raw command packet.
    bool send_command(const uint8_t* cmd, size_t len);

    // Parses one 9-byte frame.  Returns false if checksum fails.
    bool parse_frame(const uint8_t* raw,
                     uint16_t& dist_cm, uint16_t& amp, float& temp_c);

    // Background read loop.
    void read_loop();

    // Stores and publishes a completed frame.
    void publish_frame(ScanFrame frame);

    // Thread-safe error setter.
    void set_error(const std::string& msg);
};

} // namespace sensors
