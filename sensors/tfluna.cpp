// ─── tfluna.cpp ───────────────────────────────────────────────────────────────
// Driver for the Benewake TF-Luna (V1.3) ToF LiDAR.
// See include/sensors/tfluna.h for protocol details and API docs.

#include "sensors/tfluna.h"

#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <sstream>

namespace sensors {

// ── Command packets ───────────────────────────────────────────────────────────
// Format: { 0x5A, len, cmd_id, [payload...], checksum }
// checksum = sum of all preceding bytes in the packet, truncated to uint8.

// Enable output: 0x5A 0x05 0x07 0x01 0x67
static constexpr uint8_t CMD_ENABLE[]  = { 0x5A, 0x05, 0x07, 0x01, 0x67 };
// Disable output: 0x5A 0x05 0x07 0x00 0x66
static constexpr uint8_t CMD_DISABLE[] = { 0x5A, 0x05, 0x07, 0x00, 0x66 };

// ── Construction / Destruction ────────────────────────────────────────────────

TFLuna::TFLuna(std::string port_path)
    : LidarBase(std::move(port_path))
    , serial_(port_, 115200)
{}

TFLuna::~TFLuna()
{
    stop();
    close();
}

// ── open ──────────────────────────────────────────────────────────────────────

bool TFLuna::open()
{
    if (!serial_.open()) {
        set_error("Failed to open serial port '" + port_ + "': "
                  + serial_.error_message());
        return false;
    }

    // Flush any stale bytes left over from a previous session.
    serial_.flush_input();

    // Tell the sensor to start streaming (in case it was previously disabled).
    if (!send_command(CMD_ENABLE, sizeof(CMD_ENABLE))) {
        set_error("Failed to send enable-output command");
        serial_.close();
        return false;
    }

    // Brief pause so the sensor processes the command before we start reading.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    serial_.flush_input();

    return true;
}

// ── start ─────────────────────────────────────────────────────────────────────

bool TFLuna::start()
{
    if (!serial_.is_open()) {
        set_error("start() called before open()");
        return false;
    }
    if (running_.load()) {
        return true;  // already running
    }

    running_.store(true);
    read_thread_ = std::thread(&TFLuna::read_loop, this);
    return true;
}

// ── stop ──────────────────────────────────────────────────────────────────────

void TFLuna::stop()
{
    if (!running_.load()) return;

    running_.store(false);

    if (serial_.is_open()) {
        send_command(CMD_DISABLE, sizeof(CMD_DISABLE));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        serial_.flush_input();
    }

    if (read_thread_.joinable()) {
        read_thread_.join();
    }
}

// ── close ─────────────────────────────────────────────────────────────────────

void TFLuna::close()
{
    stop();
    serial_.close();
}

// ── get_latest_frame ──────────────────────────────────────────────────────────

ScanFrame TFLuna::get_latest_frame() const
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

// ── set_frame_callback ────────────────────────────────────────────────────────

void TFLuna::set_frame_callback(FrameCallback cb)
{
    std::lock_guard<std::mutex> lock(frame_cb_mutex_);
    frame_callback_ = std::move(cb);
}

// ── Status ────────────────────────────────────────────────────────────────────

bool TFLuna::is_open() const    { return serial_.is_open(); }
bool TFLuna::is_running() const { return running_.load(); }

std::string TFLuna::error_message() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_msg_;
}

float TFLuna::last_temperature_c() const
{
    std::lock_guard<std::mutex> lock(temp_mutex_);
    return temperature_c_;
}

// ── send_command ──────────────────────────────────────────────────────────────

bool TFLuna::send_command(const uint8_t* cmd, size_t len)
{
    int n = serial_.write(cmd, len);
    return n == static_cast<int>(len);
}

// ── parse_frame ───────────────────────────────────────────────────────────────
//
// Validates the checksum and extracts distance, amplitude, and temperature.
// Returns false if the checksum fails (caller should re-sync).
//
// raw[] layout (9 bytes):
//   [0] 0x59  header 1
//   [1] 0x59  header 2
//   [2] Dist_L
//   [3] Dist_H
//   [4] Amp_L
//   [5] Amp_H
//   [6] Temp_L  (units: 0.01 °C)
//   [7] Temp_H
//   [8] Checksum = (raw[0]+…+raw[7]) & 0xFF

bool TFLuna::parse_frame(const uint8_t* raw,
                          uint16_t& dist_cm, uint16_t& amp, float& temp_c)
{
    // Verify checksum.
    uint8_t sum = 0;
    for (int i = 0; i < 8; ++i) sum += raw[i];
    if (sum != raw[8]) return false;

    dist_cm = static_cast<uint16_t>(raw[2]) | (static_cast<uint16_t>(raw[3]) << 8);
    amp     = static_cast<uint16_t>(raw[4]) | (static_cast<uint16_t>(raw[5]) << 8);

    const uint16_t raw_temp = static_cast<uint16_t>(raw[6])
                            | (static_cast<uint16_t>(raw[7]) << 8);
    temp_c = static_cast<float>(raw_temp) / 100.0f;

    return true;
}

// ── read_loop ─────────────────────────────────────────────────────────────────
//
// Background thread.  Scans the byte stream for the 0x59 0x59 header pair,
// reads the remaining 7 bytes, validates the checksum, and emits a ScanFrame
// containing one ScanPoint at angle 0° (forward).

void TFLuna::read_loop()
{
    uint8_t buf[FRAME_LEN];

    while (running_.load()) {
        // ── Sync: find 0x59 0x59 header ──────────────────────────────────────
        // Read one byte at a time until we see the first header byte.
        uint8_t b = 0;
        int n = serial_.read(&b, 1, 200);
        if (n <= 0) continue;   // timeout or error — check running_ and retry
        if (b != HEADER) continue;

        // Got 0x59. Check the second header byte.
        n = serial_.read(&b, 1, 100);
        if (n <= 0 || b != HEADER) continue;

        // ── Read remaining 7 bytes ────────────────────────────────────────────
        buf[0] = HEADER;
        buf[1] = HEADER;
        if (!serial_.read_exact(buf + 2, 7, 200)) {
            // Timeout mid-frame — flush and re-sync.
            serial_.flush_input();
            continue;
        }

        // ── Parse ─────────────────────────────────────────────────────────────
        uint16_t dist_cm = 0;
        uint16_t amp     = 0;
        float    temp_c  = 0.0f;
        if (!parse_frame(buf, dist_cm, amp, temp_c)) {
            // Bad checksum — lost sync, flush and start over.
            serial_.flush_input();
            continue;
        }

        // ── Store temperature ─────────────────────────────────────────────────
        {
            std::lock_guard<std::mutex> lk(temp_mutex_);
            temperature_c_ = temp_c;
        }

        // ── Build ScanPoint ───────────────────────────────────────────────────
        ScanPoint pt;
        pt.angle_deg   = 0.0f;   // forward-facing only
        pt.is_new_scan = false;  // not a rotating scanner

        // Distance: invalid if too close, zero, or beyond max range.
        if (dist_cm == 0 || dist_cm > DIST_MAX_CM) {
            pt.distance_mm = 0.0f;  // marks invalid via is_valid()
        } else {
            pt.distance_mm = static_cast<float>(dist_cm) * 10.0f;  // cm → mm
        }

        // Quality: map signal strength to [0, 255].
        // Values below AMP_MIN_VALID are unreliable → quality = 0.
        if (amp < AMP_MIN_VALID) {
            pt.quality = 0;
        } else {
            // Scale: 100 → 1, 25 500 → 255 (clamp above).
            const uint32_t scaled = amp / 100u;
            pt.quality = static_cast<uint8_t>(scaled > 255u ? 255u : scaled);
        }

        // ── Emit frame ────────────────────────────────────────────────────────
        ScanFrame frame;
        frame.points    = { pt };
        frame.timestamp = std::chrono::steady_clock::now();
        frame.sensor_rpm = 0.0f;  // not a rotating sensor

        {
            std::lock_guard<std::mutex> lk(frame_mutex_);
            frame.frame_id = frame_counter_++;
        }

        publish_frame(std::move(frame));
    }
}

// ── publish_frame ─────────────────────────────────────────────────────────────

void TFLuna::publish_frame(ScanFrame frame)
{
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = frame;
    }

    FrameCallback cb;
    {
        std::lock_guard<std::mutex> lk(frame_cb_mutex_);
        cb = frame_callback_;
    }
    if (cb) cb(frame);
}

// ── set_error ─────────────────────────────────────────────────────────────────

void TFLuna::set_error(const std::string& msg)
{
    std::lock_guard<std::mutex> lk(error_mutex_);
    error_msg_ = msg;
}

} // namespace sensors
