// ─── rplidar_a1.cpp ──────────────────────────────────────────────────────────
// Full implementation of the RPLIDAR A1 driver.
// See include/sensors/rplidar_a1.h for the protocol overview and API docs.

#include "sensors/rplidar_a1.h"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace sensors {

// ─── Construction / Destruction ──────────────────────────────────────────────

RPLidarA1::RPLidarA1(std::string port_path)
    : LidarBase(std::move(port_path))
    , serial_(port_, 115200)
{}

RPLidarA1::~RPLidarA1()
{
    stop();
    close();
}

// ─── open ────────────────────────────────────────────────────────────────────
//
// 1. Open serial port at 115200 baud
// 2. Send STOP  (clears any in-progress scan from a previous session)
// 3. Send RESET (soft-reboot the sensor)
// 4. Wait for boot (RESET_WAIT_MS)
// 5. Flush the RX buffer (boot message is discarded)
// 6. Check health status

bool RPLidarA1::open()
{
    if (!serial_.open()) {
        set_error("Failed to open serial port '" + port_ + "': "
                  + serial_.error_message());
        return false;
    }

    // ── 1. STOP: abort any scan left over from a previous session ─────────────
    // STOP has no response — just fire and forget.
    if (!send_command(CMD_STOP)) {
        set_error("Failed to send STOP command");
        serial_.close();
        return false;
    }

    // Small delay: spec says 1 ms is sufficient after STOP.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    serial_.flush_input();

    // ── 2. RESET: soft-reboot the sensor ─────────────────────────────────────
    // RESET has no response descriptor — the sensor reboots and prints an
    // ASCII boot banner (~500 ms).  We just wait and flush it.
    if (!send_command(CMD_RESET)) {
        set_error("Failed to send RESET command");
        serial_.close();
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(RESET_WAIT_MS));
    serial_.flush_input();

    // ── 3. Check health ───────────────────────────────────────────────────────
    RPLidarHealth health;
    if (!get_health(health)) {
        // get_health already set the error message
        serial_.close();
        return false;
    }

    if (health.status == RPLidarHealthStatus::Error) {
        std::ostringstream ss;
        ss << "RPLIDAR A1 health check failed: ERROR status, error_code=0x"
           << std::hex << std::setw(4) << std::setfill('0') << health.error_code;
        set_error(ss.str());
        serial_.close();
        return false;
    }

    if (health.status == RPLidarHealthStatus::Warning) {
        // Warning is non-fatal — log it but proceed.
        std::ostringstream ss;
        ss << "RPLIDAR A1 health WARNING: error_code=0x"
           << std::hex << std::setw(4) << std::setfill('0') << health.error_code;
        set_error(ss.str());
        // Fall through — we can still scan.
    }

    return true;
}

// ─── start ───────────────────────────────────────────────────────────────────
//
// Sends the SCAN command, reads and validates the response descriptor, then
// launches the background read thread.

bool RPLidarA1::start()
{
    if (!serial_.is_open()) {
        set_error("start() called before open()");
        return false;
    }
    if (running_.load()) {
        return true;  // already running
    }

    // ── Send SCAN command ─────────────────────────────────────────────────────
    if (!send_command(CMD_SCAN)) {
        set_error("Failed to send SCAN command");
        return false;
    }

    // ── Read response descriptor (7 bytes) ────────────────────────────────────
    uint32_t data_len  = 0;
    uint8_t  data_type = 0;
    if (!read_response_descriptor(data_len, data_type)) {
        // error message already set inside read_response_descriptor
        return false;
    }

    // SCAN response: data_type = 0x81, continuous stream of 5-byte nodes.
    if (data_type != 0x81) {
        std::ostringstream ss;
        ss << "SCAN response descriptor: unexpected data_type=0x"
           << std::hex << static_cast<int>(data_type) << " (expected 0x81)";
        set_error(ss.str());
        return false;
    }

    // ── Launch background read thread ─────────────────────────────────────────
    running_.store(true);
    read_thread_ = std::thread(&RPLidarA1::read_loop, this);

    return true;
}

// ─── stop ────────────────────────────────────────────────────────────────────

void RPLidarA1::stop()
{
    if (!running_.load()) return;

    running_.store(false);

    // Send STOP to halt the sensor's scan motor and data stream.
    // If the port is still open, best-effort — ignore errors.
    if (serial_.is_open()) {
        send_command(CMD_STOP);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        serial_.flush_input();
    }

    if (read_thread_.joinable()) {
        read_thread_.join();
    }
}

// ─── close ───────────────────────────────────────────────────────────────────

void RPLidarA1::close()
{
    stop();
    serial_.close();
}

// ─── get_latest_frame ────────────────────────────────────────────────────────

ScanFrame RPLidarA1::get_latest_frame() const
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

// ─── set_frame_callback ───────────────────────────────────────────────────────

void RPLidarA1::set_frame_callback(FrameCallback cb)
{
    std::lock_guard<std::mutex> lock(frame_cb_mutex_);
    frame_callback_ = std::move(cb);
}

// ─── Status accessors ─────────────────────────────────────────────────────────

bool RPLidarA1::is_open() const
{
    return serial_.is_open();
}

bool RPLidarA1::is_running() const
{
    return running_.load();
}

std::string RPLidarA1::error_message() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
}

float RPLidarA1::estimated_rpm() const
{
    std::lock_guard<std::mutex> lock(rpm_mutex_);
    return estimated_rpm_;
}

// ─── get_device_info ─────────────────────────────────────────────────────────
//
// Sends GET_INFO and parses the 20-byte response payload:
//   byte  0      : model
//   bytes 1-2    : firmware (minor, major)
//   byte  3      : hardware
//   bytes 4-19   : serial number (16 bytes)

bool RPLidarA1::get_device_info(RPLidarDeviceInfo& info_out)
{
    if (!send_command(CMD_GET_INFO)) {
        set_error("Failed to send GET_INFO command");
        return false;
    }

    uint32_t data_len  = 0;
    uint8_t  data_type = 0;
    if (!read_response_descriptor(data_len, data_type)) {
        return false;
    }

    if (data_len != INFO_RESPONSE_LEN) {
        std::ostringstream ss;
        ss << "GET_INFO: unexpected data_len=" << data_len
           << " (expected " << INFO_RESPONSE_LEN << ")";
        set_error(ss.str());
        return false;
    }

    uint8_t payload[INFO_RESPONSE_LEN];
    if (!serial_.read_exact(payload, INFO_RESPONSE_LEN)) {
        set_error("GET_INFO: timeout reading payload: "
                  + serial_.error_message());
        return false;
    }

    info_out.model            = payload[0];
    info_out.firmware_minor   = payload[1];
    info_out.firmware_major   = payload[2];
    info_out.hardware         = payload[3];
    std::memcpy(info_out.serial_number, payload + 4, 16);

    return true;
}

// ─── get_health ──────────────────────────────────────────────────────────────
//
// Sends GET_HEALTH and parses the 3-byte response payload:
//   byte 0   : status (0=good, 1=warning, 2=error)
//   bytes 1-2: error code (little-endian uint16)

bool RPLidarA1::get_health(RPLidarHealth& health_out)
{
    if (!send_command(CMD_GET_HEALTH)) {
        set_error("Failed to send GET_HEALTH command");
        return false;
    }

    uint32_t data_len  = 0;
    uint8_t  data_type = 0;
    if (!read_response_descriptor(data_len, data_type)) {
        return false;
    }

    if (data_len != HEALTH_RESPONSE_LEN) {
        std::ostringstream ss;
        ss << "GET_HEALTH: unexpected data_len=" << data_len
           << " (expected " << HEALTH_RESPONSE_LEN << ")";
        set_error(ss.str());
        return false;
    }

    uint8_t payload[HEALTH_RESPONSE_LEN];
    if (!serial_.read_exact(payload, HEALTH_RESPONSE_LEN)) {
        set_error("GET_HEALTH: timeout reading payload: "
                  + serial_.error_message());
        return false;
    }

    switch (payload[0]) {
        case 0x00: health_out.status = RPLidarHealthStatus::Good;    break;
        case 0x01: health_out.status = RPLidarHealthStatus::Warning; break;
        case 0x02: health_out.status = RPLidarHealthStatus::Error;   break;
        default:   health_out.status = RPLidarHealthStatus::Unknown; break;
    }

    health_out.error_code = static_cast<uint16_t>(payload[1])
                          | (static_cast<uint16_t>(payload[2]) << 8);

    return true;
}

// ─── send_command ─────────────────────────────────────────────────────────────
//
// Sends a request packet with no payload:
//   { SYNC_BYTE (0xA5), cmd }
// Commands with payloads (not used in this driver) would also include a
// payload_size byte, the payload bytes, and a checksum.

bool RPLidarA1::send_command(uint8_t cmd)
{
    const uint8_t pkt[2] = { SYNC_BYTE, cmd };
    int n = serial_.write(pkt, sizeof(pkt));
    return n == static_cast<int>(sizeof(pkt));
}

// ─── read_response_descriptor ────────────────────────────────────────────────
//
// Every command response begins with a 7-byte descriptor:
//   [0]   0xA5  (sync 1)
//   [1]   0x5A  (sync 2)
//   [2-5] response data length (uint32 LE) — 0x3FFFFFFF for streaming responses
//   [6]   { send_mode[1:0], data_type[7:2] }
//
// We read until we find the two sync bytes, then read the remaining 5 bytes.

bool RPLidarA1::read_response_descriptor(uint32_t& data_response_len,
                                          uint8_t&  data_type)
{
    // Scan for sync byte pair with a timeout.
    constexpr int SYNC_TIMEOUT_MS = 2000;
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(SYNC_TIMEOUT_MS);

    uint8_t b0 = 0, b1 = 0;

    // Read first sync byte.
    while (std::chrono::steady_clock::now() < deadline) {
        if (serial_.read(&b0, 1, 100) == 1) {
            if (b0 == SYNC_BYTE) break;
        }
    }
    if (b0 != SYNC_BYTE) {
        set_error("read_response_descriptor: timeout waiting for sync byte 0xA5");
        return false;
    }

    // Read second sync byte.
    while (std::chrono::steady_clock::now() < deadline) {
        if (serial_.read(&b1, 1, 100) == 1) {
            if (b1 == SYNC_BYTE2) break;
            // Got a byte but not 0x5A — could be the first byte of a pair,
            // check if it's 0xA5.
            if (b1 == SYNC_BYTE) {
                // Keep b1 as the new b0 candidate and try again.
                b0 = b1;
                continue;
            }
        }
    }
    if (b1 != SYNC_BYTE2) {
        set_error("read_response_descriptor: timeout waiting for sync byte 0x5A");
        return false;
    }

    // Read remaining 5 bytes of the descriptor.
    uint8_t rest[5];
    if (!serial_.read_exact(rest, 5, 1000)) {
        set_error("read_response_descriptor: timeout reading descriptor body");
        return false;
    }

    // Bytes 2-5: data length (little-endian uint32).
    // Bit-mask the top 2 bits which are the send_mode field.
    data_response_len = static_cast<uint32_t>(rest[0])
                      | (static_cast<uint32_t>(rest[1]) << 8)
                      | (static_cast<uint32_t>(rest[2]) << 16)
                      | (static_cast<uint32_t>(rest[3] & 0x3F) << 24);

    // Byte 6: data_type occupies the upper 6 bits.
    data_type = rest[4];

    return true;
}

// ─── parse_scan_node ─────────────────────────────────────────────────────────
//
// Decodes one 5-byte measurement node from the continuous SCAN stream.
//
//  raw[0] bits [7:2] = quality (0..63, scaled to 0..255 here)
//  raw[0] bit  [1]   = S  (is_new_scan: 1 on the first point of a new revolution)
//  raw[0] bit  [0]   = !S (complement check bit, must be != bit[1])
//  raw[1] bits [7:1] = angle_q6[6:0]   (lower 7 bits of the 15-bit angle)
//  raw[1] bit  [0]   = C (check bit, must be 1)
//  raw[2]            = angle_q6[14:7]  (upper 8 bits of the 15-bit angle)
//  raw[3]            = distance_q2[7:0]
//  raw[4]            = distance_q2[15:8]
//
//  angle_deg   = angle_q6   / 64.0f
//  distance_mm = distance_q2 / 4.0f

ScanPoint RPLidarA1::parse_scan_node(const uint8_t* raw)
{
    ScanPoint pt;

    // ── Quality ───────────────────────────────────────────────────────────────
    // Bits [7:2] of raw[0], range 0-63.  Scale to 0-255 for the common interface.
    pt.quality = static_cast<uint8_t>((raw[0] >> 2) * 4);

    // ── New scan flag ─────────────────────────────────────────────────────────
    // S bit (bit 1) of raw[0].  The complement bit (bit 0) should be !S;
    // we don't hard-fail on mismatch but could add an assertion.
    const bool s_bit   = (raw[0] & 0x02) != 0;
    const bool s_compl = (raw[0] & 0x01) != 0;
    pt.is_new_scan = s_bit && !s_compl;

    // ── Angle ──────────────────────────────────────────────────────────────────
    // 15-bit fixed-point, resolution 1/64 degree.
    // Check bit C (raw[1] bit 0) should be 1 — ignore invalid nodes.
    const bool c_bit = (raw[1] & 0x01) != 0;
    (void)c_bit;  // could assert(c_bit) in debug builds

    const uint16_t angle_q6 = static_cast<uint16_t>(
        ((raw[1] & 0xFE) >> 1) | (static_cast<uint16_t>(raw[2]) << 7));
    pt.angle_deg = static_cast<float>(angle_q6) / 64.0f;

    // ── Distance ──────────────────────────────────────────────────────────────
    // 16-bit fixed-point, resolution 1/4 mm.
    const uint16_t dist_q2 = static_cast<uint16_t>(raw[3])
                           | (static_cast<uint16_t>(raw[4]) << 8);
    pt.distance_mm = static_cast<float>(dist_q2) / 4.0f;

    return pt;
}

// ─── read_loop ───────────────────────────────────────────────────────────────
//
// Runs on read_thread_.  Continuously reads 5-byte scan nodes from the serial
// port, parses them, and batches them into ScanFrames.
//
// Frame boundary detection:
//   When is_new_scan == true on an incoming node, the previous batch of points
//   constitutes a complete 360° sweep.  We publish it as a ScanFrame and start
//   accumulating the next one.

void RPLidarA1::read_loop()
{
    std::vector<ScanPoint> current_points;
    current_points.reserve(400);  // A1 produces ~360 points/rev at 5.5 Hz

    uint8_t node_buf[SCAN_NODE_LEN];

    while (running_.load()) {
        // Read exactly 5 bytes for one measurement node.
        // Use a short per-byte timeout so we can check running_ regularly.
        if (!serial_.read_exact(node_buf, SCAN_NODE_LEN, 200)) {
            // Timeout or error — check if we should keep running.
            if (!running_.load()) break;
            // Possible framing error: flush and re-sync.
            serial_.flush_input();
            continue;
        }

        ScanPoint pt = parse_scan_node(node_buf);

        // ── Frame boundary: is_new_scan marks the start of a new revolution ──
        if (pt.is_new_scan && !current_points.empty()) {
            // Publish the completed frame.
            ScanFrame frame;
            frame.points    = std::move(current_points);
            frame.timestamp = std::chrono::steady_clock::now();

            // RPM estimate.
            update_rpm_estimate();
            {
                std::lock_guard<std::mutex> lk(rpm_mutex_);
                frame.sensor_rpm = estimated_rpm_;
            }

            {
                std::lock_guard<std::mutex> lk(frame_mutex_);
                frame.frame_id = frame_counter_++;
            }

            publish_frame(std::move(frame));

            // Start fresh accumulation.
            current_points.clear();
            current_points.reserve(400);
        }

        current_points.push_back(pt);
    }
}

// ─── update_rpm_estimate ─────────────────────────────────────────────────────
//
// Called once per completed revolution.  Measures wall-clock time between
// consecutive is_new_scan events and converts to RPM.

void RPLidarA1::update_rpm_estimate()
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(rpm_mutex_);

    if (rpm_initialized_) {
        const float elapsed_s = std::chrono::duration<float>(
            now - last_scan_start_).count();
        if (elapsed_s > 0.0f) {
            // One revolution in elapsed_s seconds → RPM = 60 / elapsed_s
            estimated_rpm_ = 60.0f / elapsed_s;
        }
    }

    last_scan_start_  = now;
    rpm_initialized_  = true;
}

// ─── publish_frame ────────────────────────────────────────────────────────────

void RPLidarA1::publish_frame(ScanFrame frame)
{
    // Store in latest_frame_ (copy, so the callback gets a consistent snapshot).
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = frame;
    }

    // Invoke the user callback if registered.
    FrameCallback cb;
    {
        std::lock_guard<std::mutex> lk(frame_cb_mutex_);
        cb = frame_callback_;
    }
    if (cb) {
        cb(frame);
    }
}

// ─── set_error ────────────────────────────────────────────────────────────────

void RPLidarA1::set_error(const std::string& msg)
{
    std::lock_guard<std::mutex> lk(error_mutex_);
    error_message_ = msg;
}

} // namespace sensors