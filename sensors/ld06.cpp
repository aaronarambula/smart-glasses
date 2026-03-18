// ─── ld06.cpp ────────────────────────────────────────────────────────────────
// Full implementation of the LD06 LiDAR driver.
// See include/sensors/ld06.h for the protocol overview and API docs.

#include "sensors/ld06.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <thread>

namespace sensors {

// ─── Construction / Destruction ──────────────────────────────────────────────

LD06::LD06(std::string port_path)
    : LidarBase(std::move(port_path))
    , serial_(port_, 230400)
{
    current_frame_points_.reserve(500);  // ~460 points per revolution at 10 Hz
}

LD06::~LD06()
{
    stop();
    close();
}

// ─── open ────────────────────────────────────────────────────────────────────
//
// Opens the serial port at 230400 baud, 8N1.
// The LD06 streams data immediately at power-on — no init commands needed.
// We just flush the RX buffer to discard any stale partial packets.

bool LD06::open()
{
    if (!serial_.open()) {
        set_error("Failed to open serial port '" + port_ + "': "
                  + serial_.error_message());
        return false;
    }

    // Flush any bytes that arrived before we opened the port.
    serial_.flush_input();

    return true;
}

// ─── start ───────────────────────────────────────────────────────────────────

bool LD06::start()
{
    if (!serial_.is_open()) {
        set_error("start() called before open()");
        return false;
    }
    if (running_.load()) {
        return true;  // already running
    }

    running_.store(true);
    read_thread_ = std::thread(&LD06::read_loop, this);

    return true;
}

// ─── stop ────────────────────────────────────────────────────────────────────

void LD06::stop()
{
    if (!running_.load()) return;

    running_.store(false);

    if (read_thread_.joinable()) {
        read_thread_.join();
    }
}

// ─── close ───────────────────────────────────────────────────────────────────

void LD06::close()
{
    stop();
    serial_.close();
}

// ─── get_latest_frame ────────────────────────────────────────────────────────

ScanFrame LD06::get_latest_frame() const
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

// ─── set_frame_callback ───────────────────────────────────────────────────────

void LD06::set_frame_callback(FrameCallback cb)
{
    std::lock_guard<std::mutex> lock(frame_cb_mutex_);
    frame_callback_ = std::move(cb);
}

// ─── Status accessors ─────────────────────────────────────────────────────────

bool LD06::is_open() const
{
    return serial_.is_open();
}

bool LD06::is_running() const
{
    return running_.load();
}

std::string LD06::error_message() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
}

float LD06::reported_rpm() const
{
    std::lock_guard<std::mutex> lock(rpm_mutex_);
    return reported_rpm_;
}

void LD06::set_min_intensity(uint8_t threshold)
{
    min_intensity_.store(threshold);
}

uint8_t LD06::min_intensity() const
{
    return min_intensity_.load();
}

// ─── crc8 ────────────────────────────────────────────────────────────────────
//
// CRC-8/MAXIM (also known as CRC-8/1-Wire):
//   Polynomial : 0x4D (reflected: 0xB2, but we use the non-reflected form here
//                as verified against the LD06 datasheet examples)
//   Initial    : 0x00
//   Input/Output reflect: false
//
// Computed with a byte-at-a-time loop — no lookup table needed given the
// low packet rate (10 packets × 12 points per revolution).

uint8_t LD06::crc8(const uint8_t* data, size_t len)
{
    uint8_t crc = 0x00;

    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 0x80) {
                crc = static_cast<uint8_t>((crc << 1) ^ CRC_POLY);
            } else {
                crc <<= 1;
            }
        }
    }

    return crc;
}

// ─── check_crc ───────────────────────────────────────────────────────────────
//
// Verifies CRC over the first 46 bytes; byte[46] is the expected CRC.

bool LD06::check_crc(const uint8_t* raw)
{
    const uint8_t computed = crc8(raw, PACKET_LEN - 1);
    return computed == raw[PACKET_LEN - 1];
}

// ─── normalize_angle ─────────────────────────────────────────────────────────

float LD06::normalize_angle(float deg)
{
    while (deg >= 360.0f) deg -= 360.0f;
    while (deg  <   0.0f) deg += 360.0f;
    return deg;
}

// ─── angle_diff_cw ───────────────────────────────────────────────────────────
//
// Angular distance going clockwise from `from` to `to`, result in [0, 360).

float LD06::angle_diff_cw(float from_deg, float to_deg)
{
    float diff = to_deg - from_deg;
    while (diff <   0.0f) diff += 360.0f;
    while (diff >= 360.0f) diff -= 360.0f;
    return diff;
}

// ─── parse_packet ─────────────────────────────────────────────────────────────
//
// Parses a raw 47-byte LD06 packet (already CRC-validated) into an LD06Packet.
//
// Packet layout:
//   [0]     Header      : 0x54
//   [1]     VerLen      : 0x2C
//   [2-3]   Speed       : deg/s × 100, uint16 LE  →  divide by 100 for deg/s
//   [4-5]   StartAngle  : degrees × 100, uint16 LE
//   [6-41]  DataPoints  : 12 × 3 bytes
//               [0-1]  distance mm, uint16 LE
//               [2]    intensity / quality, uint8
//   [42-43] EndAngle    : degrees × 100, uint16 LE
//   [44-45] Timestamp   : milliseconds, uint16 LE (wraps at 30000)
//   [46]    CRC         : CRC-8/MAXIM

LD06Packet LD06::parse_packet(const uint8_t* raw)
{
    LD06Packet pkt;

    // Speed in degrees/second (raw value is deg/s × 100).
    const uint16_t speed_raw = static_cast<uint16_t>(raw[2])
                             | (static_cast<uint16_t>(raw[3]) << 8);
    pkt.speed_deg_per_sec = static_cast<float>(speed_raw) / 100.0f;

    // Start angle in degrees (raw value is degrees × 100).
    const uint16_t start_raw = static_cast<uint16_t>(raw[4])
                              | (static_cast<uint16_t>(raw[5]) << 8);
    pkt.start_angle_deg = static_cast<float>(start_raw) / 100.0f;

    // End angle in degrees (raw value is degrees × 100).
    const uint16_t end_raw = static_cast<uint16_t>(raw[42])
                           | (static_cast<uint16_t>(raw[43]) << 8);
    pkt.end_angle_deg = static_cast<float>(end_raw) / 100.0f;

    // Timestamp.
    pkt.timestamp_ms = static_cast<uint16_t>(raw[44])
                     | (static_cast<uint16_t>(raw[45]) << 8);

    // 12 distance + intensity measurements starting at byte 6.
    for (size_t i = 0; i < LD06Packet::NUM_POINTS; ++i) {
        const size_t offset = 6 + i * 3;

        const uint16_t dist_raw = static_cast<uint16_t>(raw[offset])
                                | (static_cast<uint16_t>(raw[offset + 1]) << 8);
        pkt.distances_mm[i]  = static_cast<float>(dist_raw);   // already in mm
        pkt.intensities[i]   = raw[offset + 2];
    }

    return pkt;
}

// ─── packet_to_scan_points ────────────────────────────────────────────────────
//
// Converts a parsed LD06Packet into ScanPoints and appends them to `dst`.
//
// Angle interpolation:
//   The 12 points are evenly distributed between start_angle and end_angle.
//   We must handle the 360° wrap-around when end_angle < start_angle
//   (e.g. start=358°, end=2°).
//
//   angular_span = clockwise arc from start_angle to end_angle
//   step = angular_span / (NUM_POINTS - 1)
//   point[i].angle = (start_angle + i * step) mod 360

void LD06::packet_to_scan_points(const LD06Packet& pkt,
                                  std::vector<ScanPoint>& dst) const
{
    const uint8_t min_int = min_intensity_.load();

    // Clockwise angular span from start to end (handles wrap-around).
    float span = angle_diff_cw(pkt.start_angle_deg, pkt.end_angle_deg);

    // For a single point (degenerate), span is 0 — treat as 0 step.
    const float step = (LD06Packet::NUM_POINTS > 1)
                     ? span / static_cast<float>(LD06Packet::NUM_POINTS - 1)
                     : 0.0f;

    for (size_t i = 0; i < LD06Packet::NUM_POINTS; ++i) {
        ScanPoint pt;

        // Interpolate angle.
        pt.angle_deg  = normalize_angle(
            pkt.start_angle_deg + static_cast<float>(i) * step);

        // Distance and quality.
        pt.distance_mm = pkt.distances_mm[i];
        pt.quality     = pkt.intensities[i];

        // Apply intensity filter: treat weak returns as invalid (distance = 0).
        if (pt.quality < min_int) {
            pt.distance_mm = 0.0f;
        }

        // The LD06 reports 0 distance for "no return" — mark as invalid.
        // (is_valid() already checks distance_mm > 0, but zero quality
        //  was already handled above.)

        dst.push_back(pt);
    }
}

// ─── find_and_parse_packet ───────────────────────────────────────────────────
//
// Scans buf[offset..buf_len) for a valid LD06 packet.
//
// A valid packet:
//   1. Starts with header byte 0x54
//   2. Followed by VerLen byte 0x2C
//   3. Is exactly PACKET_LEN (47) bytes long
//   4. Passes the CRC-8 check
//
// Returns the index one past the last consumed byte (i.e. where to continue
// reading from).  Returns `offset` unchanged if no complete packet was found.
// Populates `packet_out` on success.

size_t LD06::find_and_parse_packet(const uint8_t* buf, size_t buf_len,
                                    size_t offset,
                                    LD06Packet& packet_out)
{
    while (offset + PACKET_LEN <= buf_len) {
        // Look for the header byte.
        if (buf[offset] != HEADER_BYTE) {
            ++offset;
            continue;
        }

        // Check VerLen byte.
        if (buf[offset + 1] != VERLEN_BYTE) {
            ++offset;
            continue;
        }

        // We have a candidate packet starting at `offset`.
        // Verify CRC before accepting it.
        if (!check_crc(buf + offset)) {
            // Bad CRC: this is a false header match or a corrupted packet.
            // Advance past the header byte and keep searching.
            ++offset;
            continue;
        }

        // Valid packet found — parse it.
        packet_out = parse_packet(buf + offset);
        return offset + PACKET_LEN;
    }

    // Not enough data for a complete packet yet.
    return offset;
}

// ─── read_loop ───────────────────────────────────────────────────────────────
//
// Background thread body.  Reads raw bytes from the serial port into a ring
// buffer, extracts complete LD06 packets, converts them to ScanPoints, and
// assembles them into ScanFrames.
//
// Frame boundary detection:
//   The LD06 rotates continuously.  We detect a new 360° frame when the
//   start_angle of an incoming packet is less than the end_angle of the
//   previous packet — meaning the sensor has crossed the 0°/360° seam.
//
//   More precisely: we track last_end_angle_deg_.  When the clockwise arc
//   from last_end_angle to the new start_angle is large (>= 180°, meaning
//   the angle went backwards), we treat that as a new revolution.
//
// Buffer strategy:
//   We maintain a sliding window buffer.  After processing, consumed bytes
//   are discarded by memmove, keeping the buffer small and bounded.

void LD06::read_loop()
{
    // Raw byte accumulation buffer (larger than one packet to handle
    // partial reads and re-sync after framing errors).
    std::vector<uint8_t> buf;
    buf.reserve(READ_BUF_LEN);

    uint8_t read_chunk[READ_BUF_LEN];

    while (running_.load()) {
        // ── Read available bytes from serial port ────────────────────────────
        int n = serial_.read(read_chunk, sizeof(read_chunk), 100);
        if (n < 0) {
            // Read error — log and try to recover.
            if (!running_.load()) break;
            set_error("Serial read error: " + serial_.error_message());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            serial_.flush_input();
            buf.clear();
            last_end_angle_deg_ = -1.0f;
            continue;
        }

        if (n == 0) {
            // Timeout — no data yet.  Loop and check running_ again.
            continue;
        }

        // Append new bytes to our accumulation buffer.
        buf.insert(buf.end(), read_chunk, read_chunk + n);

        // ── Extract all complete packets from the buffer ───────────────────
        size_t pos = 0;

        while (true) {
            LD06Packet pkt;
            size_t new_pos = find_and_parse_packet(
                buf.data(), buf.size(), pos, pkt);

            if (new_pos == pos) {
                // No complete packet found starting from pos.
                break;
            }

            // Successfully parsed a packet — process it.
            pos = new_pos;

            // ── Update reported RPM ──────────────────────────────────────────
            {
                std::lock_guard<std::mutex> lk(rpm_mutex_);
                // speed_deg_per_sec / 6 = RPM  (360 deg/rev ÷ 60 s/min = 6)
                reported_rpm_ = pkt.speed_deg_per_sec / 6.0f;
            }

            // ── Detect frame boundary ────────────────────────────────────────
            //
            // A new revolution starts when the sensor crosses the 0°/360° mark.
            // We detect this when the clockwise arc from last_end_angle to the
            // new packet's start_angle is >= 180° (i.e. the angle went
            // backwards by more than half a turn).
            //
            // Special case: first packet ever (last_end_angle_deg_ == -1).

            bool is_new_revolution = false;

            if (last_end_angle_deg_ < 0.0f) {
                // Very first packet — begin accumulating but don't publish yet.
                is_new_revolution = false;
            } else {
                // Clockwise arc from last end to this start.
                float arc = angle_diff_cw(last_end_angle_deg_,
                                          pkt.start_angle_deg);
                // If the arc is > 180°, the angle wrapped backwards →
                // the sensor completed a revolution.
                is_new_revolution = (arc > 180.0f);
            }

            if (is_new_revolution && !current_frame_points_.empty()) {
                // Publish the completed frame.
                ScanFrame frame;
                frame.points    = std::move(current_frame_points_);
                frame.timestamp = std::chrono::steady_clock::now();

                {
                    std::lock_guard<std::mutex> lk(rpm_mutex_);
                    frame.sensor_rpm = reported_rpm_;
                }

                {
                    std::lock_guard<std::mutex> lk(frame_mutex_);
                    frame.frame_id = frame_counter_++;
                }

                publish_frame(std::move(frame));

                // Start a fresh frame.
                current_frame_points_.clear();
                current_frame_points_.reserve(500);
            }

            // ── Append this packet's points to the current frame ─────────────
            packet_to_scan_points(pkt, current_frame_points_);

            // Track the end angle of the last processed packet.
            last_end_angle_deg_ = pkt.end_angle_deg;
        }

        // ── Discard consumed bytes, keep the remainder ─────────────────────
        if (pos > 0) {
            buf.erase(buf.begin(), buf.begin() + static_cast<ptrdiff_t>(pos));
        }

        // Safety cap: if the buffer grew huge without producing a valid packet
        // (e.g. lots of garbage bytes from a framing error), trim it down.
        if (buf.size() > READ_BUF_LEN * 2) {
            buf.erase(buf.begin(),
                      buf.begin() + static_cast<ptrdiff_t>(buf.size() - READ_BUF_LEN));
            last_end_angle_deg_ = -1.0f;  // force re-sync
        }
    }
}

// ─── publish_frame ────────────────────────────────────────────────────────────

void LD06::publish_frame(ScanFrame frame)
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
    if (cb) {
        cb(frame);
    }
}

// ─── set_error ────────────────────────────────────────────────────────────────

void LD06::set_error(const std::string& msg)
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_message_ = msg;
}

} // namespace sensors