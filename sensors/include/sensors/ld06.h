#pragma once

// ─── ld06.h ──────────────────────────────────────────────────────────────────
// Driver for the LDROBOT LD06 (also sold as LD19 / OKDO LiDAR Hat) connected
// to a Raspberry Pi via UART (/dev/ttyAMA0, /dev/serial0, or /dev/ttyUSB0).
//
// Protocol overview (from LD06 datasheet and community reverse-engineering):
//   - UART, 230400 baud, 8N1
//   - Sensor streams packets continuously — no commands needed to start
//   - Each packet is exactly 47 bytes:
//
//       Offset  Size  Field
//       ──────  ────  ─────────────────────────────────────────────────────
//         0      1    Header        : always 0x54
//         1      1    VerLen        : always 0x2C (ver=1, len=12 points)
//         2      2    Speed         : rotation speed in degrees/second (uint16 LE)
//         4      2    StartAngle    : start angle * 100 in degrees   (uint16 LE)
//         6     36    DataPoints    : 12 × 3-byte measurement records
//                                     [0..1] distance in mm (uint16 LE)
//                                     [2]    intensity / quality (uint8)
//        42      2    EndAngle      : end angle * 100 in degrees     (uint16 LE)
//        44      2    Timestamp     : milliseconds, wraps at 30000   (uint16 LE)
//        46      1    CRC           : CRC-8/MAXIM over bytes 0..45
//
//   - The 12 distance measurements are evenly spaced between StartAngle and
//     EndAngle.  The angular step between adjacent points is:
//         step = (EndAngle - StartAngle) / 11   (handle 360° wrap-around)
//
//   - A new 360° frame is detected when the accumulated angles wrap past 360°
//     (i.e. StartAngle of the incoming packet < EndAngle of the previous packet).
//
//   - The sensor rotates at ~10 Hz by default (no PWM control needed here).
//     Speed in deg/s divided by 360 gives the approximate RPM×6.
//
// No commands are sent to the sensor — just open the port, read, and parse.
//
// Lifecycle:
//   LD06 lidar("/dev/ttyAMA0");
//   lidar.open();    // open serial port at 230400 baud
//   lidar.start();   // launch background read+parse thread
//   // ... poll lidar.get_latest_frame() or use set_frame_callback()
//   lidar.stop();    // signal thread to exit and join it
//   lidar.close();   // close serial port

#include "lidar_base.h"
#include "serial_port.h"

#include <cstdint>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <vector>
#include <array>

namespace sensors {

// ─── LD06 packet (parsed, not raw) ────────────────────────────────────────────
//
// Populated by parse_packet(); used internally to build ScanFrames.

struct LD06Packet {
    float    speed_deg_per_sec = 0.0f;  // rotation speed reported by sensor
    float    start_angle_deg   = 0.0f;  // bearing of first point in packet
    float    end_angle_deg     = 0.0f;  // bearing of last  point in packet
    uint16_t timestamp_ms      = 0;     // sensor timestamp (wraps at 30000 ms)

    // 12 measurement points carried in this packet.
    static constexpr size_t NUM_POINTS = 12;
    std::array<float,   NUM_POINTS> distances_mm  {};  // 0 = no return
    std::array<uint8_t, NUM_POINTS> intensities   {};  // signal quality [0,255]
};

// ─── LD06 ─────────────────────────────────────────────────────────────────────

class LD06 : public LidarBase {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // port_path : serial device path, e.g. "/dev/ttyAMA0" or "/dev/ttyUSB0"
    // The LD06 always streams at 230400 baud — baud rate is not configurable.
    explicit LD06(std::string port_path);

    // Destructor: stops the read thread and closes the port.
    ~LD06() override;

    // ── LidarBase interface ───────────────────────────────────────────────────

    // Opens the serial port at 230400 baud, 8N1.
    // The LD06 starts streaming immediately after power-on; open() just
    // prepares the port and flushes any stale bytes from the RX buffer.
    // Returns false if the port cannot be opened.
    bool open() override;

    // Launches the background read/parse thread.
    // open() must have been called and returned true first.
    bool start() override;

    // Signals the background thread to stop and joins it.
    void stop() override;

    // Closes the serial port. Calls stop() first if still running.
    void close() override;

    // Returns a copy of the most recently completed ScanFrame.
    // Thread-safe (protected by frame_mutex_).
    ScanFrame get_latest_frame() const override;

    // Register / clear the per-frame callback.
    // Called from the background thread — keep it fast.
    void set_frame_callback(FrameCallback cb) override;

    bool        is_open()    const override;
    bool        is_running() const override;
    std::string error_message() const override;
    std::string model_name()   const override { return "LD06"; }

    // ── Extended API ──────────────────────────────────────────────────────────

    // Returns the most recently reported rotation speed in RPM.
    // Derived from the Speed field in the latest packet
    // (speed_deg_per_sec / 6.0 = RPM).
    // Returns 0 if no packet has been received yet.
    float reported_rpm() const;

    // Minimum intensity threshold: points with intensity below this value
    // are treated as invalid (distance_mm set to 0 in the ScanPoint).
    // Default: 10.  Increase to filter noisy/weak returns.
    void  set_min_intensity(uint8_t threshold);
    uint8_t min_intensity() const;

private:
    // ── Serial port ───────────────────────────────────────────────────────────

    SerialPort serial_;

    // ── Background thread ─────────────────────────────────────────────────────

    std::thread       read_thread_;
    std::atomic<bool> running_{ false };

    // ── Latest complete frame (guarded by frame_mutex_) ───────────────────────

    mutable std::mutex frame_mutex_;
    ScanFrame          latest_frame_;
    uint64_t           frame_counter_{ 0 };

    // ── Frame callback ────────────────────────────────────────────────────────

    mutable std::mutex frame_cb_mutex_;
    FrameCallback      frame_callback_;

    // ── Speed / RPM (guarded by rpm_mutex_) ──────────────────────────────────

    mutable std::mutex rpm_mutex_;
    float              reported_rpm_{ 0.0f };

    // ── Error state (guarded by error_mutex_) ────────────────────────────────

    mutable std::mutex error_mutex_;
    std::string        error_message_;

    // ── Intensity filter ──────────────────────────────────────────────────────

    std::atomic<uint8_t> min_intensity_{ 10 };

    // ── Protocol constants ────────────────────────────────────────────────────

    static constexpr uint8_t HEADER_BYTE   = 0x54;  // first byte of every packet
    static constexpr uint8_t VERLEN_BYTE   = 0x2C;  // second byte (version + len)
    static constexpr size_t  PACKET_LEN    = 47;    // total bytes per packet
    static constexpr size_t  NUM_POINTS    = 12;    // measurements per packet
    static constexpr size_t  CRC_POLY      = 0x4D;  // CRC-8/MAXIM polynomial

    // Read buffer: we accumulate raw serial bytes here until we have a full
    // packet.  Two packets worth ensures we can always find a header even if
    // the first sync is mid-stream.
    static constexpr size_t  READ_BUF_LEN  = PACKET_LEN * 4;

    // ── In-progress frame accumulation ────────────────────────────────────────
    //
    // The LD06 sends packets continuously.  We accumulate ScanPoints from
    // successive packets into current_frame_points_ until we detect a
    // 360° wrap-around, at which point we publish the frame and start fresh.

    std::vector<ScanPoint> current_frame_points_;
    float                  last_end_angle_deg_{ -1.0f };  // -1 = not yet started

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Main loop executed by read_thread_.
    void read_loop();

    // Attempts to find and parse a complete, CRC-valid packet starting at or
    // after buf[offset].  Returns the index one past the last consumed byte,
    // or offset if no complete packet was found.
    // On success, populates packet_out.
    size_t find_and_parse_packet(const uint8_t* buf, size_t buf_len,
                                 size_t offset,
                                 LD06Packet& packet_out);

    // Parses a raw 47-byte buffer (already CRC-validated) into an LD06Packet.
    static LD06Packet parse_packet(const uint8_t* raw);

    // Converts an LD06Packet into ScanPoints and appends them to dst.
    // Applies the intensity threshold filter.
    void packet_to_scan_points(const LD06Packet& pkt,
                               std::vector<ScanPoint>& dst) const;

    // Verifies the CRC-8/MAXIM checksum of a raw 47-byte packet.
    // CRC is computed over bytes [0..45]; byte [46] is the expected CRC.
    static bool check_crc(const uint8_t* raw);

    // Computes CRC-8/MAXIM (polynomial 0x4D, init 0x00) over `len` bytes.
    static uint8_t crc8(const uint8_t* data, size_t len);

    // Publishes a completed ScanFrame: stores in latest_frame_ and calls
    // the frame callback if registered.
    void publish_frame(ScanFrame frame);

    // Stores an error message in a thread-safe way.
    void set_error(const std::string& msg);

    // Normalises an angle to [0, 360).
    static float normalize_angle(float deg);

    // Computes the angular distance from a to b going clockwise,
    // i.e. the result is always in [0, 360).
    static float angle_diff_cw(float from_deg, float to_deg);
};

} // namespace sensors