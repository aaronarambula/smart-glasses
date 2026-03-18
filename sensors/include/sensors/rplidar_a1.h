#pragma once

// ─── rplidar_a1.h ────────────────────────────────────────────────────────────
// Driver for the Slamtec RPLIDAR A1 (model A1M8) connected to a Raspberry Pi
// via USB-to-serial adapter (/dev/ttyUSB0) or UART (/dev/ttyAMA0 / serial0).
//
// Protocol overview (from Slamtec RPLIDAR Interface Protocol v2.1):
//   - UART, 115200 baud, 8N1
//   - Host sends a request packet; sensor responds with a descriptor + data
//   - Request format:  { 0xA5, cmd, payload_len, [payload], [checksum] }
//   - Response descriptor (7 bytes):
//       byte 0   : 0xA5
//       byte 1   : 0x5A
//       byte 2-5 : response data length (little-endian uint32)
//       byte 6   : { send_mode[1:0], data_type[7:2] }
//   - Scan response: continuous stream of 5-byte measurement nodes
//       byte 0   : { quality[7:2], S[1], ~S[0] }   S = start-of-new-scan flag
//       byte 1   : { angle_q6[6:0], C }             C = check bit (must be 1)
//       byte 2   : angle_q6[14:7]
//       byte 3   : distance_q2[7:0]
//       byte 4   : distance_q2[15:8]
//     angle_deg    = angle_q6  / 64.0
//     distance_mm  = distance_q2 / 4.0
//
// Commands used by this driver:
//   STOP        (0x25) – stop scanning, no response
//   RESET       (0x40) – soft reset, no response
//   SCAN        (0x20) – begin standard scan mode
//   GET_INFO    (0x50) – retrieve firmware / hardware info
//   GET_HEALTH  (0x52) – retrieve health status
//
// Lifecycle:
//   RPLidarA1 lidar("/dev/ttyUSB0");
//   lidar.open();    // open port, send STOP+RESET, verify health
//   lidar.start();   // send SCAN, launch background read thread
//   // ... poll lidar.get_latest_frame() or use set_frame_callback()
//   lidar.stop();    // send STOP, join thread
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
#include <chrono>

namespace sensors {

// ─── Device info returned by GET_INFO ────────────────────────────────────────

struct RPLidarDeviceInfo {
    uint8_t  model;                // sensor model code (A1 = 0x18)
    uint8_t  firmware_major;
    uint8_t  firmware_minor;
    uint8_t  hardware;
    uint8_t  serial_number[16];    // 128-bit serial number
};

// ─── Health status returned by GET_HEALTH ─────────────────────────────────────

enum class RPLidarHealthStatus : uint8_t {
    Good    = 0x00,
    Warning = 0x01,
    Error   = 0x02,
    Unknown = 0xFF,
};

struct RPLidarHealth {
    RPLidarHealthStatus status     = RPLidarHealthStatus::Unknown;
    uint16_t            error_code = 0;
};

// ─── RPLidarA1 ────────────────────────────────────────────────────────────────

class RPLidarA1 : public LidarBase {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // port_path : serial device, e.g. "/dev/ttyUSB0" or "/dev/ttyAMA0"
    // The RPLIDAR A1 always runs at 115200 baud — no baud argument needed.
    explicit RPLidarA1(std::string port_path);

    // Destructor: stops scanning and closes the port if still open.
    ~RPLidarA1() override;

    // ── LidarBase interface ───────────────────────────────────────────────────

    // Opens the serial port, sends STOP (clears any previous scan), sends
    // RESET, waits for the sensor to boot (≈2 ms per spec), and verifies
    // health status.  Returns false if the port cannot be opened or the
    // sensor reports an error.
    bool open() override;

    // Sends the SCAN command and starts the background read thread.
    // open() must have been called and returned true first.
    bool start() override;

    // Sends STOP, signals the read thread to exit, and joins it.
    // The sensor motor spins down after STOP.
    void stop() override;

    // Closes the serial port.  Calls stop() first if still running.
    void close() override;

    // Returns a copy of the most recently completed ScanFrame.
    // Thread-safe (protected by frame_mutex_).
    ScanFrame get_latest_frame() const override;

    // Register / clear the frame-complete callback.
    // Invoked from the background thread — keep it short.
    void set_frame_callback(FrameCallback cb) override;

    bool        is_open()    const override;
    bool        is_running() const override;
    std::string error_message() const override;
    std::string model_name()   const override { return "RPLIDAR A1M8"; }

    // ── Extended API ──────────────────────────────────────────────────────────

    // Query device firmware / hardware info (GET_INFO command).
    // Returns false if the query fails.
    bool get_device_info(RPLidarDeviceInfo& info_out);

    // Query sensor health (GET_HEALTH command).
    // Returns false if the query fails.
    bool get_health(RPLidarHealth& health_out);

    // Returns the most recently computed rotation speed in RPM.
    // Estimated by the driver from the time between consecutive
    // is_new_scan markers.  Returns 0 if not yet available.
    float estimated_rpm() const;

private:
    // ── Serial port ───────────────────────────────────────────────────────────

    SerialPort serial_;

    // ── Background thread ─────────────────────────────────────────────────────

    std::thread      read_thread_;
    std::atomic<bool> running_{ false };

    // ── Latest frame (guarded by frame_mutex_) ─────────────────────────────

    mutable std::mutex frame_mutex_;
    ScanFrame          latest_frame_;
    uint64_t           frame_counter_{ 0 };

    // ── Frame callback ────────────────────────────────────────────────────────

    mutable std::mutex frame_cb_mutex_;
    FrameCallback      frame_callback_;

    // ── RPM estimation ────────────────────────────────────────────────────────

    mutable std::mutex rpm_mutex_;
    float              estimated_rpm_{ 0.0f };
    std::chrono::time_point<std::chrono::steady_clock> last_scan_start_;
    bool               rpm_initialized_{ false };

    // ── Error state ───────────────────────────────────────────────────────────

    mutable std::mutex  error_mutex_;
    std::string         error_message_;

    // ── Protocol constants ────────────────────────────────────────────────────

    static constexpr uint8_t SYNC_BYTE          = 0xA5;
    static constexpr uint8_t SYNC_BYTE2         = 0x5A;

    static constexpr uint8_t CMD_STOP           = 0x25;
    static constexpr uint8_t CMD_RESET          = 0x40;
    static constexpr uint8_t CMD_SCAN           = 0x20;
    static constexpr uint8_t CMD_GET_INFO       = 0x50;
    static constexpr uint8_t CMD_GET_HEALTH     = 0x52;

    static constexpr size_t  DESCRIPTOR_LEN     = 7;
    static constexpr size_t  SCAN_NODE_LEN      = 5;   // bytes per measurement
    static constexpr size_t  INFO_RESPONSE_LEN  = 20;  // GET_INFO payload bytes
    static constexpr size_t  HEALTH_RESPONSE_LEN = 3;  // GET_HEALTH payload bytes

    // How long to wait after RESET before the sensor is ready (ms).
    static constexpr int RESET_WAIT_MS          = 500;

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Sends a command with no payload.
    bool send_command(uint8_t cmd);

    // Reads the 7-byte response descriptor that precedes every command reply.
    // Populates data_response_len and data_type on success.
    bool read_response_descriptor(uint32_t& data_response_len,
                                  uint8_t&  data_type);

    // Parses one 5-byte scan node into a ScanPoint.
    static ScanPoint parse_scan_node(const uint8_t* raw);

    // Main loop executed by read_thread_.
    void read_loop();

    // Stores an error message in a thread-safe way.
    void set_error(const std::string& msg);

    // Updates the RPM estimate when a new scan start is detected.
    void update_rpm_estimate();

    // Publishes a completed frame: stores it in latest_frame_ and
    // invokes the frame callback if set.
    void publish_frame(ScanFrame frame);
};

} // namespace sensors