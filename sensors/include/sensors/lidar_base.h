#pragma once

// ─── lidar_base.h ────────────────────────────────────────────────────────────
// Abstract base class for all LiDAR sensors used in the smart-glasses project.
//
// Concrete drivers (RPLidar A1, LD06) inherit from LidarBase and implement
// the pure-virtual interface.  The rest of the pipeline (obstacle detection,
// risk model) only ever touches LidarBase* so sensors are interchangeable.
//
// Data model
// ──────────
//   ScanPoint  – one distance+angle+quality measurement
//   ScanFrame  – one full 360° sweep (or the best approximation the sensor
//                can provide), containing N ScanPoints
//
// Thread safety
// ─────────────
//   The driver runs its read loop on an internal thread.  Callers retrieve
//   completed frames through get_latest_frame() which is protected by a mutex,
//   so it is safe to call from any thread.

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <chrono>

namespace sensors {

// ─── ScanPoint ───────────────────────────────────────────────────────────────
//
// One distance measurement from the LiDAR.
//
//   angle_deg  : bearing in degrees, 0° = forward, clockwise positive
//                range [0, 360)
//   distance_mm: distance to the nearest object in millimetres
//                0 means "invalid / no return"
//   quality    : signal quality, sensor-specific scale [0, 255]
//                0 = unusable, 255 = perfect
//   is_new_scan: true on the first point of a fresh 360° revolution
//                (used internally by drivers to detect frame boundaries)

struct ScanPoint {
    float    angle_deg    = 0.0f;
    float    distance_mm  = 0.0f;
    uint8_t  quality      = 0;
    bool     is_new_scan  = false;

    // ── Derived helpers ───────────────────────────────────────────────────────

    // Convert polar (angle, distance) to Cartesian (x forward, y left).
    // x: distance along the forward axis (positive = in front)
    // y: distance along the lateral axis (positive = to the left)
    void to_cartesian(float& x_mm, float& y_mm) const {
        const float rad = angle_deg * (static_cast<float>(M_PI) / 180.0f);
        x_mm =  distance_mm * std::cos(rad);
        y_mm = -distance_mm * std::sin(rad);
    }

    // Returns true if the measurement should be trusted.
    bool is_valid() const {
        return distance_mm > 0.0f && quality > 0;
    }
};

// ─── ScanFrame ───────────────────────────────────────────────────────────────
//
// A complete (or near-complete) 360° sweep.
//
//   points      : all ScanPoints collected during this revolution, in the
//                 order the sensor reported them (typically increasing angle)
//   timestamp   : wall-clock time when the frame was completed
//   frame_id    : monotonically increasing counter, starts at 0
//   sensor_rpm  : rotation speed in RPM reported or estimated by the driver
//                 (0 if the sensor does not expose this)

struct ScanFrame {
    std::vector<ScanPoint>                          points;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    uint64_t                                        frame_id  = 0;
    float                                           sensor_rpm = 0.0f;

    // ── Convenience accessors ─────────────────────────────────────────────────

    bool empty() const { return points.empty(); }
    size_t size() const { return points.size(); }

    // Returns the closest valid point in the frame, or a zeroed ScanPoint
    // if the frame is empty / all points are invalid.
    ScanPoint closest_point() const {
        ScanPoint best;
        float best_dist = std::numeric_limits<float>::max();
        for (const auto& p : points) {
            if (p.is_valid() && p.distance_mm < best_dist) {
                best_dist = p.distance_mm;
                best = p;
            }
        }
        return best;
    }

    // Returns the closest valid point within [min_angle, max_angle] degrees.
    // Useful for narrowing detection to the forward cone.
    ScanPoint closest_in_sector(float min_angle_deg, float max_angle_deg) const {
        ScanPoint best;
        float best_dist = std::numeric_limits<float>::max();
        for (const auto& p : points) {
            if (!p.is_valid()) continue;
            if (p.angle_deg < min_angle_deg || p.angle_deg > max_angle_deg) continue;
            if (p.distance_mm < best_dist) {
                best_dist = p.distance_mm;
                best = p;
            }
        }
        return best;
    }

    // Filter points to only those within a distance threshold (mm).
    std::vector<ScanPoint> points_within(float max_distance_mm) const {
        std::vector<ScanPoint> result;
        for (const auto& p : points) {
            if (p.is_valid() && p.distance_mm <= max_distance_mm) {
                result.push_back(p);
            }
        }
        return result;
    }
};

// ─── LidarBase ───────────────────────────────────────────────────────────────
//
// Abstract interface all concrete LiDAR drivers must implement.
//
// Lifecycle:
//   1. Construct the driver with the serial port path.
//   2. Call open()  — opens the port, negotiates with the sensor.
//   3. Call start() — begins the background scan loop.
//   4. Poll get_latest_frame() from your main thread as fast as you need.
//      Or register an on_frame callback for push-style delivery.
//   5. Call stop()  — halts the scan loop (sensor motor stops if supported).
//   6. Call close() — closes the serial port and releases resources.
//      (Destructor calls stop()+close() automatically if still running.)

class LidarBase {
public:
    // ── Callback type ─────────────────────────────────────────────────────────
    // Registered via set_frame_callback().  Called from the driver's internal
    // thread — keep it short or hand off to a queue.
    using FrameCallback = std::function<void(const ScanFrame&)>;

    // ── Construction / destruction ────────────────────────────────────────────

    // port: e.g. "/dev/ttyUSB0" or "/dev/ttyAMA0"
    explicit LidarBase(std::string port) : port_(std::move(port)) {}

    // Destructor stops the driver gracefully.
    virtual ~LidarBase() = default;

    // Non-copyable (owns a serial fd and a thread).
    LidarBase(const LidarBase&)            = delete;
    LidarBase& operator=(const LidarBase&) = delete;
    LidarBase(LidarBase&&)                 = default;
    LidarBase& operator=(LidarBase&&)      = default;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Opens the serial port and initialises the sensor.
    // Returns true on success, false on error (check error_message()).
    virtual bool open()  = 0;

    // Starts the background scan thread. open() must have succeeded first.
    virtual bool start() = 0;

    // Stops scanning and the background thread.  Safe to call multiple times.
    virtual void stop()  = 0;

    // Closes the serial port and frees all resources.
    virtual void close() = 0;

    // ── Data retrieval ────────────────────────────────────────────────────────

    // Returns a copy of the most recently completed ScanFrame.
    // Thread-safe.  Returns an empty frame if no scan has completed yet.
    virtual ScanFrame get_latest_frame() const = 0;

    // Register a callback invoked on every completed frame.
    // Pass nullptr to clear the callback.
    // The callback runs on the driver's internal thread.
    virtual void set_frame_callback(FrameCallback cb) = 0;

    // ── Status ────────────────────────────────────────────────────────────────

    virtual bool        is_open()    const = 0;
    virtual bool        is_running() const = 0;

    // Human-readable description of the last error (empty if none).
    virtual std::string error_message() const = 0;

    // Sensor model string, e.g. "RPLIDAR A1M8" or "LD06".
    virtual std::string model_name()   const = 0;

    // Serial port path supplied at construction.
    const std::string& port() const { return port_; }

protected:
    std::string port_;
};

} // namespace sensors