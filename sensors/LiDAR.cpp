// ─── LiDAR.cpp ───────────────────────────────────────────────────────────────
// Main LiDAR pipeline for the smart-glasses project.
//
// Responsibilities:
//   1. Instantiate the correct LiDAR driver (RPLIDAR A1 or LD06) based on
//      runtime configuration or compile-time default.
//   2. Open + start the sensor on the Raspberry Pi serial port.
//   3. On every completed ScanFrame:
//        a. Filter invalid / out-of-range points
//        b. Run obstacle detection (sector-based closest-point analysis)
//        c. Classify risk level (CLEAR, CAUTION, WARNING, DANGER)
//        d. Invoke an alert callback so the glasses hardware can respond
//           (buzzer, haptic, LED, etc.)
//   4. Provide a clean shutdown path (SIGINT / SIGTERM).
//
// Build:
//   This file is compiled as part of the sensors static library (see
//   sensors/CMakeLists.txt).  It exposes a single public entry point:
//
//     sensors::lidar_pipeline_run(config)
//
//   which blocks until the pipeline is stopped (Ctrl-C or stop() call).
//
// ─── Coordinate frame ────────────────────────────────────────────────────────
//
//   0° = directly in front of the wearer (forward)
//   Angles increase clockwise when viewed from above.
//
//   Detection sectors (configurable):
//     Forward cone  : 340° – 360° ∪ 0° – 20°   (±20° ahead)
//     Left flank    : 270° – 340°
//     Right flank   :  20° –  90°
//     Rear          :  90° – 270°   (less critical for glasses use-case)

#include "sensors/sensors.h"

#include <cstdint>
#include <cmath>
#include <csignal>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace sensors {

// ─── Risk levels ──────────────────────────────────────────────────────────────

enum class RiskLevel : uint8_t {
    CLEAR   = 0,   // nothing within the danger zone
    CAUTION = 1,   // object within outer warning zone
    WARNING = 2,   // object within inner warning zone
    DANGER  = 3,   // object within immediate danger zone
};

inline const char* risk_level_name(RiskLevel r) {
    switch (r) {
        case RiskLevel::CLEAR:   return "CLEAR";
        case RiskLevel::CAUTION: return "CAUTION";
        case RiskLevel::WARNING: return "WARNING";
        case RiskLevel::DANGER:  return "DANGER";
        default:                 return "UNKNOWN";
    }
}

// ─── Obstacle ─────────────────────────────────────────────────────────────────
//
// Represents the closest detected object in a particular angular sector.

struct Obstacle {
    float    angle_deg   = 0.0f;    // bearing to the closest point
    float    distance_mm = 0.0f;    // distance to the closest point
    RiskLevel risk       = RiskLevel::CLEAR;

    bool detected() const { return distance_mm > 0.0f; }
};

// ─── DetectionResult ─────────────────────────────────────────────────────────
//
// Full obstacle analysis for one ScanFrame, one entry per sector.

struct DetectionResult {
    uint64_t   frame_id    = 0;
    float      sensor_rpm  = 0.0f;

    Obstacle   forward;       // ±20° cone
    Obstacle   left_flank;    // 270°–340°
    Obstacle   right_flank;   //  20°–90°
    Obstacle   rear;          //  90°–270°

    // Overall highest risk across all sectors.
    RiskLevel overall_risk = RiskLevel::CLEAR;

    void compute_overall() {
        overall_risk = std::max({ forward.risk,
                                  left_flank.risk,
                                  right_flank.risk,
                                  rear.risk });
    }
};

// ─── PipelineConfig ───────────────────────────────────────────────────────────
//
// All tunable parameters for the LiDAR pipeline.  Provide your own instance
// and pass it to lidar_pipeline_run().

struct PipelineConfig {
    // ── Sensor selection ──────────────────────────────────────────────────────
    LidarModel  model       = LidarModel::RPLidarA1;
    std::string port        = "/dev/ttyUSB0";

    // ── Distance thresholds (mm) ──────────────────────────────────────────────
    float danger_mm  =  500.0f;   // DANGER  if object is closer than this
    float warning_mm = 1000.0f;   // WARNING if object is closer than this
    float caution_mm = 2000.0f;   // CAUTION if object is closer than this
    // Beyond caution_mm → CLEAR

    // ── Sector definitions (degrees) ──────────────────────────────────────────
    // Forward cone spans from fwd_min to fwd_max going through 0°.
    // Because it straddles 0°, we split it into two sub-ranges:
    //   [360 - fwd_half, 360) ∪ [0, fwd_half]
    float fwd_half_deg   = 20.0f;   // ±20° forward cone
    float right_max_deg  = 90.0f;   // right flank: fwd_half_deg → right_max_deg
    float rear_max_deg   = 270.0f;  // rear: right_max_deg → rear_max_deg
    // left flank: rear_max_deg → (360 - fwd_half_deg)

    // ── Filters ───────────────────────────────────────────────────────────────
    float    min_distance_mm = 50.0f;   // ignore points closer than this (sensor
                                         // noise / housing reflections)
    float    max_distance_mm = 6000.0f; // ignore points beyond this range (mm)
    uint8_t  min_quality     = 10;      // ignore low-quality returns

    // ── Alert callback ────────────────────────────────────────────────────────
    // Called from the pipeline thread on every frame with the detection result.
    // Keep it short — offload heavy work to another thread if needed.
    std::function<void(const DetectionResult&)> on_detection;

    // ── Debug output ──────────────────────────────────────────────────────────
    bool verbose = false;   // print per-frame summary to stdout
};

// ─── Pipeline internals ───────────────────────────────────────────────────────

namespace detail {

// Classifies a distance into a RiskLevel given the config thresholds.
inline RiskLevel classify_distance(float dist_mm, const PipelineConfig& cfg)
{
    if (dist_mm <= 0.0f)          return RiskLevel::CLEAR;
    if (dist_mm < cfg.danger_mm)  return RiskLevel::DANGER;
    if (dist_mm < cfg.warning_mm) return RiskLevel::WARNING;
    if (dist_mm < cfg.caution_mm) return RiskLevel::CAUTION;
    return RiskLevel::CLEAR;
}

// Returns true if `angle` falls in the range [lo, hi] (inclusive).
// Does NOT handle wrap-around; call twice for the forward sector.
inline bool in_sector(float angle, float lo, float hi)
{
    return angle >= lo && angle <= hi;
}

// Finds the closest valid point within [lo_deg, hi_deg] in the frame.
// Returns a zeroed ScanPoint if none found.
inline ScanPoint closest_in_range(const ScanFrame& frame,
                                   float lo_deg, float hi_deg,
                                   const PipelineConfig& cfg)
{
    ScanPoint best;
    float best_dist = cfg.max_distance_mm + 1.0f;

    for (const auto& pt : frame.points) {
        if (!pt.is_valid())                          continue;
        if (pt.quality < cfg.min_quality)            continue;
        if (pt.distance_mm < cfg.min_distance_mm)    continue;
        if (pt.distance_mm > cfg.max_distance_mm)    continue;
        if (!in_sector(pt.angle_deg, lo_deg, hi_deg)) continue;

        if (pt.distance_mm < best_dist) {
            best_dist = pt.distance_mm;
            best = pt;
        }
    }

    return best;
}

// Analyse one ScanFrame and fill a DetectionResult.
DetectionResult analyse_frame(const ScanFrame& frame,
                               const PipelineConfig& cfg)
{
    DetectionResult result;
    result.frame_id   = frame.frame_id;
    result.sensor_rpm = frame.sensor_rpm;

    const float fwd_lo  = 360.0f - cfg.fwd_half_deg;  // e.g. 340°
    const float fwd_hi  = cfg.fwd_half_deg;            // e.g.  20°

    // ── Forward (straddles 0°): search [340°,360°) and [0°,20°] ──────────────
    {
        // Right sub-range of forward cone: [0, fwd_hi]
        ScanPoint best_r = closest_in_range(frame, 0.0f, fwd_hi, cfg);
        // Left sub-range of forward cone: [fwd_lo, 360°]
        ScanPoint best_l = closest_in_range(frame, fwd_lo, 360.0f, cfg);

        // Pick the closer of the two halves.
        ScanPoint best;
        if (best_r.is_valid() && best_l.is_valid()) {
            best = (best_r.distance_mm <= best_l.distance_mm) ? best_r : best_l;
        } else if (best_r.is_valid()) {
            best = best_r;
        } else {
            best = best_l;
        }

        if (best.is_valid()) {
            result.forward.angle_deg   = best.angle_deg;
            result.forward.distance_mm = best.distance_mm;
            result.forward.risk        = classify_distance(best.distance_mm, cfg);
        }
    }

    // ── Right flank: [fwd_hi, right_max_deg] — e.g. [20°, 90°] ──────────────
    {
        ScanPoint best = closest_in_range(
            frame, cfg.fwd_half_deg, cfg.right_max_deg, cfg);
        if (best.is_valid()) {
            result.right_flank.angle_deg   = best.angle_deg;
            result.right_flank.distance_mm = best.distance_mm;
            result.right_flank.risk        = classify_distance(best.distance_mm, cfg);
        }
    }

    // ── Rear: [right_max_deg, rear_max_deg] — e.g. [90°, 270°] ─────────────
    {
        ScanPoint best = closest_in_range(
            frame, cfg.right_max_deg, cfg.rear_max_deg, cfg);
        if (best.is_valid()) {
            result.rear.angle_deg   = best.angle_deg;
            result.rear.distance_mm = best.distance_mm;
            result.rear.risk        = classify_distance(best.distance_mm, cfg);
        }
    }

    // ── Left flank: [rear_max_deg, 360 - fwd_half_deg] — e.g. [270°, 340°] ──
    {
        ScanPoint best = closest_in_range(
            frame, cfg.rear_max_deg, fwd_lo, cfg);
        if (best.is_valid()) {
            result.left_flank.angle_deg   = best.angle_deg;
            result.left_flank.distance_mm = best.distance_mm;
            result.left_flank.risk        = classify_distance(best.distance_mm, cfg);
        }
    }

    result.compute_overall();
    return result;
}

// Pretty-print one DetectionResult to stdout.
void print_result(const DetectionResult& r)
{
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "[frame " << std::setw(6) << r.frame_id
              << " | " << std::setw(5) << std::setprecision(1) << r.sensor_rpm
              << " RPM | " << risk_level_name(r.overall_risk) << "]\n";

    auto print_obstacle = [](const char* label, const Obstacle& obs) {
        if (obs.detected()) {
            std::cout << "  " << std::left << std::setw(12) << label
                      << std::right << std::setw(6) << std::setprecision(0)
                      << std::fixed << obs.distance_mm << " mm"
                      << "  @ " << std::setw(5) << std::setprecision(1)
                      << obs.angle_deg << "°"
                      << "  [" << risk_level_name(obs.risk) << "]\n";
        } else {
            std::cout << "  " << std::left << std::setw(12) << label
                      << "  clear\n";
        }
    };

    print_obstacle("Forward",     r.forward);
    print_obstacle("Right flank", r.right_flank);
    print_obstacle("Rear",        r.rear);
    print_obstacle("Left flank",  r.left_flank);
}

} // namespace detail

// ─── Global stop flag ────────────────────────────────────────────────────────
//
// Set by SIGINT/SIGTERM handler.  The pipeline checks this each frame.

static std::atomic<bool> g_stop_requested{ false };

static void signal_handler(int /*sig*/)
{
    g_stop_requested.store(true);
}

// ─── lidar_pipeline_run ──────────────────────────────────────────────────────
//
// Blocking entry point.  Call from main() or a dedicated thread.
//
//   1. Installs SIGINT / SIGTERM handlers for clean shutdown.
//   2. Opens and starts the LiDAR driver.
//   3. Registers a frame callback that runs the obstacle detection pipeline.
//   4. Blocks (with a sleep loop) until g_stop_requested or stop() is called.
//   5. Stops and closes the sensor before returning.
//
// Returns 0 on clean shutdown, non-zero on error.

int lidar_pipeline_run(PipelineConfig config)
{
    // ── Signal handling ───────────────────────────────────────────────────────
    g_stop_requested.store(false);
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── Instantiate driver ────────────────────────────────────────────────────
    std::unique_ptr<LidarBase> lidar;
    try {
        lidar = make_lidar(config.model, config.port);
    } catch (const std::exception& e) {
        std::cerr << "[LiDAR] Failed to create driver: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[LiDAR] Using " << lidar->model_name()
              << " on " << config.port << "\n";

    // ── Open serial port ──────────────────────────────────────────────────────
    if (!lidar->open()) {
        std::cerr << "[LiDAR] open() failed: " << lidar->error_message() << "\n";
        return 1;
    }
    std::cout << "[LiDAR] Port opened successfully.\n";

    // ── Register per-frame callback ───────────────────────────────────────────
    //
    // This lambda runs on the driver's internal read thread.  It:
    //   1. Analyses the frame for obstacles.
    //   2. Optionally prints verbose output.
    //   3. Calls the user-supplied on_detection callback.
    //
    // We capture `config` by reference — it is valid for the lifetime of
    // `lidar` because both are owned by this function's stack frame.

    lidar->set_frame_callback([&config](const ScanFrame& frame) {
        if (frame.empty()) return;

        // Run obstacle detection.
        DetectionResult result = detail::analyse_frame(frame, config);

        // Print verbose output if requested.
        if (config.verbose) {
            detail::print_result(result);
        }

        // Invoke the user alert callback.
        if (config.on_detection) {
            config.on_detection(result);
        }
    });

    // ── Start scanning ────────────────────────────────────────────────────────
    if (!lidar->start()) {
        std::cerr << "[LiDAR] start() failed: " << lidar->error_message() << "\n";
        lidar->close();
        return 1;
    }
    std::cout << "[LiDAR] Scanning started. Press Ctrl-C to stop.\n";

    // ── Main loop: block until stop is requested ──────────────────────────────
    while (!g_stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Surface any errors the driver logged (non-fatal).
        const std::string err = lidar->error_message();
        if (!err.empty()) {
            std::cerr << "[LiDAR] Driver warning: " << err << "\n";
        }
    }

    // ── Clean shutdown ────────────────────────────────────────────────────────
    std::cout << "\n[LiDAR] Shutdown requested — stopping sensor...\n";
    lidar->stop();
    lidar->close();
    std::cout << "[LiDAR] Sensor stopped. Pipeline exiting.\n";

    return 0;
}

// ─── stop_pipeline ────────────────────────────────────────────────────────────
//
// Can be called from any thread to request a graceful pipeline shutdown,
// equivalent to sending SIGINT.  Useful when the pipeline runs in a thread
// managed by the rest of the application rather than as the main process.

void stop_pipeline()
{
    g_stop_requested.store(true);
}

} // namespace sensors


// ─── Standalone main (optional) ──────────────────────────────────────────────
//
// Compiled only when LIDAR_STANDALONE is defined, allowing this file to be
// built as a self-contained executable for testing the sensor independently
// of the rest of the smart-glasses pipeline.
//
// Build example:
//   g++ -std=c++17 -DLIDAR_STANDALONE -Iinclude \
//       LiDAR.cpp rplidar_a1.cpp ld06.cpp -lpthread -o lidar_test
//
// Usage:
//   ./lidar_test [rplidar|ld06] [port]
//   ./lidar_test rplidar /dev/ttyUSB0
//   ./lidar_test ld06    /dev/ttyAMA0

#ifdef LIDAR_STANDALONE

#include <cstring>

int main(int argc, char* argv[])
{
    sensors::PipelineConfig cfg;

    // ── Parse CLI arguments ───────────────────────────────────────────────────
    if (argc >= 2) {
        if (std::strcmp(argv[1], "ld06") == 0) {
            cfg.model = sensors::LidarModel::LD06;
            cfg.port  = "/dev/ttyAMA0";
        } else {
            // Default: RPLIDAR A1
            cfg.model = sensors::LidarModel::RPLidarA1;
            cfg.port  = "/dev/ttyUSB0";
        }
    }
    if (argc >= 3) {
        cfg.port = argv[2];
    }

    // ── Configure detection thresholds ────────────────────────────────────────
    cfg.danger_mm  =  500.0f;   // < 0.5 m → DANGER
    cfg.warning_mm = 1000.0f;   // < 1.0 m → WARNING
    cfg.caution_mm = 2000.0f;   // < 2.0 m → CAUTION
    cfg.verbose    = true;      // print every frame to stdout

    // ── Alert callback: print a prominent warning for forward DANGER ──────────
    cfg.on_detection = [](const sensors::DetectionResult& r) {
        if (r.forward.risk == sensors::RiskLevel::DANGER) {
            std::cout << "\a"  // terminal bell
                      << "*** DANGER: obstacle "
                      << static_cast<int>(r.forward.distance_mm)
                      << " mm ahead at "
                      << static_cast<int>(r.forward.angle_deg)
                      << "° ***\n";
        }
    };

    return sensors::lidar_pipeline_run(std::move(cfg));
}

#endif // LIDAR_STANDALONE