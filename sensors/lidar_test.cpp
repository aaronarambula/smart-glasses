// ─── lidar_test.cpp ──────────────────────────────────────────────────────────
// Standalone test that opens a LiDAR sensor, streams frames for a few seconds,
// and prints per-frame stats to stdout.
//
// Build:  cmake --build sensors/build --target lidar_test
// Run:    ./sensors/build/lidar_test [port] [model]
//
//   port   – serial device, default /dev/ttyUSB0
//   model  – "rplidar" (default) | "ld06"
//
// Example:
//   ./sensors/build/lidar_test /dev/ttyUSB0 rplidar
//   ./sensors/build/lidar_test /dev/ttyAMA0 ld06

#include "sensors/sensors.h"

#include <csignal>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <string>

static std::atomic<bool> g_quit{ false };

static void sig_handler(int) { g_quit.store(true); }

int main(int argc, char* argv[])
{
    // ── Parse arguments ───────────────────────────────────────────────────────

    sensors::LidarModel model = sensors::LidarModel::RPLidarA1;
    std::string port = sensors::default_port(model);

    if (argc >= 2) {
        port = argv[1];
    }
    if (argc >= 3) {
        const std::string arg = argv[2];
        if (arg == "ld06") {
            model = sensors::LidarModel::LD06;
        } else if (arg == "rplidar") {
            model = sensors::LidarModel::RPLidarA1;
        } else if (arg == "tfluna") {
            model = sensors::LidarModel::TFLuna;
        } else {
            std::fprintf(stderr,
                "Unknown model '%s'. Use 'rplidar', 'ld06', or 'tfluna'.\n",
                argv[2]);
            return 1;
        }
    }

    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    // ── Construct driver ──────────────────────────────────────────────────────

    std::printf("sensor  : %s\n", sensors::model_name(model).c_str());
    std::printf("port    : %s\n", port.c_str());
    std::fflush(stdout);

    auto lidar = sensors::make_lidar(model, port);

    // ── Open ──────────────────────────────────────────────────────────────────

    std::printf("Opening sensor …\n");
    if (!lidar->open()) {
        std::fprintf(stderr, "open() failed: %s\n",
                     lidar->error_message().c_str());
        return 1;
    }
    std::printf("Sensor open.\n");

    // ── Start ─────────────────────────────────────────────────────────────────

    std::printf("Starting scan …\n");
    if (!lidar->start()) {
        std::fprintf(stderr, "start() failed: %s\n",
                     lidar->error_message().c_str());
        lidar->close();
        return 1;
    }
    std::printf("Scanning. Press Ctrl+C to stop.\n\n");

    // ── Register frame callback and stream ────────────────────────────────────

    std::atomic<uint64_t> frame_count{ 0 };

    lidar->set_frame_callback([&](const sensors::ScanFrame& frame) {
        frame_count.fetch_add(1, std::memory_order_relaxed);

        // Closest point in the full 360°.
        const sensors::ScanPoint closest = frame.closest_point();

        // Closest point in the forward ±30° sector (330–30°).
        // Split into two ranges because the sector wraps around 0°.
        sensors::ScanPoint fwd;
        {
            float best_dist = std::numeric_limits<float>::max();
            for (const auto& p : frame.points) {
                if (!p.is_valid()) continue;
                const bool in_sector = p.angle_deg <= 30.0f || p.angle_deg >= 330.0f;
                if (in_sector && p.distance_mm < best_dist) {
                    best_dist = p.distance_mm;
                    fwd = p;
                }
            }
        }

        std::printf("frame %4llu | pts %3zu | rpm %5.1f | "
                    "closest %6.0f mm @ %5.1f° | "
                    "fwd(±30°) %s\n",
                    static_cast<unsigned long long>(frame.frame_id),
                    frame.size(),
                    frame.sensor_rpm,
                    closest.is_valid() ? closest.distance_mm : 0.0f,
                    closest.is_valid() ? closest.angle_deg   : 0.0f,
                    fwd.is_valid()
                        ? (std::to_string(static_cast<int>(fwd.distance_mm)) + " mm").c_str()
                        : "none");
        std::fflush(stdout);
    });

    // ── Wait until Ctrl+C ─────────────────────────────────────────────────────

    while (!g_quit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    std::printf("\nStopping …\n");
    lidar->stop();
    lidar->close();

    std::printf("Done. %llu frames captured.\n",
                static_cast<unsigned long long>(frame_count.load()));
    return 0;
}
