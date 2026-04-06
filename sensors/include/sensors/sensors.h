#pragma once

// ─── sensors.h ───────────────────────────────────────────────────────────────
// Umbrella header for the sensors subsystem.
//
// Include this single file to pull in:
//   - ScanPoint / ScanFrame data types
//   - LidarBase abstract interface
//   - RPLidarA1 driver
//   - LD06 driver
//   - make_lidar() factory function
//
// Usage:
//   #include "sensors/sensors.h"
//
//   // Auto-detect or explicitly choose a sensor:
//   auto lidar = sensors::make_lidar(sensors::LidarModel::RPLidarA1,
//                                    "/dev/ttyUSB0");
//   lidar->open();
//   lidar->start();
//
//   while (true) {
//       auto frame = lidar->get_latest_frame();
//       auto closest = frame.closest_in_sector(340.0f, 360.0f);  // forward ±20°
//       if (closest.is_valid() && closest.distance_mm < 500.0f) {
//           // obstacle within 0.5 m — trigger alert
//       }
//   }
//
//   lidar->stop();
//   lidar->close();

#include "lidar_base.h"
#include "rplidar_a1.h"
#include "ld06.h"
#include "tfluna.h"
#include "ultrasonic_fallback.h"
#include "ultrasonic_hc_sr04.h"
#include "camera_fallback.h"

#include <memory>
#include <string>
#include <stdexcept>

// Forward declaration for the sim bridge function (global namespace, extern "C").
// Defined in sim/src/sim_lidar.cpp. Only resolves at link time when sim_lib
// is linked (cmake -DUSE_SIM=ON). Declared here so make_lidar(Sim) can call
// it without pulling in any sim/ headers into sensors/.
#ifdef USE_SIM
extern "C" sensors::LidarBase* sim_make_lidar_raw(const std::string&);
#endif

namespace sensors {

// ─── LidarModel ──────────────────────────────────────────────────────────────
//
// Enum identifying which physical sensor is connected.
// Pass to make_lidar() to get the correct driver.

enum class LidarModel {
    RPLidarA1,  // Slamtec RPLIDAR A1M8 — USB-serial, 115200 baud
    LD06,       // LDROBOT LD06 / LD19 — UART, 230400 baud
    TFLuna,     // Benewake TF-Luna V1.3 — single-point ToF, UART, 115200 baud
    Ultrasonic, // HC-SR04-style forward fallback, exposed as a narrow cluster
    Camera,     // OpenCV front camera fallback, exposed as a narrow cluster
    Sim,        // SimLidar — synthetic outdoor scenes, no hardware required
                // port_path = "sim://sidewalk" | "sim://crossing" |
                //             "sim://hallway"  | "sim://parking_lot" |
                //             "sim://cyclist_overtake" | "sim://crowd"
};

// ─── make_lidar ──────────────────────────────────────────────────────────────
//
// Factory function: constructs the appropriate driver and returns it as a
// unique_ptr to the abstract LidarBase interface.
//
//   model     : which sensor is connected
//   port_path : serial device path, e.g. "/dev/ttyUSB0" or "/dev/ttyAMA0"
//
// Throws std::invalid_argument if `model` is unrecognised.
//
// Example:
//   auto lidar = sensors::make_lidar(sensors::LidarModel::LD06,
//                                    "/dev/ttyAMA0");

inline std::unique_ptr<LidarBase> make_lidar(LidarModel model,
                                              const std::string& port_path)
{
    switch (model) {
        case LidarModel::RPLidarA1:
            return std::make_unique<RPLidarA1>(port_path);

        case LidarModel::LD06:
            return std::make_unique<LD06>(port_path);

        case LidarModel::TFLuna:
            return std::make_unique<sensors::TFLuna>(port_path);

        case LidarModel::Ultrasonic:
            return std::make_unique<UltrasonicFallback>(port_path);

        case LidarModel::Camera:
            return std::make_unique<CameraFallback>(port_path);

#ifdef USE_SIM
        case LidarModel::Sim: {
            // sim_make_lidar_raw is declared in the GLOBAL namespace (extern "C"
            // in sim/src/sim_lidar.cpp) to avoid C++ name mangling. We declare
            // it here at the point of use, outside any namespace, by temporarily
            // closing the sensors namespace, declaring the extern, and re-opening.
            // This keeps sensors.h free of any #include of sim/ headers.
            return std::unique_ptr<LidarBase>(::sim_make_lidar_raw(port_path));
        }
#endif


        default:
            throw std::invalid_argument(
                "sensors::make_lidar: unknown LidarModel");
    }
}

// ─── default_port ────────────────────────────────────────────────────────────
//
// Returns the most likely serial port for a given sensor on Raspberry Pi OS.
//   RPLidarA1 : USB-to-serial adapter → /dev/ttyUSB0
//   LD06      : GPIO UART header      → /dev/ttyAMA0
//
// These are just sensible defaults; always allow the user to override.

inline std::string default_port(LidarModel model) {
    switch (model) {
        case LidarModel::RPLidarA1: return "/dev/ttyUSB0";
        case LidarModel::LD06:      return "/dev/ttyAMA0";
        case LidarModel::TFLuna:    return "/dev/ttyAMA0";
        case LidarModel::Ultrasonic:return "ultrasonic://23,24?hz=10";
        case LidarModel::Camera:    return "camera://0?width=640&height=480&hz=10";
#ifdef USE_SIM
        case LidarModel::Sim:       return "sim://sidewalk";
#endif
        default:                    return "/dev/ttyUSB0";
    }
}

// ─── model_name ──────────────────────────────────────────────────────────────
//
// Human-readable sensor name, without constructing a driver object.

inline std::string model_name(LidarModel model) {
    switch (model) {
        case LidarModel::RPLidarA1: return "RPLIDAR A1M8";
        case LidarModel::LD06:      return "LD06";
        case LidarModel::TFLuna:    return "TF-Luna";
        case LidarModel::Ultrasonic:return "UltrasonicFallback";
        case LidarModel::Camera:    return "CameraFallback";
#ifdef USE_SIM
        case LidarModel::Sim:       return "SimLidar (synthetic)";
#endif
        default:                    return "Unknown";
    }
}

} // namespace sensors
