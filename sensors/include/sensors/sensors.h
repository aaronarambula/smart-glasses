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

#include <memory>
#include <string>
#include <stdexcept>

namespace sensors {

// ─── LidarModel ──────────────────────────────────────────────────────────────
//
// Enum identifying which physical sensor is connected.
// Pass to make_lidar() to get the correct driver.

enum class LidarModel {
    RPLidarA1,  // Slamtec RPLIDAR A1M8 — USB-serial, 115200 baud
    LD06,       // LDROBOT LD06 / LD19 — UART, 230400 baud
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
        default:                    return "Unknown";
    }
}

} // namespace sensors