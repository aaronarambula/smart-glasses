// ─── LiDAR.cpp ───────────────────────────────────────────────────────────────
// Sensor utility helpers shared across the sensors_lib components.
//
// The full obstacle-detection pipeline (perception → prediction → audio →
// agent) now lives in the modules above sensors/. This file retains only the
// thin helpers that sensors_lib itself legitimately needs:
//
//   make_lidar()      — factory that constructs the correct driver
//   default_port()    — returns the default serial path per sensor model
//   model_name()      — human-readable sensor name string
//
// All three are declared inline in include/sensors/sensors.h and therefore
// require no definitions here. This file exists to satisfy the CMake
// sensors_lib STATIC target (which lists LiDAR.cpp as a source) and as the
// natural home for any future sensor-layer utilities that are too large to
// live in a header.
//
// ─── What was removed ────────────────────────────────────────────────────────
//
// The previous version of this file contained a self-contained pipeline:
//   RiskLevel enum, Obstacle, DetectionResult, PipelineConfig,
//   lidar_pipeline_run(), stop_pipeline(), and a #ifdef LIDAR_STANDALONE main()
//
// All of that has been superseded by:
//   perception/   — OccupancyMap, DBSCAN clusterer, Kalman tracker
//   prediction/   — TTC engine, aaronnet MLP risk predictor
//   audio/        — TTS engine, alert policy
//   agent/        — OpenAI GPT-4o integration
//   app/main.cpp  — single entry point wiring everything together
//
// Keeping the old code would create two competing risk-classification systems
// in the same binary and confuse anyone reading the codebase.

#include "sensors/sensors.h"

// Nothing to define — all sensors_lib public API is either:
//   a) implemented in rplidar_a1.cpp  (RPLidarA1 driver)
//   b) implemented in ld06.cpp        (LD06 driver)
//   c) defined inline in sensors.h   (make_lidar, default_port, model_name)
//
// This translation unit intentionally left without additional definitions.