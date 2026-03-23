#pragma once

// ─── sim.h ───────────────────────────────────────────────────────────────────
// Umbrella header for the simulation module.
//
// Include this INSTEAD of "sensors/sensors.h" when you want to use the
// simulated sensor. It re-exports everything from sensors/sensors.h and
// additionally provides:
//
//   sim::SimLidar       — drop-in LidarBase implementation driven by SimWorld
//   sim::SimWorld       — outdoor environment with realistic noise model
//   sim::SimConfig      — tunable scan rate, noise, seed, user motion
//   sim::SceneId        — enum of built-in outdoor scenes
//   sim::make_sim_lidar — factory matching make_lidar() signature exactly
//
// Integration with main.cpp
// ─────────────────────────
// main.cpp uses make_lidar() from sensors/sensors.h and holds a
// unique_ptr<LidarBase>. To switch to simulation you have TWO options:
//
//   OPTION A — Compile-time flag (recommended for CI / development):
//     Pass --sensor sim to the binary:
//       ./smart_glasses --sensor sim --scene sidewalk --verbose
//     main.cpp detects "sim" in the --sensor argument and calls
//     sim::make_sim_lidar() instead of sensors::make_lidar().
//     No other code changes needed.
//
//   OPTION B — Direct construction (for unit tests):
//     #include "sim/sim.h"
//     auto lidar = sim::make_sim_lidar("sim://crossing");
//     lidar->open();
//     lidar->start();
//     // feed into the same perception/prediction pipeline
//
// Isolation guarantee
// ───────────────────
// The sim module ONLY includes:
//   sensors/lidar_base.h    — for LidarBase (bottom of the dep stack)
//   sim/sim_world.h         — geometry + noise model (no project deps)
//   sim/sim_lidar.h         — SimLidar driver
//   Standard library headers only
//
// It does NOT include perception/, prediction/, audio/, or agent/.
// Adding #include "sim/sim.h" to a file that already includes
// "sensors/sensors.h" is safe — all guards are #pragma once.
//
// Scene reference
// ───────────────
//   "sim://sidewalk"         Default. Suburban sidewalk, building wall on
//                            the left, parked cars on the right, two
//                            pedestrians walking in various directions,
//                            utility poles, a bench. User walks forward
//                            at 1.2 m/s.
//
//   "sim://crossing"         User approaches an intersection. A pedestrian
//                            crosses the path from right to left at 1.4 m/s.
//                            A car is parked on the far side. Curb lines
//                            on both sides. Tests TTC prediction accuracy.
//
//   "sim://hallway"          Narrow corridor (1.4 m wide), walls on both
//                            sides. A person is walking toward the user from
//                            10 m ahead. Tests narrow-passage WARNING/DANGER
//                            escalation with walls on both flanks.
//
//   "sim://parking_lot"      Open space, multiple parked cars in a grid.
//                            A slow-moving car (0.5 m/s) crosses the path
//                            from the right. Tests rectangular obstacle
//                            detection and CPA computation.
//
//   "sim://cyclist_overtake" User on a shared path. A cyclist approaches
//                            from behind-left at 6 m/s and overtakes.
//                            Tests rear-sector detection and fast-moving
//                            object tracking.
//
//   "sim://crowd"            Busy street: 6 pedestrians moving in different
//                            directions, two storefronts (glass — high
//                            dropout), a bus stop shelter. Tests the
//                            tracker's Hungarian assignment under high
//                            object-count conditions.
//
// Noise model summary (LD06-accurate)
// ─────────────────────────────────────
//   Range noise     : Gaussian σ=15mm at 1m, scales as √distance
//   Angular jitter  : Gaussian σ=0.1° per measurement
//   Dropout         : 2% base; up to 40% at grazing angles (<20°)
//   Glass surfaces  : 12× dropout multiplier (storefronts, car windows)
//   Foliage         : 5× dropout multiplier (trees, bushes)
//   Specular ghost  : 1% chance of a spurious 4–6m return (wet pavement,
//                     chrome, reflective signs)
//   Occlusion       : physically correct — ray stops at first hit
//   Quality falloff : decreases with distance and incidence angle

#include "sensors/sensors.h"
#include "sim/sim_world.h"
#include "sim/sim_lidar.h"

#include <memory>
#include <string>

namespace sim {

// ─── make_sim_lidar ──────────────────────────────────────────────────────────
//
// Factory function matching the make_lidar() signature from sensors/sensors.h.
// Returns a unique_ptr<LidarBase> wrapping a SimLidar.
//
// port_or_scene : "sim://sidewalk", "sim://hallway", etc.
//                 Any string not prefixed with "sim://" is treated as the
//                 sidewalk scene.
// config        : optional SimConfig for noise, scan rate, seed, etc.
//
// Example:
//   auto lidar = sim::make_sim_lidar("sim://crowd");
//   lidar->open();
//   lidar->start();
//   // identical to using make_lidar(LidarModel::LD06, "/dev/ttyAMA0")

inline std::unique_ptr<sensors::LidarBase>
make_sim_lidar(const std::string& port_or_scene = "sim://sidewalk",
               SimConfig          config         = SimConfig{})
{
    return std::make_unique<SimLidar>(port_or_scene, std::move(config));
}

// ─── scene_description ───────────────────────────────────────────────────────
//
// Returns a human-readable one-line description of each built-in scene.
// Used by main.cpp to print the scene info at startup.

inline const char* scene_description(SceneId id) {
    switch (id) {
        case SceneId::Sidewalk:
            return "Suburban sidewalk: building wall left, parked cars right, "
                   "2 pedestrians, utility poles, bench. User walks forward 1.2 m/s.";
        case SceneId::Crossing:
            return "Intersection approach: pedestrian crosses path right→left "
                   "at 1.4 m/s, parked car far side, curb lines both sides.";
        case SceneId::Hallway:
            return "Narrow corridor 1.4m wide: walls both sides, "
                   "pedestrian approaching head-on from 10m.";
        case SceneId::ParkingLot:
            return "Parking lot: grid of parked cars, slow vehicle (0.5 m/s) "
                   "crossing from the right.";
        case SceneId::CyclistOvertake:
            return "Shared path: cyclist overtakes from behind-left at 6 m/s.";
        case SceneId::Crowd:
            return "Busy street: 6 pedestrians, glass storefronts, bus stop shelter.";
        default:
            return "Unknown scene.";
    }
}

// ─── scene_name ──────────────────────────────────────────────────────────────
//
// Returns the canonical "sim://name" string for a SceneId.

inline const char* scene_name(SceneId id) {
    switch (id) {
        case SceneId::Sidewalk:         return "sim://sidewalk";
        case SceneId::Crossing:         return "sim://crossing";
        case SceneId::Hallway:          return "sim://hallway";
        case SceneId::ParkingLot:       return "sim://parking_lot";
        case SceneId::CyclistOvertake:  return "sim://cyclist_overtake";
        case SceneId::Crowd:            return "sim://crowd";
        default:                        return "sim://sidewalk";
    }
}

// ─── all_scenes ──────────────────────────────────────────────────────────────
//
// Returns all available scene IDs in a fixed array.
// Used by main.cpp to list scenes in --help output.

inline std::array<SceneId, 6> all_scenes() {
    return {
        SceneId::Sidewalk,
        SceneId::Crossing,
        SceneId::Hallway,
        SceneId::ParkingLot,
        SceneId::CyclistOvertake,
        SceneId::Crowd,
    };
}

} // namespace sim