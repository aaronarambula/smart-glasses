#pragma once

// ─── perception.h ────────────────────────────────────────────────────────────
// Umbrella header for the perception module.
//
// Include this single file to pull in:
//   - OccupancyMap  : rolling 2-D probabilistic occupancy grid
//   - Clusterer     : DBSCAN point clusterer  → ClusterList
//   - Tracker       : Kalman multi-object tracker → TrackedObjectList
//
// Typical usage in the pipeline:
//
//   #include "perception/perception.h"
//   #include "sensors/sensors.h"
//
//   perception::OccupancyMap  map;
//   perception::Clusterer     clusterer;
//   perception::Tracker       tracker;
//
//   // Called once per ScanFrame (10 Hz):
//   void on_frame(const sensors::ScanFrame& frame, float dt_s) {
//       map.update(frame);
//       auto clusters = clusterer.run(frame);
//       auto objects  = tracker.update(clusters, dt_s);
//       // objects → TTC engine → risk predictor → audio alerts
//   }
//
// Dependency graph (no cycles):
//
//   sensors/lidar_base.h
//          │
//          ▼
//   occupancy_map.h   clusterer.h
//                          │
//                          ▼
//                      tracker.h
//                          │
//                          ▼
//                    perception.h  ← this file

#include "occupancy_map.h"
#include "clusterer.h"
#include "tracker.h"

namespace perception {

// ─── PerceptionResult ────────────────────────────────────────────────────────
//
// Aggregated output of one full perception pass on a single ScanFrame.
// Produced by PerceptionPipeline::process() and consumed by the prediction
// module (TTC engine + risk predictor MLP).
//
// Owns no heap memory beyond what the vectors carry — safe to copy.

struct PerceptionResult {
    uint64_t          frame_id   = 0;
    float             dt_s       = 0.0f;        // elapsed time since last frame

    ClusterList       clusters;                 // raw DBSCAN clusters this frame
    TrackedObjectList objects;                  // Kalman-tracked objects

    // Grid snapshot — value copy, safe to pass to other threads.
    OccupancyGrid     grid;

    // Convenience accessors ───────────────────────────────────────────────────

    // Number of confirmed tracked objects.
    int confirmed_count() const {
        int n = 0;
        for (const auto& o : objects)
            if (o.is_confirmed()) ++n;
        return n;
    }

    // Closest confirmed object (by distance), or nullptr if none.
    const TrackedObject* closest_confirmed() const {
        const TrackedObject* best = nullptr;
        float best_dist = std::numeric_limits<float>::max();
        for (const auto& o : objects) {
            if (o.is_confirmed() && o.distance_mm < best_dist) {
                best_dist = o.distance_mm;
                best = &o;
            }
        }
        return best;
    }

    // Closest confirmed object in the forward cone (±half_deg of 0°).
    const TrackedObject* closest_forward(float half_deg = 30.0f) const {
        const TrackedObject* best = nullptr;
        float best_dist = std::numeric_limits<float>::max();
        for (const auto& o : objects) {
            if (!o.is_confirmed()) continue;
            float b = o.bearing_deg;
            bool fwd = (b <= half_deg) || (b >= 360.0f - half_deg);
            if (fwd && o.distance_mm < best_dist) {
                best_dist = o.distance_mm;
                best = &o;
            }
        }
        return best;
    }

    // True if any confirmed object is approaching within danger_mm.
    bool has_imminent_threat(float danger_mm = 800.0f) const {
        for (const auto& o : objects) {
            if (o.is_confirmed() &&
                o.is_approaching() &&
                o.distance_mm < danger_mm)
                return true;
        }
        return false;
    }
};

// ─── PerceptionPipeline ───────────────────────────────────────────────────────
//
// Convenience wrapper that owns and sequences all three perception components:
//   OccupancyMap → Clusterer → Tracker
//
// This is the single object the rest of the system instantiates. Calling
// process() once per ScanFrame drives the whole perception stack and returns
// a self-contained PerceptionResult.
//
// Owns all internal state (map, clusterer, tracker). Non-copyable.

class PerceptionPipeline {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // All parameters take sensible defaults for the LD06 at 10 Hz.
    // eps_mm   : DBSCAN neighbourhood radius (mm)
    // min_pts  : DBSCAN minimum cluster size
    explicit PerceptionPipeline(float   eps_mm  = 150.0f,
                                int     min_pts = 3)
        : clusterer_(eps_mm, min_pts)
    {}

    // ── Main entry point ──────────────────────────────────────────────────────

    // Process one ScanFrame through the full perception stack.
    //
    //   dt_s : elapsed wall-clock seconds since the previous frame.
    //          The caller should measure this with steady_clock.
    //          Pass 0.0f on the very first call.
    //
    // Returns a PerceptionResult containing clusters, tracked objects, and a
    // grid snapshot. The result is a value type — safe to move to another thread.
    PerceptionResult process(const sensors::ScanFrame& frame, float dt_s) {
        PerceptionResult result;
        result.frame_id = frame.frame_id;
        result.dt_s     = dt_s;

        // 1. Update occupancy map (decay + ray-trace).
        map_.update(frame);

        // 2. DBSCAN cluster the valid scan points.
        result.clusters = clusterer_.run(frame);

        // 3. Kalman-track the clusters across frames.
        result.objects = tracker_.update(result.clusters, dt_s);

        // 4. Snapshot the grid (value copy, cheap — 160 KB memcpy).
        result.grid = map_.get_grid_copy();

        return result;
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    // Clears all perception state (map + tracker). Call on pipeline restart.
    void reset() {
        map_.reset();
        tracker_.reset();
    }

    // ── Component access (for tuning / debugging) ─────────────────────────────

    OccupancyMap&       map()       { return map_; }
    const OccupancyMap& map() const { return map_; }

    Clusterer&          clusterer()       { return clusterer_; }
    const Clusterer&    clusterer() const { return clusterer_; }

    Tracker&            tracker()       { return tracker_; }
    const Tracker&      tracker() const { return tracker_; }

private:
    OccupancyMap map_;
    Clusterer    clusterer_;
    Tracker      tracker_;
};

} // namespace perception