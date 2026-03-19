#pragma once

// ─── clusterer.h ─────────────────────────────────────────────────────────────
// DBSCAN-based point clusterer for 2-D LiDAR scan data.
//
// Design
// ──────
// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is the
// right algorithm here for three reasons:
//   1. It doesn't require knowing the number of clusters in advance.
//   2. It handles arbitrary cluster shapes (people, walls, chairs are all
//      irregular in polar scan data).
//   3. Noise points (isolated reflections) are naturally rejected — they never
//      join any cluster.
//
// Algorithm recap
// ───────────────
//   For each point P:
//     If P is already labelled, skip.
//     Find all points within eps_mm of P  (neighbourhood).
//     If |neighbourhood| < min_pts  →  label P as NOISE (temporary).
//     Else  →  start a new cluster:
//       Add P and all reachable density-connected points to the cluster.
//       A point Q is density-reachable from P if there exists a chain of
//       core points connecting them, each within eps_mm of the next.
//
// Complexity: O(N²) in the worst case, but N ≤ 460 points per LD06 frame,
// so the naive implementation is fast enough (< 0.5 ms on a Pi 4).
//
// Coordinate frame
// ────────────────
//   Same as the occupancy map:
//     +X = forward, +Y = left, origin = user, all units in millimetres.
//
// Output types
// ────────────
//   BoundingBox  — axis-aligned rectangle enclosing all cluster points
//   Cluster      — centroid, bounding box, point list, estimated width/depth
//   ClusterList  — vector of Cluster, sorted by distance to origin (closest first)
//
// Parameters
// ──────────
//   eps_mm   : neighbourhood radius. 150 mm works well for pedestrian-scale
//              objects. Reduce to 80 mm for tighter separation in dense scenes.
//   min_pts  : minimum points to form a core point. 3 is standard for 2-D data.
//
// Usage
// ─────
//   Clusterer clusterer;                  // default eps=150mm, minPts=3
//   auto clusters = clusterer.run(frame); // one call per ScanFrame
//   for (auto& c : clusters) {
//       printf("obstacle at (%.0f, %.0f) mm, width=%.0f mm\n",
//              c.centroid_x, c.centroid_y, c.width_mm);
//   }

#include "sensors/lidar_base.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

namespace perception {

// ─── BoundingBox ──────────────────────────────────────────────────────────────
//
// Axis-aligned bounding box in world coordinates (mm).

struct BoundingBox {
    float x_min =  std::numeric_limits<float>::max();
    float x_max = -std::numeric_limits<float>::max();
    float y_min =  std::numeric_limits<float>::max();
    float y_max = -std::numeric_limits<float>::max();

    void expand(float x, float y) {
        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
    }

    float width()  const { return x_max - x_min; }   // extent along X (forward)
    float height() const { return y_max - y_min; }   // extent along Y (left)
    float area()   const { return width() * height(); }

    float centre_x() const { return 0.5f * (x_min + x_max); }
    float centre_y() const { return 0.5f * (y_min + y_max); }

    bool valid() const { return x_min <= x_max && y_min <= y_max; }
};

// ─── CartesianPoint ───────────────────────────────────────────────────────────
//
// A single scan point converted to Cartesian coordinates.
// Carries the original polar data for traceability.

struct CartesianPoint {
    float   x_mm        = 0.0f;
    float   y_mm        = 0.0f;
    float   distance_mm = 0.0f;   // original polar distance
    float   angle_deg   = 0.0f;   // original polar angle
    uint8_t quality     = 0;

    // Squared Euclidean distance to another point (avoids sqrt in DBSCAN inner loop).
    float dist_sq(const CartesianPoint& o) const {
        float dx = x_mm - o.x_mm;
        float dy = y_mm - o.y_mm;
        return dx*dx + dy*dy;
    }

    float dist(const CartesianPoint& o) const {
        return std::sqrt(dist_sq(o));
    }
};

// ─── Cluster ──────────────────────────────────────────────────────────────────
//
// One detected obstacle cluster.

struct Cluster {
    // ── Geometry ──────────────────────────────────────────────────────────────

    float centroid_x  = 0.0f;   // mean X of all member points (mm)
    float centroid_y  = 0.0f;   // mean Y of all member points (mm)
    float width_mm    = 0.0f;   // bounding box width  (forward extent)
    float depth_mm    = 0.0f;   // bounding box height (lateral extent)
    BoundingBox bbox;

    // ── Distance / bearing to user ────────────────────────────────────────────

    // Euclidean distance from origin (user) to centroid.
    float distance_mm = 0.0f;

    // Bearing from user to centroid, degrees [0, 360). 0° = forward.
    float bearing_deg = 0.0f;

    // ── Point membership ──────────────────────────────────────────────────────

    // All Cartesian points that belong to this cluster.
    std::vector<CartesianPoint> points;

    int point_count() const { return static_cast<int>(points.size()); }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Estimated linear size of the obstacle in mm (max of width, depth).
    float size_mm() const { return std::max(width_mm, depth_mm); }

    // Rough classification based on size:
    //   < 300 mm  → "small"   (post, leg, narrow obstacle)
    //   < 800 mm  → "medium"  (person, chair)
    //   < 2000 mm → "large"   (car, wall segment)
    //   ≥ 2000 mm → "wall"
    const char* size_label() const {
        float s = size_mm();
        if (s < 300.0f)  return "small";
        if (s < 800.0f)  return "medium";
        if (s < 2000.0f) return "large";
        return "wall";
    }

    // True if the cluster is in the forward cone (bearing within half_deg of 0°).
    bool is_forward(float half_deg = 30.0f) const {
        float b = bearing_deg;
        return b <= half_deg || b >= (360.0f - half_deg);
    }

    // Human-readable one-liner for logging.
    std::string str() const;
};

// Sorted by distance to user ascending (closest first).
using ClusterList = std::vector<Cluster>;

// ─── Clusterer ────────────────────────────────────────────────────────────────
//
// Stateless DBSCAN clusterer.  Instantiate once and call run() per frame.
// The object itself holds no per-frame state — all state lives in run().

class Clusterer {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // eps_mm   : neighbourhood radius in mm (default 150 mm)
    // min_pts  : minimum neighbours to be a core point (default 3)
    // min_quality : ScanPoints with quality below this are skipped
    // min_dist_mm : ignore points closer than this (sensor housing noise)
    // max_dist_mm : ignore points beyond this range
    explicit Clusterer(float   eps_mm      = 150.0f,
                       int     min_pts     = 3,
                       uint8_t min_quality = 10,
                       float   min_dist_mm = 50.0f,
                       float   max_dist_mm = 6000.0f)
        : eps_mm_(eps_mm)
        , eps_sq_(eps_mm * eps_mm)
        , min_pts_(min_pts)
        , min_quality_(min_quality)
        , min_dist_mm_(min_dist_mm)
        , max_dist_mm_(max_dist_mm)
    {}

    // ── Main entry point ──────────────────────────────────────────────────────

    // Runs DBSCAN on the valid points of `frame`.
    // Returns clusters sorted by distance to origin, closest first.
    // Noise points (isolated returns) are silently discarded.
    ClusterList run(const sensors::ScanFrame& frame) const;

    // Run on a pre-built list of Cartesian points (for testing / simulation).
    ClusterList run(const std::vector<CartesianPoint>& pts) const;

    // ── Parameter access ──────────────────────────────────────────────────────

    float   eps_mm()      const { return eps_mm_; }
    int     min_pts()     const { return min_pts_; }
    void    set_eps_mm(float e) { eps_mm_ = e; eps_sq_ = e * e; }
    void    set_min_pts(int m)  { min_pts_ = m; }

private:
    float   eps_mm_;
    float   eps_sq_;        // eps² — avoids sqrt in inner loop
    int     min_pts_;
    uint8_t min_quality_;
    float   min_dist_mm_;
    float   max_dist_mm_;

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Convert valid ScanPoints in a frame to Cartesian coordinates.
    std::vector<CartesianPoint> to_cartesian(
        const sensors::ScanFrame& frame) const;

    // Core DBSCAN implementation — shared by both run() overloads.
    ClusterList dbscan(const std::vector<CartesianPoint>& pts) const;

    // Returns indices of all points within eps of pts[idx].
    std::vector<int> region_query(const std::vector<CartesianPoint>& pts,
                                  int idx) const;

    // Expands cluster `cluster_id` by BFS from seed point `seed_idx`.
    // Mutates `labels` in-place.
    void expand_cluster(const std::vector<CartesianPoint>& pts,
                        std::vector<int>& labels,
                        int seed_idx,
                        int cluster_id,
                        const std::vector<std::vector<int>>& neighbours) const;

    // Builds a Cluster value from the points assigned label `cluster_id`.
    static Cluster build_cluster(const std::vector<CartesianPoint>& pts,
                                 const std::vector<int>& labels,
                                 int cluster_id);
};

} // namespace perception