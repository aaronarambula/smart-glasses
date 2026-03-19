// ─── clusterer.cpp ───────────────────────────────────────────────────────────
// Full DBSCAN implementation for 2-D LiDAR point clustering.
// See include/perception/clusterer.h for the full design notes.

#include "perception/clusterer.h"

#include <cmath>
#include <cstring>
#include <queue>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cassert>

namespace perception {

// ─── Cluster::str ─────────────────────────────────────────────────────────────

std::string Cluster::str() const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0);
    ss << "Cluster{"
       << "centroid=(" << centroid_x << ", " << centroid_y << ") mm"
       << " dist=" << distance_mm << " mm"
       << " bearing=" << std::setprecision(1) << bearing_deg << "°"
       << std::setprecision(0)
       << " size=" << size_mm() << " mm (" << size_label() << ")"
       << " pts=" << point_count()
       << "}";
    return ss.str();
}

// ─── to_cartesian ─────────────────────────────────────────────────────────────
//
// Converts all valid ScanPoints in the frame to Cartesian coordinates,
// applying quality and range filters.

std::vector<CartesianPoint> Clusterer::to_cartesian(
    const sensors::ScanFrame& frame) const
{
    std::vector<CartesianPoint> pts;
    pts.reserve(frame.size());

    for (const auto& sp : frame.points) {
        // Quality filter
        if (!sp.is_valid())                    continue;
        if (sp.quality     < min_quality_)     continue;
        if (sp.distance_mm < min_dist_mm_)     continue;
        if (sp.distance_mm > max_dist_mm_)     continue;

        CartesianPoint cp;
        sp.to_cartesian(cp.x_mm, cp.y_mm);
        cp.distance_mm = sp.distance_mm;
        cp.angle_deg   = sp.angle_deg;
        cp.quality     = sp.quality;
        pts.push_back(cp);
    }

    return pts;
}

// ─── region_query ─────────────────────────────────────────────────────────────
//
// Returns the indices of all points within eps of pts[idx].
// Uses the pre-computed eps_sq_ to avoid sqrt in the inner loop.

std::vector<int> Clusterer::region_query(
    const std::vector<CartesianPoint>& pts,
    int idx) const
{
    std::vector<int> neighbours;
    const CartesianPoint& p = pts[static_cast<size_t>(idx)];

    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if (i == idx) continue;
        if (p.dist_sq(pts[static_cast<size_t>(i)]) <= eps_sq_) {
            neighbours.push_back(i);
        }
    }

    return neighbours;
}

// ─── expand_cluster ───────────────────────────────────────────────────────────
//
// BFS expansion of a cluster starting from seed_idx.
// All density-reachable points are labelled with cluster_id.
//
// `neighbours` is the pre-computed neighbourhood for every point (avoids
// recomputing inside the BFS queue). This is the key optimisation that keeps
// the algorithm fast: we compute all neighbourhoods once in dbscan(), then
// expand_cluster just does queue operations.

void Clusterer::expand_cluster(
    [[maybe_unused]] const std::vector<CartesianPoint>& pts,
    std::vector<int>& labels,
    int seed_idx,
    int cluster_id,
    const std::vector<std::vector<int>>& neighbours) const
{
    std::queue<int> q;
    q.push(seed_idx);

    while (!q.empty()) {
        int idx = q.front();
        q.pop();

        // If this point was previously labelled NOISE, absorb it into the cluster.
        // If already labelled with this or another cluster, skip.
        if (labels[static_cast<size_t>(idx)] != -1 &&
            labels[static_cast<size_t>(idx)] != 0) {
            // Already assigned to a cluster — skip.
            continue;
        }

        // Assign to this cluster.
        labels[static_cast<size_t>(idx)] = cluster_id;

        // If this point is a core point (has enough neighbours), expand further.
        const auto& nbrs = neighbours[static_cast<size_t>(idx)];
        if (static_cast<int>(nbrs.size()) >= min_pts_ - 1) {
            // -1 because nbrs doesn't include idx itself
            for (int n : nbrs) {
                if (labels[static_cast<size_t>(n)] <= 0) {
                    // Unvisited or noise — add to queue.
                    q.push(n);
                }
            }
        }
    }
}

// ─── build_cluster ────────────────────────────────────────────────────────────
//
// Collects all points assigned to cluster_id, computes centroid, bounding box,
// width, depth, distance to origin, and bearing.

Cluster Clusterer::build_cluster(
    const std::vector<CartesianPoint>& pts,
    const std::vector<int>& labels,
    int cluster_id)
{
    Cluster c;

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (size_t i = 0; i < pts.size(); ++i) {
        if (labels[i] != cluster_id) continue;

        const auto& p = pts[i];
        c.points.push_back(p);
        sum_x += p.x_mm;
        sum_y += p.y_mm;
        c.bbox.expand(p.x_mm, p.y_mm);
    }

    if (c.points.empty()) return c;

    const float n = static_cast<float>(c.points.size());
    c.centroid_x = sum_x / n;
    c.centroid_y = sum_y / n;

    c.width_mm = c.bbox.width();
    c.depth_mm = c.bbox.height();

    // Distance from origin (user) to centroid.
    c.distance_mm = std::sqrt(c.centroid_x * c.centroid_x +
                               c.centroid_y * c.centroid_y);

    // Bearing: angle from +X axis (forward), clockwise positive.
    // atan2 returns CCW from +X in standard maths, so we negate Y.
    float bearing_rad = std::atan2(-c.centroid_y, c.centroid_x);
    float bearing_deg = bearing_rad * (180.0f / static_cast<float>(M_PI));
    if (bearing_deg < 0.0f) bearing_deg += 360.0f;
    c.bearing_deg = bearing_deg;

    return c;
}

// ─── dbscan ───────────────────────────────────────────────────────────────────
//
// Core DBSCAN implementation.
//
// Label conventions:
//   -1  : UNVISITED (initial state)
//    0  : NOISE (visited but not part of any cluster)
//   > 0 : cluster ID
//
// Two-pass approach:
//   Pass 1: compute all neighbourhoods and identify core points.
//   Pass 2: BFS-expand each unvisited core point into a cluster.
//
// This avoids recomputing neighbourhood queries inside expand_cluster.

ClusterList Clusterer::dbscan(const std::vector<CartesianPoint>& pts) const
{
    const int N = static_cast<int>(pts.size());
    if (N == 0) return {};

    // Label array: -1 = unvisited, 0 = noise, >0 = cluster id.
    std::vector<int> labels(static_cast<size_t>(N), -1);

    // Pre-compute all neighbourhoods — O(N²) but N ≤ 460, so ~200k ops.
    std::vector<std::vector<int>> neighbours(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        neighbours[static_cast<size_t>(i)] = region_query(pts, i);
    }

    int cluster_id = 0;

    for (int i = 0; i < N; ++i) {
        // Skip already-labelled points.
        if (labels[static_cast<size_t>(i)] != -1) continue;

        const auto& nbrs = neighbours[static_cast<size_t>(i)];

        // Not a core point → noise (may be absorbed later by another core point).
        if (static_cast<int>(nbrs.size()) < min_pts_ - 1) {
            labels[static_cast<size_t>(i)] = 0;  // noise
            continue;
        }

        // Core point → start a new cluster.
        ++cluster_id;
        labels[static_cast<size_t>(i)] = cluster_id;
        expand_cluster(pts, labels, i, cluster_id, neighbours);
    }

    // ── Build Cluster objects ─────────────────────────────────────────────────

    ClusterList result;
    result.reserve(static_cast<size_t>(cluster_id));

    for (int cid = 1; cid <= cluster_id; ++cid) {
        Cluster c = build_cluster(pts, labels, cid);
        if (!c.points.empty()) {
            result.push_back(std::move(c));
        }
    }

    // Sort by distance to origin, closest first.
    std::sort(result.begin(), result.end(),
              [](const Cluster& a, const Cluster& b) {
                  return a.distance_mm < b.distance_mm;
              });

    return result;
}

// ─── run (from ScanFrame) ─────────────────────────────────────────────────────

ClusterList Clusterer::run(const sensors::ScanFrame& frame) const
{
    auto pts = to_cartesian(frame);
    return dbscan(pts);
}

// ─── run (from CartesianPoint list) ──────────────────────────────────────────

ClusterList Clusterer::run(const std::vector<CartesianPoint>& pts) const
{
    return dbscan(pts);
}

} // namespace perception