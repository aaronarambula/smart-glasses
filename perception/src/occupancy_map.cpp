// ─── occupancy_map.cpp ───────────────────────────────────────────────────────
// Implementation of the rolling 2-D probabilistic occupancy grid.
// See include/perception/occupancy_map.h for the full design notes.

#include "perception/occupancy_map.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace perception {

// ─── Construction ─────────────────────────────────────────────────────────────

OccupancyMap::OccupancyMap()
{
    reset();
}

void OccupancyMap::reset()
{
    grid_.fill(0.0f);
    frames_processed_ = 0;
}

// ─── add_log_odds ─────────────────────────────────────────────────────────────
//
// Adds `delta` to cell (col, row), clamping to [LOG_MIN, LOG_MAX].
// Bounds-checked: silently ignores out-of-bounds indices.

inline void OccupancyMap::add_log_odds(int col, int row, float delta)
{
    if (col < 0 || col >= GRID_CELLS || row < 0 || row >= GRID_CELLS) return;
    float& lo = cell(col, row);
    lo = std::clamp(lo + delta, LOG_MIN, LOG_MAX);
}

// ─── decay ────────────────────────────────────────────────────────────────────
//
// Multiplies every cell's log-odds by DECAY_FACTOR.
// This implements exponential forgetting — cells that stop being observed
// gradually return toward zero (unknown) rather than staying permanently marked.
//
// We use a raw pointer loop here because iterating 160,000 floats with
// std::transform is measurably faster than index-based access on ARM.

void OccupancyMap::decay()
{
    float* ptr = grid_.data();
    const size_t n = grid_.size();
    for (size_t i = 0; i < n; ++i) {
        ptr[i] *= DECAY_FACTOR;
    }
}

// ─── ray_trace ────────────────────────────────────────────────────────────────
//
// Bresenham's line algorithm from (c0,r0) to (c1,r1).
//
// For every cell along the ray from origin to the endpoint:
//   - Intermediate cells → add LOG_FREE_HIT (negative: evidence of free space)
//   - Endpoint cell      → add LOG_OCC_HIT  (positive: evidence of occupancy)
//                          only when mark_endpoint == true
//
// Bresenham's algorithm traces a line with integer arithmetic only (no float
// per-step), which is both accurate and fast.

void OccupancyMap::ray_trace(int c0, int r0, int c1, int r1, bool mark_endpoint)
{
    // Bresenham setup.
    int dc = std::abs(c1 - c0);
    int dr = std::abs(r1 - r0);
    int sc = (c0 < c1) ? 1 : -1;
    int sr = (r0 < r1) ? 1 : -1;

    int err = dc - dr;

    int c = c0;
    int r = r0;

    while (true) {
        bool at_endpoint = (c == c1 && r == r1);

        if (at_endpoint) {
            if (mark_endpoint) {
                add_log_odds(c, r, LOG_OCC_HIT);
            }
            break;
        }

        // Intermediate cell — mark as free space.
        // Skip the origin cell itself (c0,r0) — that's always free (user stands there).
        if (!(c == c0 && r == r0)) {
            add_log_odds(c, r, LOG_FREE_HIT);
        }

        // Bresenham step.
        int e2 = 2 * err;
        if (e2 > -dr) { err -= dr; c += sc; }
        if (e2 <  dc) { err += dc; r += sr; }
    }
}

// ─── update ───────────────────────────────────────────────────────────────────
//
// Full per-frame update:
//   1. Decay all cells.
//   2. For each valid ScanPoint:
//        a. Convert to Cartesian.
//        b. Convert to grid cell indices.
//        c. Ray-trace from origin to the endpoint.
//           - If distance <= max_range_mm: mark endpoint as occupied.
//           - If distance >  max_range_mm: only mark free space along the ray
//             (distant returns are often specular wall reflections — we trust
//             the free-space information but not the endpoint occupancy).
//   3. Clamping is handled per-cell inside add_log_odds().

void OccupancyMap::update(const sensors::ScanFrame& frame, float max_range_mm)
{
    decay();

    for (const auto& pt : frame.points) {
        if (!pt.is_valid()) continue;

        // Convert polar → Cartesian.
        float x_mm, y_mm;
        pt.to_cartesian(x_mm, y_mm);

        // Convert to grid cell.
        int c1, r1;
        if (!OccupancyGrid::world_to_cell(x_mm, y_mm, c1, r1)) continue;

        // Origin cell is always the grid centre.
        const int c0 = CENTRE;
        const int r0 = CENTRE;

        // Ray-trace. Mark endpoint only if within reliable range.
        bool mark_end = (pt.distance_mm <= max_range_mm);
        ray_trace(c0, r0, c1, r1, mark_end);
    }

    ++frames_processed_;
}

// ─── log_odds ─────────────────────────────────────────────────────────────────

float OccupancyMap::log_odds(int col, int row) const
{
    if (col < 0 || col >= GRID_CELLS || row < 0 || row >= GRID_CELLS) return 0.0f;
    return cell(col, row);
}

// ─── probability_at ───────────────────────────────────────────────────────────

float OccupancyMap::probability_at(float x_mm, float y_mm) const
{
    int col, row;
    if (!OccupancyGrid::world_to_cell(x_mm, y_mm, col, row)) return 0.5f;
    float lo = cell(col, row);
    return 1.0f / (1.0f + std::exp(-lo));
}

// ─── is_occupied_at ───────────────────────────────────────────────────────────

bool OccupancyMap::is_occupied_at(float x_mm, float y_mm) const
{
    int col, row;
    if (!OccupancyGrid::world_to_cell(x_mm, y_mm, col, row)) return false;
    return cell(col, row) > 0.5f;
}

// ─── local_density ────────────────────────────────────────────────────────────
//
// Returns the fraction of cells within radius_mm of the origin that have
// positive log-odds (i.e. are more likely occupied than free).
// Used by the risk predictor as a "how cluttered is the local environment" feature.

float OccupancyMap::local_density(float radius_mm) const
{
    const int r_cells = static_cast<int>(radius_mm / CELL_MM) + 1;
    int total = 0;
    int occupied = 0;

    for (int dr = -r_cells; dr <= r_cells; ++dr) {
        for (int dc = -r_cells; dc <= r_cells; ++dc) {
            // Check if this cell is within the circular radius.
            float dist = std::sqrt(static_cast<float>(dr*dr + dc*dc)) * CELL_MM;
            if (dist > radius_mm) continue;

            int col = CENTRE + dc;
            int row = CENTRE + dr;
            if (col < 0 || col >= GRID_CELLS || row < 0 || row >= GRID_CELLS) continue;

            ++total;
            if (cell(col, row) > 0.0f) ++occupied;
        }
    }

    return (total > 0) ? static_cast<float>(occupied) / total : 0.0f;
}

// ─── closest_occupied_mm ─────────────────────────────────────────────────────
//
// Scans a radial sweep from the origin through the given angular sector,
// stepping outward in cell-sized increments, and returns the distance (mm)
// to the first occupied cell. Returns -1 if none found within max_range_mm.
//
// This is used by the risk predictor to produce per-sector distance features
// directly from the map (complementing the raw scan data from the clusterer).

float OccupancyMap::closest_occupied_mm(float angle_lo_deg, float angle_hi_deg,
                                         float max_range_mm) const
{
    const int max_steps = static_cast<int>(max_range_mm / CELL_MM) + 1;

    // Sample angular resolution: 1 degree per ray.
    const float angle_step = 1.0f;

    float best_dist = -1.0f;

    for (float ang = angle_lo_deg; ang <= angle_hi_deg; ang += angle_step) {
        const float rad = ang * (static_cast<float>(M_PI) / 180.0f);
        const float cos_a = std::cos(rad);
        const float sin_a = std::sin(rad);

        for (int step = 1; step <= max_steps; ++step) {
            float x_mm =  cos_a * step * CELL_MM;
            float y_mm = -sin_a * step * CELL_MM;   // -sin: +Y = left, angle clockwise

            int col, row;
            if (!OccupancyGrid::world_to_cell(x_mm, y_mm, col, row)) break;

            if (cell(col, row) > 0.5f) {
                float dist = step * CELL_MM;
                if (best_dist < 0.0f || dist < best_dist) {
                    best_dist = dist;
                }
                break;  // found first hit along this ray
            }
        }
    }

    return best_dist;
}

// ─── get_grid_copy ────────────────────────────────────────────────────────────

OccupancyGrid OccupancyMap::get_grid_copy() const
{
    OccupancyGrid g;
    g.data = grid_;   // std::array copy
    return g;
}

// ─── debug_ascii ──────────────────────────────────────────────────────────────
//
// Renders the grid as ASCII art to a string for terminal debugging.
// stride=8 means every 8th cell is sampled → 50×50 characters for a 400×400 grid.
// Legend: '#' = occupied, '.' = free, ' ' = unknown

std::string OccupancyMap::debug_ascii(int stride) const
{
    std::ostringstream ss;
    const int display_size = GRID_CELLS / stride;

    ss << "OccupancyMap [" << display_size << "x" << display_size << "] "
       << "(" << static_cast<int>(GRID_MM / 1000) << "m x "
       << static_cast<int>(GRID_MM / 1000) << "m)\n";

    // Print top-to-bottom: high row index = +Y = left side of display.
    for (int r = display_size - 1; r >= 0; --r) {
        for (int c = 0; c < display_size; ++c) {
            int grid_col = c * stride;
            int grid_row = r * stride;
            float lo = cell(grid_col, grid_row);

            // Mark the user position (centre) with '@'.
            bool is_centre = (std::abs(grid_col - CENTRE) < stride) &&
                             (std::abs(grid_row - CENTRE) < stride);
            if (is_centre) {
                ss << '@';
            } else if (lo > 0.5f) {
                ss << '#';
            } else if (lo < -0.2f) {
                ss << '.';
            } else {
                ss << ' ';
            }
        }
        ss << '\n';
    }

    ss << "Frames processed: " << frames_processed_ << '\n';
    return ss.str();
}

} // namespace perception