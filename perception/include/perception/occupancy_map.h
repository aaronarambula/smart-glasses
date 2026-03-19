#pragma once

// ─── occupancy_map.h ─────────────────────────────────────────────────────────
// Rolling 2-D probabilistic occupancy grid centred on the user.
//
// Design
// ──────
// The grid is a fixed-size square array of log-odds values. Each cell stores
// log( p / (1-p) ) where p is the probability that the cell is occupied.
// This representation lets us do Bayesian updates with simple addition instead
// of multiplication, which is both numerically stable and fast.
//
// Coordinate frame
// ────────────────
//   Origin  : user position (centre of the grid)
//   +X axis : forward  (0° bearing)
//   +Y axis : left     (90° bearing)
//   All distances in millimetres.
//
// Parameters (compile-time)
// ─────────────────────────
//   GRID_CELLS : number of cells along each axis (square grid)
//   CELL_MM    : physical size of one cell in mm
//   Total coverage = GRID_CELLS × CELL_MM   e.g. 400 × 25 = 10 000 mm = 10 m
//
// Per-frame update
// ────────────────
//   1. decay()       — multiply all log-odds by DECAY_FACTOR (exponential
//                      forgetting: old obstacles fade out as user moves)
//   2. update(frame) — for every valid ScanPoint:
//        • mark the endpoint cell as occupied  (+LOG_OCC_HIT)
//        • ray-trace from origin to endpoint,
//          marking intermediate cells as free  (+LOG_FREE_HIT, negative)
//   3. clamp all log-odds to [LOG_MIN, LOG_MAX] to prevent saturation
//
// Thread safety
// ─────────────
//   Not internally locked. The caller (pipeline thread) owns the grid
//   and must not call update() from multiple threads simultaneously.
//   get_grid_copy() returns a value copy that is safe to read from any thread.

#include "sensors/lidar_base.h"

#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

namespace perception {

// ─── Grid parameters ──────────────────────────────────────────────────────────

static constexpr int   GRID_CELLS   = 400;      // cells per axis
static constexpr float CELL_MM      = 25.0f;    // mm per cell
static constexpr float GRID_MM      = GRID_CELLS * CELL_MM;   // 10 000 mm = 10 m
static constexpr float HALF_GRID_MM = GRID_MM / 2.0f;         // 5 000 mm

// Log-odds update magnitudes.
static constexpr float LOG_OCC_HIT  =  0.85f;   // strong evidence of occupancy
static constexpr float LOG_FREE_HIT = -0.40f;   // weaker evidence of free space
                                                 // (asymmetric: trust hits more)

// Clamping bounds — prevent saturation at ±∞.
static constexpr float LOG_MAX      =  3.5f;    // ≈ 97% probability occupied
static constexpr float LOG_MIN      = -2.0f;    // ≈ 12% probability occupied

// Exponential decay factor applied once per scan frame.
// 0.85 means a cell that stopped being hit decays to ~10% after ~15 frames.
static constexpr float DECAY_FACTOR =  0.85f;

// ─── Cell ─────────────────────────────────────────────────────────────────────

// One grid cell.  Stored as a float log-odds value.
// Positive  → probably occupied.
// Negative  → probably free.
// Zero      → completely unknown (initial state).
using LogOdds = float;

// ─── OccupancyGrid ────────────────────────────────────────────────────────────
//
// Plain value type used for the grid snapshot returned by get_grid_copy().
// Row-major: cell(row, col) = data[row * GRID_CELLS + col]
// row increases in the +Y direction (left), col increases in +X (forward).

struct OccupancyGrid {
    std::array<LogOdds, GRID_CELLS * GRID_CELLS> data{};

    // Returns probability of occupancy for cell (row, col) in [0, 1].
    float probability(int row, int col) const {
        float lo = data[static_cast<size_t>(row) * GRID_CELLS + col];
        return 1.0f / (1.0f + std::exp(-lo));
    }

    // Returns true if the cell is considered occupied (log-odds > 0.5).
    bool is_occupied(int row, int col) const {
        return data[static_cast<size_t>(row) * GRID_CELLS + col] > 0.5f;
    }

    // Returns true if world coordinates (x_mm, y_mm) are within the grid.
    static bool in_bounds_mm(float x_mm, float y_mm) {
        return std::abs(x_mm) < HALF_GRID_MM &&
               std::abs(y_mm) < HALF_GRID_MM;
    }

    // Convert world (x_mm, y_mm) to (col, row) grid indices.
    // Returns false if the point is outside the grid.
    static bool world_to_cell(float x_mm, float y_mm,
                               int& col_out, int& row_out)
    {
        if (!in_bounds_mm(x_mm, y_mm)) return false;
        col_out = static_cast<int>((x_mm + HALF_GRID_MM) / CELL_MM);
        row_out = static_cast<int>((y_mm + HALF_GRID_MM) / CELL_MM);
        col_out = std::clamp(col_out, 0, GRID_CELLS - 1);
        row_out = std::clamp(row_out, 0, GRID_CELLS - 1);
        return true;
    }

    // Convert (col, row) back to world coordinates (centre of the cell).
    static void cell_to_world(int col, int row,
                               float& x_mm_out, float& y_mm_out)
    {
        x_mm_out = (col + 0.5f) * CELL_MM - HALF_GRID_MM;
        y_mm_out = (row + 0.5f) * CELL_MM - HALF_GRID_MM;
    }
};

// ─── OccupancyMap ─────────────────────────────────────────────────────────────

class OccupancyMap {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    OccupancyMap();

    // Reset all cells to log-odds = 0 (completely unknown).
    void reset();

    // ── Per-frame update ──────────────────────────────────────────────────────

    // Full update cycle for one scan frame:
    //   1. Decay all cells by DECAY_FACTOR.
    //   2. Ray-trace each valid ScanPoint — mark free space + endpoint.
    //   3. Clamp all log-odds to [LOG_MIN, LOG_MAX].
    //
    // max_range_mm: points beyond this range are used for free-space marking
    //              only (no occupied update at their endpoint), because long
    //              returns are often wall reflections or noise.
    void update(const sensors::ScanFrame& frame,
                float max_range_mm = 5000.0f);

    // ── Querying ──────────────────────────────────────────────────────────────

    // Returns the log-odds of cell (col, row). 0 if out of bounds.
    float log_odds(int col, int row) const;

    // Returns occupancy probability [0,1] of the cell containing (x_mm, y_mm).
    // Returns 0.5 (unknown) if outside the grid.
    float probability_at(float x_mm, float y_mm) const;

    // Returns true if the cell containing (x_mm, y_mm) is occupied.
    bool is_occupied_at(float x_mm, float y_mm) const;

    // Returns the fraction of cells within radius_mm of the origin that are
    // occupied.  Used by the risk predictor as a local density feature.
    float local_density(float radius_mm) const;

    // Returns the closest occupied cell in the forward sector
    // [angle_lo_deg, angle_hi_deg], up to max_range_mm.
    // Returns -1 if none found.
    float closest_occupied_mm(float angle_lo_deg, float angle_hi_deg,
                               float max_range_mm = 5000.0f) const;

    // ── Snapshot ──────────────────────────────────────────────────────────────

    // Returns a value copy of the internal grid.
    // Safe to pass to other threads — no reference to internal state.
    OccupancyGrid get_grid_copy() const;

    // ── Statistics ────────────────────────────────────────────────────────────

    uint64_t frames_processed() const { return frames_processed_; }

    // ASCII art dump for debugging (80-char wide, every Nth cell).
    std::string debug_ascii(int stride = 8) const;

private:
    // ── Internal grid storage ─────────────────────────────────────────────────

    // Flat array, row-major. Index = row * GRID_CELLS + col.
    std::array<LogOdds, GRID_CELLS * GRID_CELLS> grid_{};

    // Centre cell indices (always GRID_CELLS/2).
    static constexpr int CENTRE = GRID_CELLS / 2;

    uint64_t frames_processed_ = 0;

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Apply DECAY_FACTOR to every cell.
    void decay();

    // Bresenham ray-trace from (c0,r0) to (c1,r1).
    // Marks all intermediate cells with LOG_FREE_HIT.
    // Marks the endpoint (c1,r1) with LOG_OCC_HIT (if mark_endpoint=true).
    // Clamps each touched cell to [LOG_MIN, LOG_MAX].
    void ray_trace(int c0, int r0, int c1, int r1, bool mark_endpoint);

    // Safe log-odds update for cell (col, row) — bounds-checks and clamps.
    inline void add_log_odds(int col, int row, float delta);

    // Returns reference to cell — no bounds check, caller must validate.
    inline LogOdds& cell(int col, int row) {
        return grid_[static_cast<size_t>(row) * GRID_CELLS + col];
    }
    inline const LogOdds& cell(int col, int row) const {
        return grid_[static_cast<size_t>(row) * GRID_CELLS + col];
    }
};

} // namespace perception