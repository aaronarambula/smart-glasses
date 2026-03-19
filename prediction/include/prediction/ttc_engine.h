#pragma once

// ─── ttc_engine.h ─────────────────────────────────────────────────────────────
// Time-To-Collision (TTC) engine.
//
// Design
// ──────
// Given a list of Kalman-tracked objects (each with position px,py and
// velocity vx,vy in mm and mm/s), this engine computes for every object:
//
//   1. TTC  — seconds until the object reaches the user, assuming constant
//             velocity (linear extrapolation).  This is the most useful single
//             number for the audio alert system.
//
//   2. CPA  — Closest Point of Approach.  The minimum distance (mm) the object
//             will reach if neither party changes course, and the time (s) at
//             which that happens.  An object with TTC = ∞ (parallel path) may
//             still have a dangerously small CPA.
//
//   3. Projected path — a short sequence of (x,y) waypoints at fixed time
//             steps showing where the object is predicted to be over the next
//             HORIZON_S seconds.  Consumed by the scene builder for the OpenAI
//             agent context.
//
//   4. Sector threat table — one TTCResult per 45° sector summarising the
//             most dangerous object in each sector.  This is the primary input
//             to the risk predictor feature vector.
//
// Physics model
// ─────────────
// Constant-velocity linear extrapolation:
//
//   object position at time t:  p(t) = [px + vx*t,  py + vy*t]
//   user   position at time t:  u(t) = [0, 0]  (user is always origin)
//
// Relative position:            r(t) = p(t) - u(t) = [px + vx*t, py + vy*t]
// Distance squared:             d²(t) = (px+vx*t)² + (py+vy*t)²
//
// TTC: solve d(t) = COLLISION_RADIUS_MM for smallest positive t.
//   d²(t) = 0 is a quadratic in t:
//     (vx²+vy²)t² + 2(px*vx+py*vy)t + (px²+py²-R²) = 0
//   where R = COLLISION_RADIUS_MM.
//   If discriminant < 0 → no collision on current trajectory → TTC = ∞.
//   If both roots negative → already past us → TTC = ∞.
//   Else TTC = smallest positive root.
//
// CPA: d²(t) is minimised at t* = -dot(p,v)/dot(v,v).
//   If t* < 0 → object is moving away → CPA is current distance.
//
// Coordinate frame
// ────────────────
//   Same as perception:  +X forward, +Y left, origin = user, mm and mm/s.
//
// Thread safety
// ─────────────
//   TTCEngine is stateless (all methods are const or static).  Safe to call
//   from any thread simultaneously.

#include "perception/tracker.h"

#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>

namespace prediction {

// ─── Constants ────────────────────────────────────────────────────────────────

// Objects closer than this are treated as a physical collision (mm).
// Roughly half a body width — 300 mm radius around the user.
static constexpr float COLLISION_RADIUS_MM  = 300.0f;

// Minimum closing speed to compute a meaningful TTC (mm/s).
// Below this, the object is essentially stationary relative to the user.
static constexpr float MIN_CLOSING_SPEED    = 30.0f;

// TTC values above this are clipped to ∞ (irrelevant for immediate safety).
static constexpr float MAX_TTC_S            = 10.0f;

// Time horizon for projected path waypoints (seconds).
static constexpr float HORIZON_S            = 3.0f;

// Number of waypoints in the projected path (including t=0).
static constexpr int   PATH_STEPS           = 7;

// Number of 45° angular sectors (0°–45°, 45°–90°, …, 315°–360°).
static constexpr int   NUM_SECTORS          = 8;

// Velocity below which we consider a track stationary (mm/s).
static constexpr float STATIONARY_THRESHOLD = 50.0f;

// ─── PathPoint ────────────────────────────────────────────────────────────────
//
// One waypoint in a projected trajectory.

struct PathPoint {
    float t_s   = 0.0f;   // time offset from now (seconds)
    float x_mm  = 0.0f;   // predicted X position (mm)
    float y_mm  = 0.0f;   // predicted Y position (mm)
    float dist_mm = 0.0f; // distance from user at this time
};

// ─── CPAResult ────────────────────────────────────────────────────────────────
//
// Closest Point of Approach for one tracked object.

struct CPAResult {
    float time_s     = 0.0f;   // time of CPA from now (seconds); 0 if already past
    float distance_mm = 0.0f;  // minimum distance at CPA (mm)
    float x_mm       = 0.0f;   // object X at CPA
    float y_mm       = 0.0f;   // object Y at CPA

    // True if the CPA is dangerously close (within 2× COLLISION_RADIUS_MM).
    bool is_dangerous() const {
        return distance_mm < COLLISION_RADIUS_MM * 2.0f;
    }
};

// ─── TTCResult ────────────────────────────────────────────────────────────────
//
// Full collision analysis for one tracked object.

struct TTCResult {
    // ── Source identity ───────────────────────────────────────────────────────
    uint32_t object_id    = 0;
    int      sector       = -1;   // 45° sector index [0,7]; -1 if unassigned

    // ── Position & motion ─────────────────────────────────────────────────────
    float    distance_mm       = 0.0f;
    float    bearing_deg       = 0.0f;
    float    closing_speed_mm_s = 0.0f;
    float    speed_mm_s        = 0.0f;

    // ── TTC ───────────────────────────────────────────────────────────────────
    // Seconds until object reaches COLLISION_RADIUS_MM of the user.
    // Set to +∞ if no collision predicted on current trajectory.
    float    ttc_s = std::numeric_limits<float>::infinity();

    // True if TTC is a finite, meaningful value (< MAX_TTC_S).
    bool has_ttc() const {
        return std::isfinite(ttc_s) && ttc_s < MAX_TTC_S;
    }

    // ── CPA ───────────────────────────────────────────────────────────────────
    CPAResult cpa;

    // ── Trajectory projection ─────────────────────────────────────────────────
    // PATH_STEPS waypoints from t=0 to t=HORIZON_S at equal time intervals.
    std::array<PathPoint, PATH_STEPS> path{};

    // ── Object metadata ───────────────────────────────────────────────────────
    float    width_mm    = 0.0f;
    float    depth_mm    = 0.0f;
    int      point_count = 0;
    const char* size_label = "unknown";
    bool     velocity_reliable = false;
    bool     is_stationary     = false;

    // ── Derived urgency score ─────────────────────────────────────────────────
    // Composite urgency in [0, 1]:
    //   1.0 = immediate collision,  0.0 = no threat.
    // Used to sort TTCResults and select the most urgent object per sector.
    float urgency_score() const {
        if (!std::isfinite(ttc_s) || ttc_s <= 0.0f) {
            // Already inside collision radius or no trajectory collision.
            // Use CPA danger as fallback.
            float cpa_danger = cpa.is_dangerous()
                ? 1.0f - (cpa.distance_mm / (COLLISION_RADIUS_MM * 2.0f))
                : 0.0f;
            return std::clamp(cpa_danger, 0.0f, 1.0f);
        }
        // TTC score: 1.0 at t=0, 0.0 at t=MAX_TTC_S.
        float ttc_score = 1.0f - std::clamp(ttc_s / MAX_TTC_S, 0.0f, 1.0f);

        // Proximity score: 1.0 at distance=0, 0.0 at distance=5000mm.
        float prox_score = 1.0f - std::clamp(distance_mm / 5000.0f, 0.0f, 1.0f);

        // Weighted combination.
        return 0.7f * ttc_score + 0.3f * prox_score;
    }

    // ── Alert string ──────────────────────────────────────────────────────────
    // Human-readable summary for the TTS engine.
    // e.g. "Obstacle 1.2 metres ahead, collision in 2 seconds"
    std::string alert_str() const;
};

// ─── SectorThreat ─────────────────────────────────────────────────────────────
//
// Summary of the most dangerous object in one 45° angular sector.
// The 8-element array of SectorThreats is the primary feature input to the
// risk predictor MLP.

struct SectorThreat {
    int   sector          = -1;     // sector index [0,7]
    float sector_lo_deg   = 0.0f;   // lower bound of sector (degrees)
    float sector_hi_deg   = 0.0f;   // upper bound of sector (degrees)

    bool  occupied        = false;  // true if any confirmed object is in sector

    // Fields below are only valid when occupied == true.
    float min_distance_mm = std::numeric_limits<float>::max();
    float min_ttc_s       = std::numeric_limits<float>::infinity();
    float max_closing_speed = 0.0f;
    float max_urgency     = 0.0f;
    int   object_count    = 0;
    uint32_t closest_id   = 0;      // id of the closest object in this sector

    // Normalised distance: 1.0 = nothing detected, 0.0 = right on top of user.
    // Safe to use directly as a MLP feature (always in [0,1]).
    float normalised_distance() const {
        if (!occupied) return 1.0f;
        return std::clamp(min_distance_mm / 6000.0f, 0.0f, 1.0f);
    }

    // Normalised TTC: 1.0 = no collision / far away, 0.0 = imminent.
    float normalised_ttc() const {
        if (!occupied || !std::isfinite(min_ttc_s)) return 1.0f;
        return std::clamp(min_ttc_s / MAX_TTC_S, 0.0f, 1.0f);
    }

    // Human-readable sector name.
    const char* name() const {
        switch (sector) {
            case 0: return "ahead";
            case 1: return "ahead-right";
            case 2: return "right";
            case 3: return "behind-right";
            case 4: return "behind";
            case 5: return "behind-left";
            case 6: return "left";
            case 7: return "ahead-left";
            default: return "unknown";
        }
    }
};

// ─── TTCFrame ─────────────────────────────────────────────────────────────────
//
// Complete TTC analysis for one perception frame.
// Produced by TTCEngine::compute() and consumed by:
//   - RiskPredictor (feature extraction)
//   - AlertPolicy   (most urgent threat)
//   - SceneBuilder  (OpenAI agent context)

struct TTCFrame {
    uint64_t frame_id  = 0;
    float    dt_s      = 0.0f;

    // Per-object TTC results, sorted by urgency descending (most urgent first).
    std::vector<TTCResult> results;

    // Per-sector threat summary (8 sectors × 45° each).
    std::array<SectorThreat, NUM_SECTORS> sectors{};

    // ── Convenience accessors ─────────────────────────────────────────────────

    // Most urgent TTCResult across all objects, or nullptr if results empty.
    const TTCResult* most_urgent() const {
        if (results.empty()) return nullptr;
        return &results.front();   // sorted descending by urgency
    }

    // Most urgent object in the forward sector (sector 0, 0°–45°, wrapping
    // through sector 7, 315°–360°).
    const TTCResult* most_urgent_forward() const {
        const TTCResult* best = nullptr;
        float best_urgency = -1.0f;
        for (const auto& r : results) {
            bool fwd = (r.bearing_deg <= 22.5f || r.bearing_deg > 337.5f);
            if (fwd && r.urgency_score() > best_urgency) {
                best_urgency = r.urgency_score();
                best = &r;
            }
        }
        return best;
    }

    // True if any object has a finite TTC within threshold_s seconds.
    bool has_collision_within(float threshold_s) const {
        for (const auto& r : results) {
            if (r.has_ttc() && r.ttc_s <= threshold_s) return true;
        }
        return false;
    }

    // Returns all results with TTC < threshold_s, sorted by TTC ascending.
    std::vector<const TTCResult*> imminent(float threshold_s = 3.0f) const {
        std::vector<const TTCResult*> out;
        for (const auto& r : results) {
            if (r.has_ttc() && r.ttc_s <= threshold_s) {
                out.push_back(&r);
            }
        }
        std::sort(out.begin(), out.end(),
                  [](const TTCResult* a, const TTCResult* b) {
                      return a->ttc_s < b->ttc_s;
                  });
        return out;
    }

    // Global minimum TTC across all objects (∞ if none imminent).
    float min_ttc() const {
        float m = std::numeric_limits<float>::infinity();
        for (const auto& r : results) {
            if (r.has_ttc()) m = std::min(m, r.ttc_s);
        }
        return m;
    }
};

// ─── TTCEngine ────────────────────────────────────────────────────────────────
//
// Stateless engine — holds no per-frame state.
// Instantiate once; call compute() on every PerceptionResult.

class TTCEngine {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // collision_radius_mm : radius around user treated as a collision (mm)
    // Only confirmed tracks are processed by default (skip_tentative = true).
    explicit TTCEngine(float collision_radius_mm = COLLISION_RADIUS_MM,
                       bool  skip_tentative      = true)
        : collision_radius_mm_(collision_radius_mm)
        , skip_tentative_(skip_tentative)
    {}

    // ── Main entry point ──────────────────────────────────────────────────────

    // Compute full TTC analysis for a list of tracked objects.
    //
    // Returns a TTCFrame containing:
    //   - Per-object TTCResults sorted by urgency (most urgent first)
    //   - Sector threat table (8 sectors)
    //
    // dt_s is forwarded from the PerceptionResult and stored for reference.
    TTCFrame compute(const perception::TrackedObjectList& objects,
                     uint64_t frame_id = 0,
                     float    dt_s     = 0.0f) const;

    // ── Individual computations (public for testing / inspection) ─────────────

    // Compute TTC for a single object.
    // Returns ∞ if no collision predicted on current trajectory.
    float compute_ttc(float px, float py, float vx, float vy) const;

    // Compute CPA for a single object.
    CPAResult compute_cpa(float px, float py, float vx, float vy) const;

    // Project a constant-velocity trajectory PATH_STEPS waypoints forward.
    std::array<PathPoint, PATH_STEPS> project_path(
        float px, float py, float vx, float vy) const;

    // Assign an object at bearing_deg to its 45° sector index [0,7].
    // Sector 0 = 0°–45°, sector 7 = 315°–360°.
    static int bearing_to_sector(float bearing_deg);

    // Accessor for collision radius (for unit testing).
    float collision_radius_mm() const { return collision_radius_mm_; }

private:
    float collision_radius_mm_;
    bool  skip_tentative_;

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Build one TTCResult from a TrackedObject.
    TTCResult build_result(const perception::TrackedObject& obj) const;

    // Build the 8-element sector threat table from a list of TTCResults.
    static std::array<SectorThreat, NUM_SECTORS> build_sector_table(
        const std::vector<TTCResult>& results);
};

} // namespace prediction