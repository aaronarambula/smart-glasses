// ─── ttc_engine.cpp ───────────────────────────────────────────────────────────
// Time-To-Collision engine implementation.
// See include/prediction/ttc_engine.h for the full design notes and physics.

#include "prediction/ttc_engine.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

namespace prediction {

namespace {
bool debug_ttc_enabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("SMART_GLASSES_DEBUG_TTC");
        return v && v[0] != '\0' && std::string(v) != "0";
    }();
    return enabled;
}
} // namespace

// ─── TTCResult::alert_str ─────────────────────────────────────────────────────

std::string TTCResult::alert_str() const
{
    std::ostringstream ss;
    ss << std::fixed;

    // Direction
    const char* dir = "ahead";
    if (bearing_deg > 22.5f  && bearing_deg <= 67.5f)  dir = "ahead and to your right";
    else if (bearing_deg > 67.5f  && bearing_deg <= 112.5f) dir = "to your right";
    else if (bearing_deg > 112.5f && bearing_deg <= 157.5f) dir = "behind and to your right";
    else if (bearing_deg > 157.5f && bearing_deg <= 202.5f) dir = "behind you";
    else if (bearing_deg > 202.5f && bearing_deg <= 247.5f) dir = "behind and to your left";
    else if (bearing_deg > 247.5f && bearing_deg <= 292.5f) dir = "to your left";
    else if (bearing_deg > 292.5f && bearing_deg <= 337.5f) dir = "ahead and to your left";

    // Distance string
    ss << std::setprecision(1);
    float dist_m = distance_mm / 1000.0f;

    ss << size_label << " obstacle " << dist_m << " metres " << dir;

    if (has_ttc()) {
        if (ttc_s < 1.5f) {
            ss << " — collision imminent";
        } else {
            ss << " — collision in " << std::setprecision(0) << std::round(ttc_s)
               << " second" << (std::round(ttc_s) != 1.0f ? "s" : "");
        }
    } else if (cpa.is_dangerous()) {
        ss << std::setprecision(1)
           << " — will pass within " << cpa.distance_mm / 1000.0f << " metres";
    }

    return ss.str();
}

// ─── bearing_to_sector ────────────────────────────────────────────────────────
//
// Maps a bearing in [0, 360) to one of 8 sectors of 45° each.
// Sector 0 centred on 0° (forward), sector 1 centred on 45°, etc.
// We offset by half a sector width (22.5°) so that 0° falls in the middle
// of sector 0, not on the boundary between sector 0 and sector 7.

int TTCEngine::bearing_to_sector(float bearing_deg)
{
    // Normalise to [0, 360)
    while (bearing_deg <   0.0f) bearing_deg += 360.0f;
    while (bearing_deg >= 360.0f) bearing_deg -= 360.0f;

    return static_cast<int>(bearing_deg / 45.0f) % NUM_SECTORS;
}

// ─── compute_ttc ─────────────────────────────────────────────────────────────
//
// Solves for the smallest positive t such that |p + v*t| = collision_radius.
//
// Expanding:
//   (px + vx*t)² + (py + vy*t)² = R²
//   (vx²+vy²)t² + 2(px*vx+py*vy)t + (px²+py²-R²) = 0
//
//   a = vx²+vy²
//   b = 2*(px*vx + py*vy)
//   c = px²+py² - R²
//
// If a ≈ 0  → object nearly stationary relative to user → no TTC.
// If disc < 0 → no real roots → trajectories don't intersect → no TTC.
// Take the smallest positive root.

float TTCEngine::compute_ttc(float px, float py, float vx, float vy) const
{
    if (!std::isfinite(px) || !std::isfinite(py) ||
        !std::isfinite(vx) || !std::isfinite(vy)) {
        return std::numeric_limits<float>::infinity();
    }

    const float R  = collision_radius_mm_;
    const float a  = vx*vx + vy*vy;

    // Object is essentially stationary.
    if (a < 1.0f) return std::numeric_limits<float>::infinity();

    const float b  =  2.0f * (px*vx + py*vy);
    const float c  = px*px + py*py - R*R;

    const float disc = b*b - 4.0f*a*c;

    // No real intersection.
    if (disc < 0.0f) return std::numeric_limits<float>::infinity();

    const float sqrt_disc = std::sqrt(disc);
    const float t1 = (-b - sqrt_disc) / (2.0f * a);
    const float t2 = (-b + sqrt_disc) / (2.0f * a);

    // We want the smallest positive root — i.e. the first future collision.
    float ttc = std::numeric_limits<float>::infinity();
    if (t1 > 0.0f) ttc = std::min(ttc, t1);
    if (t2 > 0.0f) ttc = std::min(ttc, t2);

    // Clip to MAX_TTC_S — beyond that it's not actionable.
    if (!std::isfinite(ttc) || ttc > MAX_TTC_S) {
        return std::numeric_limits<float>::infinity();
    }

    return ttc;
}

// ─── compute_cpa ─────────────────────────────────────────────────────────────
//
// Closest Point of Approach.
//
// d²(t) = |p + v*t|² = (vx²+vy²)t² + 2(px*vx+py*vy)t + (px²+py²)
//
// This is a upward-opening parabola in t.
// Minimum at t* = -(px*vx + py*vy) / (vx²+vy²).
//
// If t* < 0, the closest approach already happened (object moving away).
// In that case CPA is the current distance.

CPAResult TTCEngine::compute_cpa(float px, float py, float vx, float vy) const
{
    CPAResult cpa;

    const float speed_sq = vx*vx + vy*vy;
    float t_star = 0.0f;

    if (speed_sq > 1.0f) {
        t_star = -(px*vx + py*vy) / speed_sq;
    }

    // Clamp to [0, HORIZON_S] — only future approaches within our horizon.
    t_star = std::max(0.0f, std::min(t_star, HORIZON_S));

    cpa.time_s = t_star;
    cpa.x_mm   = px + vx * t_star;
    cpa.y_mm   = py + vy * t_star;
    cpa.distance_mm = std::sqrt(cpa.x_mm * cpa.x_mm + cpa.y_mm * cpa.y_mm);

    return cpa;
}

// ─── project_path ────────────────────────────────────────────────────────────
//
// Produces PATH_STEPS waypoints evenly spaced from t=0 to t=HORIZON_S.
// Each waypoint gives the object's predicted (x,y) and distance from user.

std::array<PathPoint, PATH_STEPS> TTCEngine::project_path(
    float px, float py, float vx, float vy) const
{
    std::array<PathPoint, PATH_STEPS> path{};

    const float dt = HORIZON_S / static_cast<float>(PATH_STEPS - 1);

    for (int i = 0; i < PATH_STEPS; ++i) {
        const float t = i * dt;
        path[static_cast<size_t>(i)].t_s     = t;
        path[static_cast<size_t>(i)].x_mm    = px + vx * t;
        path[static_cast<size_t>(i)].y_mm    = py + vy * t;

        float dx = path[static_cast<size_t>(i)].x_mm;
        float dy = path[static_cast<size_t>(i)].y_mm;
        path[static_cast<size_t>(i)].dist_mm = std::sqrt(dx*dx + dy*dy);
    }

    return path;
}

// ─── build_result ────────────────────────────────────────────────────────────
//
// Assembles a complete TTCResult from one TrackedObject.

TTCResult TTCEngine::build_result(const perception::TrackedObject& obj) const
{
    TTCResult r;

    r.object_id   = obj.id;
    r.distance_mm = obj.distance_mm;
    r.bearing_deg = obj.bearing_deg;
    r.closing_speed_mm_s = obj.closing_speed_mm_s;
    r.speed_mm_s  = obj.speed_mm_s;
    r.width_mm    = obj.width_mm;
    r.depth_mm    = obj.depth_mm;
    r.point_count = obj.point_count;
    r.size_label  = obj.size_label;
    r.velocity_reliable = obj.velocity_reliable();
    r.is_stationary     = (obj.speed_mm_s < STATIONARY_THRESHOLD);
    r.sector      = bearing_to_sector(obj.bearing_deg);

    if (!std::isfinite(obj.px) || !std::isfinite(obj.py) ||
        !std::isfinite(obj.vx) || !std::isfinite(obj.vy)) {
        r.cpa = CPAResult{};
        r.path = project_path(0.0f, 0.0f, 0.0f, 0.0f);
        return r;
    }

    // ── TTC ───────────────────────────────────────────────────────────────────
    // Relax the gating slightly so early confirmed tracks can still produce a
    // physically meaningful TTC before the tracker reaches full "reliable"
    // status. This helps the system detect approaching objects earlier without
    // rewriting the tracker.
    const bool has_motion   = obj.speed_mm_s > MIN_CLOSING_SPEED;
    const bool approaching  = obj.closing_speed_mm_s > MIN_CLOSING_SPEED;
    const bool enough_hits  = obj.hits >= 2;

    if (enough_hits && has_motion && approaching) {
        r.ttc_s = compute_ttc(obj.px, obj.py, obj.vx, obj.vy);
    }
    // Proximity override: even a stationary object triggers a finite "TTC" if
    // it is already inside the collision radius.  This handles walls that the
    // user is walking toward faster than the sensor can detect approach.
    if (obj.distance_mm < collision_radius_mm_ * 1.5f) {
        // Use distance / walking speed as a conservative estimate.
        // Average walking speed ≈ 1400 mm/s.
        constexpr float WALK_SPEED_MM_S = 1400.0f;
        float dist_ttc = obj.distance_mm / WALK_SPEED_MM_S;
        if (!std::isfinite(r.ttc_s) || dist_ttc < r.ttc_s) {
            r.ttc_s = dist_ttc;
        }
    }

    // ── CPA ───────────────────────────────────────────────────────────────────
    r.cpa = compute_cpa(obj.px, obj.py, obj.vx, obj.vy);

    // ── Projected path ────────────────────────────────────────────────────────
    r.path = project_path(obj.px, obj.py, obj.vx, obj.vy);

    if (debug_ttc_enabled() && enough_hits) {
        std::cout << "[debug-ttc] id=" << r.object_id
                  << " px=" << obj.px
                  << " py=" << obj.py
                  << " vx=" << obj.vx
                  << " vy=" << obj.vy
                  << " dist=" << r.distance_mm
                  << " closing=" << r.closing_speed_mm_s
                  << " ttc=" << (std::isfinite(r.ttc_s) ? r.ttc_s : -1.0f)
                  << " reliable=" << obj.velocity_reliable()
                  << " hits=" << obj.hits
                  << "\n";
    }

    return r;
}

// ─── build_sector_table ───────────────────────────────────────────────────────
//
// Groups TTCResults into 8 × 45° sectors and for each sector records:
//   - Whether any object is present
//   - The minimum distance
//   - The minimum TTC
//   - The maximum closing speed
//   - The maximum urgency score
//   - The count of objects
//   - The ID of the closest object
//
// This 8-element table is the backbone of the 24-feature MLP input vector.

std::array<SectorThreat, NUM_SECTORS> TTCEngine::build_sector_table(
    const std::vector<TTCResult>& results)
{
    std::array<SectorThreat, NUM_SECTORS> sectors{};

    // Initialise sector metadata.
    for (int s = 0; s < NUM_SECTORS; ++s) {
        sectors[static_cast<size_t>(s)].sector       = s;
        sectors[static_cast<size_t>(s)].sector_lo_deg = s * 45.0f;
        sectors[static_cast<size_t>(s)].sector_hi_deg = (s + 1) * 45.0f;
    }

    for (const auto& r : results) {
        if (r.sector < 0 || r.sector >= NUM_SECTORS) continue;

        auto& st = sectors[static_cast<size_t>(r.sector)];

        st.occupied     = true;
        st.object_count++;

        // Closest object in sector.
        if (r.distance_mm < st.min_distance_mm) {
            st.min_distance_mm = r.distance_mm;
            st.closest_id      = r.object_id;
        }

        // Minimum TTC in sector.
        if (std::isfinite(r.ttc_s) && r.ttc_s < st.min_ttc_s) {
            st.min_ttc_s = r.ttc_s;
        }

        // Maximum closing speed in sector.
        if (r.closing_speed_mm_s > st.max_closing_speed) {
            st.max_closing_speed = r.closing_speed_mm_s;
        }

        // Maximum urgency in sector.
        float u = r.urgency_score();
        if (u > st.max_urgency) {
            st.max_urgency = u;
        }
    }

    return sectors;
}

// ─── compute ─────────────────────────────────────────────────────────────────
//
// Main entry point. Processes all tracked objects, builds TTCResults and the
// sector threat table, returns a fully populated TTCFrame.

TTCFrame TTCEngine::compute(const perception::TrackedObjectList& objects,
                             uint64_t frame_id,
                             float    dt_s) const
{
    TTCFrame frame;
    frame.frame_id = frame_id;
    frame.dt_s     = dt_s;

    frame.results.reserve(objects.size());

    for (const auto& obj : objects) {
        // Skip dead or (optionally) tentative tracks.
        if (obj.state == perception::TrackState::DEAD) continue;
        if (skip_tentative_ && obj.state == perception::TrackState::TENTATIVE) continue;

        frame.results.push_back(build_result(obj));
    }

    // Sort results by urgency descending — most urgent threat first.
    std::sort(frame.results.begin(), frame.results.end(),
              [](const TTCResult& a, const TTCResult& b) {
                  return a.urgency_score() > b.urgency_score();
              });

    // Build the 8-sector threat table.
    frame.sectors = build_sector_table(frame.results);

    return frame;
}

} // namespace prediction
