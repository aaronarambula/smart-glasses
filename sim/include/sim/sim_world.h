#pragma once

// ─── sim_world.h ─────────────────────────────────────────────────────────────
// Simulation world primitives for the smart glasses LiDAR simulator.
//
// This file defines everything the simulated world is made of:
//   - Geometry primitives (circles, line segments, polygons)
//   - SimObject types matching real outdoor environments
//   - LiDAR noise model matching the LD06 sensor's actual characteristics
//   - A complete outdoor street scene with realistic motion
//
// Design constraints
// ──────────────────
// 1. ZERO coupling to any other module. This header includes only
//    <cmath>, <vector>, <array>, <string>, <random>, and <chrono>.
//    The sim is completely self-contained — it can be compiled and
//    tested in total isolation from perception, prediction, audio, agent.
//
// 2. The noise model is based on published LD06 datasheet specs plus
//    community measurements:
//      - Range noise:       Gaussian σ = 15 mm at 1 m, σ = 35 mm at 5 m
//      - Angular jitter:    Gaussian σ = 0.1° per measurement
//      - Dropout rate:      2% of rays return nothing (glass, mirror, far edge)
//      - Specular return:   1% of rays return a spurious far-distance reading
//        (sunlight on wet pavement, chrome bumpers, reflective signs)
//      - Shadow/occlusion:  objects behind closer objects are not visible
//        (ray stops at the first hit — just like real physics)
//      - Intensity falloff: quality drops with distance and incidence angle
//      - Scan motor jitter: ±0.05° angular position uncertainty per point
//
// 3. The outdoor environment is a representative 20m × 20m slice of a
//    typical suburban sidewalk/street scene:
//      - Building facade (wall) along one side
//      - Parked cars (static rectangular obstacles)
//      - Moving pedestrians (circles, realistic walking speed + sway)
//      - Moving cyclist (faster, narrower)
//      - Utility poles / trees (thin static circles)
//      - Curb line (low wall along street edge)
//      - Benches (static rectangles on sidewalk)
//
// Coordinate frame
// ────────────────
//   Same as the rest of the project:
//     +X = forward  (direction the user faces / walks)
//     +Y = left
//     Origin = user position (updates each frame as user walks forward)
//     All units: millimetres
//
// Usage
// ─────
//   SimWorld world;                      // build default outdoor scene
//   world.step(dt_s);                    // advance physics by dt seconds
//   auto scan = world.cast_rays(460);    // generate one LiDAR sweep
//   // cast_rays() returns a vector<RawRay> in angle-ascending order,
//   // ready for SimLidar to pack into a ScanFrame.

#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <chrono>
#include <limits>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <memory>

namespace sim {

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr float PI      = 3.14159265358979323846f;
static constexpr float TWO_PI  = 2.0f * PI;
static constexpr float DEG2RAD = PI / 180.0f;
static constexpr float RAD2DEG = 180.0f / PI;

// LD06 physical limits
static constexpr float LD06_MIN_RANGE_MM  =    50.0f;   // closer returns are noise
static constexpr float LD06_MAX_RANGE_MM  =  6000.0f;   // datasheet max reliable range
static constexpr float LD06_RPM           =   600.0f;   // 10 Hz × 60

// ─── Vec2 ─────────────────────────────────────────────────────────────────────
// 2-D vector in the world coordinate frame (mm).

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2  operator+(const Vec2& o) const { return {x+o.x, y+o.y}; }
    Vec2  operator-(const Vec2& o) const { return {x-o.x, y-o.y}; }
    Vec2  operator*(float s)       const { return {x*s,   y*s  }; }
    Vec2& operator+=(const Vec2& o)      { x+=o.x; y+=o.y; return *this; }

    float dot(const Vec2& o)  const { return x*o.x + y*o.y; }
    float length()            const { return std::sqrt(x*x + y*y); }
    float length_sq()         const { return x*x + y*y; }

    Vec2 normalised() const {
        float l = length();
        return (l > 1e-6f) ? Vec2{x/l, y/l} : Vec2{0,0};
    }

    // Rotate by angle_rad counter-clockwise.
    Vec2 rotated(float angle_rad) const {
        float c = std::cos(angle_rad);
        float s = std::sin(angle_rad);
        return {x*c - y*s, x*s + y*c};
    }
};

// ─── Ray ──────────────────────────────────────────────────────────────────────
// A ray originating from the user (origin = world user position).

struct Ray {
    float angle_deg = 0.0f;   // bearing [0, 360), 0 = forward, clockwise

    // Direction unit vector in world frame.
    Vec2 direction() const {
        float rad = angle_deg * DEG2RAD;
        return { std::cos(rad), -std::sin(rad) };  // -sin: +Y = left, angle clockwise
    }
};

// ─── RawRay ───────────────────────────────────────────────────────────────────
// Result of casting one ray through the world, before noise is applied.
// The noise model runs on top of this to produce the final sensor reading.

struct RawRay {
    float   angle_deg     = 0.0f;   // requested bearing
    float   true_dist_mm  = 0.0f;   // geometric hit distance (0 = no hit)
    float   incidence_deg = 90.0f;  // angle between ray and surface normal at hit
                                     // 90° = perpendicular (best return)
                                     //  0° = grazing      (worst return, likely dropout)
    bool    hit           = false;  // false = ray escaped to max range
    int     object_id     = -1;     // which SimObject was hit (-1 = none)
};

// ─── NoisedRay ────────────────────────────────────────────────────────────────
// One ray after the full LD06 noise model has been applied.
// This maps directly to one ScanPoint.

struct NoisedRay {
    float   angle_deg    = 0.0f;   // angle after motor jitter applied
    float   distance_mm  = 0.0f;   // measured distance (noisy), 0 = no return
    uint8_t quality      = 0;      // simulated signal quality [0, 255]
    bool    valid        = false;  // false = dropout or no hit
};

// ─── NoiseModel ───────────────────────────────────────────────────────────────
// Applies realistic LD06 sensor imperfections to a geometric ray result.
//
// Parameters are tuned to match LD06 datasheet + community measurements.
// All are public so scenarios can override them (e.g. simulate a rainy day
// with higher dropout rate).

struct NoiseModel {
    // ── Range noise ───────────────────────────────────────────────────────────
    // Standard deviation of Gaussian range noise, in mm, at 1 m range.
    // Noise scales with distance: sigma(d) = sigma_1m * sqrt(d / 1000)
    float sigma_range_1m_mm   = 15.0f;

    // ── Angular jitter ────────────────────────────────────────────────────────
    // Std dev of angular position uncertainty per measurement (degrees).
    // Comes from motor encoder resolution + bearing play.
    float sigma_angle_deg     = 0.10f;

    // ── Dropout ───────────────────────────────────────────────────────────────
    // Probability that a valid return is simply not reported (0.0–1.0).
    // Happens with glass surfaces, specular materials, and far edges.
    float dropout_prob        = 0.02f;

    // ── Grazing dropout ───────────────────────────────────────────────────────
    // At shallow incidence angles (<= grazing_threshold_deg), dropout
    // probability increases linearly to grazing_dropout_prob.
    float grazing_threshold_deg   = 20.0f;
    float grazing_dropout_prob    = 0.40f;

    // ── Specular ghost return ─────────────────────────────────────────────────
    // Probability that a ray returns a spurious far-distance ghost reading.
    // Models sunlight reflection off wet pavement, car paint, glass facades.
    float specular_prob       = 0.01f;
    float specular_min_mm     = 4000.0f;  // ghost is always far away
    float specular_max_mm     = 6000.0f;

    // ── Quality model ─────────────────────────────────────────────────────────
    // Base quality at 0 distance; decreases with range and incidence angle.
    float quality_at_zero     = 240.0f;   // max realistic quality
    float quality_range_decay = 0.03f;    // quality lost per meter of range
    float quality_angle_decay = 1.5f;     // quality lost per degree off-perpendicular

    // ── Apply ─────────────────────────────────────────────────────────────────
    // Applies all noise effects to a RawRay. Modifies the provided rng.
    NoisedRay apply(const RawRay& raw, std::mt19937& rng) const {
        NoisedRay out;
        out.angle_deg = raw.angle_deg;

        if (!raw.hit) {
            // No geometric hit. Very small chance of a specular ghost.
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            if (u(rng) < specular_prob * 0.5f) {
                // Ghost return even with no real object.
                std::uniform_real_distribution<float> ghost(specular_min_mm, specular_max_mm);
                out.distance_mm = ghost(rng);
                out.quality     = 15;
                out.valid       = true;
            }
            return out;
        }

        // ── Compute dropout probability ───────────────────────────────────────
        float drop_p = dropout_prob;

        // Grazing angle penalty.
        float incidence = std::max(0.0f, std::min(90.0f, raw.incidence_deg));
        if (incidence < grazing_threshold_deg) {
            float t = 1.0f - incidence / grazing_threshold_deg;
            drop_p += t * (grazing_dropout_prob - dropout_prob);
        }
        drop_p = std::min(drop_p, 0.98f);

        std::uniform_real_distribution<float> u01(0.0f, 1.0f);

        // ── Dropout check ─────────────────────────────────────────────────────
        if (u01(rng) < drop_p) {
            // Point dropped. Occasionally replaced by specular ghost.
            if (u01(rng) < specular_prob) {
                std::uniform_real_distribution<float> ghost(specular_min_mm, specular_max_mm);
                out.distance_mm = ghost(rng);
                out.quality     = 12;
                out.valid       = true;
            }
            return out;
        }

        // ── Apply range noise ─────────────────────────────────────────────────
        float dist = raw.true_dist_mm;
        float sigma = sigma_range_1m_mm * std::sqrt(std::max(dist, 1.0f) / 1000.0f);
        std::normal_distribution<float> range_noise(0.0f, sigma);
        dist += range_noise(rng);
        dist = std::max(LD06_MIN_RANGE_MM, std::min(dist, LD06_MAX_RANGE_MM));

        // ── Apply angular jitter ──────────────────────────────────────────────
        std::normal_distribution<float> angle_noise(0.0f, sigma_angle_deg);
        out.angle_deg = raw.angle_deg + angle_noise(rng);
        // Keep in [0, 360)
        while (out.angle_deg <   0.0f) out.angle_deg += 360.0f;
        while (out.angle_deg >= 360.0f) out.angle_deg -= 360.0f;

        // ── Specular ghost overlay (rare) ─────────────────────────────────────
        if (u01(rng) < specular_prob) {
            std::uniform_real_distribution<float> ghost(specular_min_mm, specular_max_mm);
            dist    = ghost(rng);
            out.quality = 10;
            out.distance_mm = dist;
            out.valid = true;
            return out;
        }

        // ── Compute quality ───────────────────────────────────────────────────
        float q = quality_at_zero;
        q -= quality_range_decay * (dist / 1000.0f) * 255.0f;
        q -= quality_angle_decay * (90.0f - incidence);
        q  = std::max(5.0f, std::min(255.0f, q));

        // Small random quality variation (sensor electronics noise).
        std::normal_distribution<float> q_noise(0.0f, 8.0f);
        q += q_noise(rng);
        q = std::max(1.0f, std::min(255.0f, q));

        out.distance_mm = dist;
        out.quality     = static_cast<uint8_t>(q);
        out.valid       = true;
        return out;
    }
};

// ─── Material ──────────────────────────────────────────────────────────────────
// Surface material — affects dropout and quality without changing geometry.

enum class Material : uint8_t {
    Concrete,      // sidewalk, road — good return, low gloss
    Brick,         // building facades — good return
    Glass,         // storefronts, car windows — high dropout
    MetalMatte,    // lamp posts, signposts — good return
    MetalShiny,    // car bodies, bike frames — occasional ghost
    Foliage,       // bushes, trees — high dropout, noisy
    Clothing,      // pedestrians — moderate quality
    Retroreflect,  // road signs, safety vests — very high quality
    Asphalt,       // road surface — moderate quality
};

// Returns a dropout modifier for a material (multiplied on top of base dropout).
inline float material_dropout_multiplier(Material m) {
    switch (m) {
        case Material::Glass:       return 12.0f;
        case Material::Foliage:     return  5.0f;
        case Material::MetalShiny:  return  2.0f;
        case Material::Clothing:    return  1.2f;
        case Material::Retroreflect:return  0.1f;
        default:                    return  1.0f;
    }
}

// Returns a quality modifier for a material (additive, applied before clamping).
inline float material_quality_bonus(Material m) {
    switch (m) {
        case Material::Retroreflect: return  60.0f;
        case Material::Concrete:     return   5.0f;
        case Material::Brick:        return  10.0f;
        case Material::Glass:        return -40.0f;
        case Material::Foliage:      return -30.0f;
        default:                     return   0.0f;
    }
}

// ─── SimObject (base) ─────────────────────────────────────────────────────────
// Abstract base for everything in the simulated world.
// Each subclass implements ray_intersect() which returns the distance from
// the user (origin) to the surface, or -1 if no intersection.

struct SimObject {
    int      id       = 0;
    bool     is_static = true;    // false = moves each step()
    Material material  = Material::Concrete;
    std::string label;            // human-readable ("building_wall", "pedestrian_1")

    virtual ~SimObject() = default;

    // Returns distance from ray origin to the surface along the ray direction,
    // or -1.0f if the ray does not intersect this object.
    // Also fills incidence_deg_out: angle between the ray and the surface normal
    // at the hit point (90° = perfectly perpendicular).
    virtual float ray_intersect(const Vec2& origin,
                                 const Vec2& dir,
                                 float& incidence_deg_out) const = 0;

    // Advance the object's position/state by dt seconds.
    // Only called for !is_static objects.
    virtual void step(float dt_s) {}

    // Bounding radius (mm) — used for early-out culling.
    virtual float bounding_radius() const = 0;

    // Centre position in world frame (mm).
    virtual Vec2 centre() const = 0;
};

// ─── CircleObject ────────────────────────────────────────────────────────────
// A filled circle — models pedestrians, tree trunks, utility poles, wheels.

struct CircleObject : SimObject {
    Vec2  pos;          // centre in world frame (mm)
    float radius_mm;    // physical radius

    CircleObject(int id_, Vec2 pos_, float radius_mm_, Material mat, const std::string& lbl) {
        id         = id_;
        pos        = pos_;
        radius_mm  = radius_mm_;
        material   = mat;
        label      = lbl;
    }

    Vec2  centre()          const override { return pos; }
    float bounding_radius() const override { return radius_mm; }

    float ray_intersect(const Vec2& origin, const Vec2& dir,
                        float& incidence_deg_out) const override
    {
        // Analytic ray-circle intersection.
        // Solve: |origin + t*dir - pos|² = radius²
        Vec2 oc = origin - pos;
        float a = dir.dot(dir);           // always 1 since dir is normalised
        float b = 2.0f * oc.dot(dir);
        float c = oc.dot(oc) - radius_mm * radius_mm;
        float disc = b*b - 4.0f*a*c;

        if (disc < 0.0f) return -1.0f;   // no intersection

        float sqrt_disc = std::sqrt(disc);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);

        // We want the smallest positive t (front face of the circle).
        float t = -1.0f;
        if (t1 > 1.0f)       t = t1;
        else if (t2 > 1.0f)  t = t2;

        if (t < 0.0f) return -1.0f;

        // Compute incidence angle.
        Vec2 hit   = origin + dir * t;
        Vec2 normal = (hit - pos).normalised();
        float cos_inc = std::abs(normal.dot(dir));
        incidence_deg_out = std::acos(std::min(1.0f, cos_inc)) * RAD2DEG;
        // Convert: incidence angle to surface is 90° - angle between ray and normal
        incidence_deg_out = 90.0f - incidence_deg_out;

        return t;
    }
};

// ─── LineSegmentObject ────────────────────────────────────────────────────────
// An infinite-thickness wall segment — models building facades, curbs, fences.

struct LineSegmentObject : SimObject {
    Vec2 p0;   // start point (mm)
    Vec2 p1;   // end  point  (mm)

    LineSegmentObject(int id_, Vec2 p0_, Vec2 p1_, Material mat, const std::string& lbl) {
        id       = id_;
        p0       = p0_;
        p1       = p1_;
        material = mat;
        label    = lbl;
        is_static = true;
    }

    Vec2 centre() const override {
        return {(p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f};
    }
    float bounding_radius() const override {
        return (p1 - p0).length() * 0.5f + 100.0f;
    }

    float ray_intersect(const Vec2& origin, const Vec2& dir,
                        float& incidence_deg_out) const override
    {
        // Ray-segment intersection using parametric form.
        // Ray:     P(t) = origin + t * dir
        // Segment: Q(u) = p0 + u * (p1 - p0),  u in [0, 1]
        Vec2 seg = p1 - p0;
        Vec2 r   = origin - p0;

        float denom = dir.x * seg.y - dir.y * seg.x;
        if (std::abs(denom) < 1e-6f) return -1.0f;  // parallel

        float t = (r.x * seg.y - r.y * seg.x) / denom;
        float u = (r.x * dir.y - r.y * dir.x) / denom;

        if (t < 1.0f || u < 0.0f || u > 1.0f) return -1.0f;

        // Surface normal of the segment (perpendicular, left-hand side).
        Vec2 seg_norm = Vec2{-seg.y, seg.x}.normalised();
        float cos_inc = std::abs(seg_norm.dot(dir));
        // Incidence = 90° when ray hits perpendicularly (cos_inc = 1)
        incidence_deg_out = std::acos(std::min(1.0f, cos_inc)) * RAD2DEG;
        incidence_deg_out = 90.0f - incidence_deg_out;

        return t;
    }
};

// ─── RectObject ──────────────────────────────────────────────────────────────
// An axis-aligned (or rotated) filled rectangle — models parked cars, benches,
// bollards, dumpsters.

struct RectObject : SimObject {
    Vec2  pos;           // centre (mm)
    float width_mm;      // extent along local X
    float depth_mm;      // extent along local Y
    float heading_rad;   // rotation of the object (0 = aligned with world X)

    RectObject(int id_, Vec2 pos_, float w, float d, float heading,
               Material mat, const std::string& lbl)
        : pos(pos_), width_mm(w), depth_mm(d), heading_rad(heading)
    {
        id       = id_;
        material = mat;
        label    = lbl;
        is_static = true;
    }

    Vec2  centre()          const override { return pos; }
    float bounding_radius() const override {
        return std::sqrt(width_mm*width_mm + depth_mm*depth_mm) * 0.5f;
    }

    float ray_intersect(const Vec2& origin, const Vec2& dir,
                        float& incidence_deg_out) const override
    {
        // Transform ray into the rectangle's local frame, then do AABB test.
        Vec2 local_origin = (origin - pos).rotated(-heading_rad);
        Vec2 local_dir    = dir.rotated(-heading_rad);

        float hw = width_mm * 0.5f;
        float hd = depth_mm * 0.5f;

        // Slab method: intersect with 4 planes.
        float tx1 = (-hw - local_origin.x);
        float tx2 = ( hw - local_origin.x);
        float ty1 = (-hd - local_origin.y);
        float ty2 = ( hd - local_origin.y);

        if (std::abs(local_dir.x) > 1e-9f) {
            tx1 /= local_dir.x;
            tx2 /= local_dir.x;
        } else {
            tx1 = (tx1 < 0) ? -std::numeric_limits<float>::max()
                             :  std::numeric_limits<float>::max();
            tx2 = (tx2 < 0) ? -std::numeric_limits<float>::max()
                             :  std::numeric_limits<float>::max();
        }
        if (std::abs(local_dir.y) > 1e-9f) {
            ty1 /= local_dir.y;
            ty2 /= local_dir.y;
        } else {
            ty1 = (ty1 < 0) ? -std::numeric_limits<float>::max()
                             :  std::numeric_limits<float>::max();
            ty2 = (ty2 < 0) ? -std::numeric_limits<float>::max()
                             :  std::numeric_limits<float>::max();
        }

        float tmin = std::max(std::min(tx1, tx2), std::min(ty1, ty2));
        float tmax = std::min(std::max(tx1, tx2), std::max(ty1, ty2));

        if (tmax < 1.0f || tmin > tmax) return -1.0f;

        float t = (tmin > 1.0f) ? tmin : tmax;
        if (t < 1.0f) return -1.0f;

        // Determine which face was hit and compute incidence angle.
        Vec2 hit_local = local_origin + local_dir * t;
        Vec2 face_normal_local{0,0};

        float fx = std::abs(std::abs(hit_local.x) - hw);
        float fy = std::abs(std::abs(hit_local.y) - hd);

        if (fx < fy) {
            face_normal_local = Vec2{(hit_local.x > 0) ? 1.0f : -1.0f, 0.0f};
        } else {
            face_normal_local = Vec2{0.0f, (hit_local.y > 0) ? 1.0f : -1.0f};
        }

        // Rotate normal back to world frame.
        Vec2 face_normal = face_normal_local.rotated(heading_rad);
        float cos_inc = std::abs(face_normal.dot(dir));
        incidence_deg_out = 90.0f - std::acos(std::min(1.0f, cos_inc)) * RAD2DEG;

        return t;
    }
};

// ─── MovingPedestrian ────────────────────────────────────────────────────────
// A pedestrian modelled as a circle with:
//   - Constant forward velocity with realistic walking sway
//   - Body sway: the torso rocks ±80mm laterally at step frequency (1.8 Hz)
//   - Arm swing: modelled as occasional point reflections off the arms
//     which appear as extra noisy points very close to the body
//   - Pausing: 3% chance per second of stopping briefly (checking phone, etc.)

struct MovingPedestrian : SimObject {
    Vec2  pos;
    Vec2  velocity_mm_s;       // nominal walking direction and speed
    float radius_mm = 250.0f;  // shoulder width / 2

    // Sway state
    float sway_phase = 0.0f;   // radians, driven by step frequency
    static constexpr float SWAY_FREQ_HZ  = 1.8f;   // steps per second
    static constexpr float SWAY_AMP_MM   = 80.0f;   // lateral sway amplitude

    // Pause state
    bool  paused       = false;
    float pause_timer  = 0.0f;

    MovingPedestrian(int id_, Vec2 pos_, Vec2 vel, Material mat = Material::Clothing,
                     const std::string& lbl = "pedestrian") {
        id         = id_;
        pos        = pos_;
        velocity_mm_s = vel;
        material   = mat;
        label      = lbl;
        is_static  = false;
        radius_mm  = 250.0f;
    }

    Vec2  centre()          const override { return pos; }
    float bounding_radius() const override { return radius_mm + SWAY_AMP_MM; }

    void step(float dt_s) override {
        // Advance sway phase.
        sway_phase += TWO_PI * SWAY_FREQ_HZ * dt_s;

        if (paused) {
            pause_timer -= dt_s;
            if (pause_timer <= 0.0f) {
                paused = false;
            }
            return;
        }

        // Advance position.
        pos += velocity_mm_s * dt_s;

        // Apply body sway (lateral oscillation perpendicular to walking direction).
        // We bake it into the effective position for ray casting.
        // The sway is already included via the radius — we just modulate
        // the effective centre position here.
        Vec2 walk_dir = velocity_mm_s.normalised();
        Vec2 sway_dir = {-walk_dir.y, walk_dir.x};  // perpendicular (left)
        float sway_offset = SWAY_AMP_MM * std::sin(sway_phase);
        pos += sway_dir * (sway_offset * dt_s * SWAY_FREQ_HZ * 0.1f);
    }

    float ray_intersect(const Vec2& origin, const Vec2& dir,
                        float& incidence_deg_out) const override
    {
        // Use the circle intersection from CircleObject logic.
        Vec2 oc = origin - pos;
        float b = 2.0f * oc.dot(dir);
        float c = oc.dot(oc) - radius_mm * radius_mm;
        float disc = b*b - 4.0f*c;
        if (disc < 0.0f) return -1.0f;

        float sqrt_disc = std::sqrt(disc);
        float t1 = (-b - sqrt_disc) * 0.5f;
        float t2 = (-b + sqrt_disc) * 0.5f;

        float t = -1.0f;
        if (t1 > 1.0f)      t = t1;
        else if (t2 > 1.0f) t = t2;
        if (t < 0.0f) return -1.0f;

        Vec2 hit    = origin + dir * t;
        Vec2 normal = (hit - pos).normalised();
        float cos_inc = std::abs(normal.dot(dir));
        incidence_deg_out = 90.0f - std::acos(std::min(1.0f, cos_inc)) * RAD2DEG;

        return t;
    }
};

// ─── MovingCyclist ────────────────────────────────────────────────────────────
// A cyclist: faster, narrower, less predictable path (slight sinusoidal weave).

struct MovingCyclist : SimObject {
    Vec2  pos;
    Vec2  velocity_mm_s;
    float radius_mm  = 200.0f;
    float weave_phase = 0.0f;
    static constexpr float WEAVE_FREQ_HZ = 0.5f;
    static constexpr float WEAVE_AMP_MM  = 150.0f;

    MovingCyclist(int id_, Vec2 pos_, Vec2 vel, const std::string& lbl = "cyclist") {
        id            = id_;
        pos           = pos_;
        velocity_mm_s = vel;
        material      = Material::MetalShiny;
        label         = lbl;
        is_static     = false;
    }

    Vec2  centre()          const override { return pos; }
    float bounding_radius() const override { return radius_mm + WEAVE_AMP_MM; }

    void step(float dt_s) override {
        weave_phase += TWO_PI * WEAVE_FREQ_HZ * dt_s;
        pos += velocity_mm_s * dt_s;

        Vec2 walk_dir = velocity_mm_s.normalised();
        Vec2 perp     = {-walk_dir.y, walk_dir.x};
        float weave   = WEAVE_AMP_MM * std::sin(weave_phase) * dt_s * 0.5f;
        pos += perp * weave;
    }

    float ray_intersect(const Vec2& origin, const Vec2& dir,
                        float& incidence_deg_out) const override
    {
        Vec2 oc = origin - pos;
        float b = 2.0f * oc.dot(dir);
        float c = oc.dot(oc) - radius_mm * radius_mm;
        float disc = b*b - 4.0f*c;
        if (disc < 0.0f) return -1.0f;

        float sqrt_disc = std::sqrt(disc);
        float t1 = (-b - sqrt_disc) * 0.5f;
        float t2 = (-b + sqrt_disc) * 0.5f;

        float t = -1.0f;
        if (t1 > 1.0f)      t = t1;
        else if (t2 > 1.0f) t = t2;
        if (t < 0.0f) return -1.0f;

        Vec2 hit    = origin + dir * t;
        Vec2 normal = (hit - pos).normalised();
        float cos_inc = std::abs(normal.dot(dir));
        incidence_deg_out = 90.0f - std::acos(std::min(1.0f, cos_inc)) * RAD2DEG;

        return t;
    }
};

// ─── SimWorld ─────────────────────────────────────────────────────────────────
// The complete simulated outdoor environment.
//
// Default scene: a suburban sidewalk on a city block.
// The user walks forward at ~1.2 m/s (average pedestrian speed).
// All world coordinates are absolute; the user position shifts forward each frame.

class SimWorld {
public:
    // ── User state ────────────────────────────────────────────────────────────
    Vec2  user_pos{0.0f, 0.0f};    // user position in world frame (mm)
    float user_speed_mm_s = 1200.0f; // 1.2 m/s walking speed
    float sim_time_s      = 0.0f;

    // ── Noise model ───────────────────────────────────────────────────────────
    NoiseModel noise;

    // ── Objects ───────────────────────────────────────────────────────────────
    std::vector<std::unique_ptr<SimObject>> objects;

    // ── RNG ───────────────────────────────────────────────────────────────────
    std::mt19937 rng;

    // ── Construction ──────────────────────────────────────────────────────────
    // seed: random seed (0 = time-based, for reproducible runs use e.g. 42)
    explicit SimWorld(uint32_t seed = 0) {
        rng = std::mt19937(seed == 0
            ? static_cast<uint32_t>(
                std::chrono::steady_clock::now().time_since_epoch().count())
            : seed);
        build_default_scene();
    }

    // ── Step ──────────────────────────────────────────────────────────────────
    // Advance all moving objects and the user forward by dt_s seconds.
    void step(float dt_s) {
        sim_time_s += dt_s;

        // Advance user.
        user_pos.x += user_speed_mm_s * dt_s;

        // Advance all dynamic objects.
        for (auto& obj : objects) {
            if (!obj->is_static) {
                obj->step(dt_s);
            }
        }
    }

    // ── Cast rays ─────────────────────────────────────────────────────────────
    // Generate one full 360° sweep of num_rays rays from the user position.
    // Returns NoisedRay results in angle-ascending order, ready for SimLidar
    // to pack into a ScanFrame.
    //
    // Ray casting is physically correct:
    //   - Rays stop at the FIRST object they hit (occlusion is free)
    //   - All objects within LD06_MAX_RANGE_MM are tested
    //   - Noise is applied per-ray after geometry
    std::vector<NoisedRay> cast_rays(int num_rays) {
        std::vector<NoisedRay> results;
        results.reserve(static_cast<size_t>(num_rays));

        const float angle_step = 360.0f / static_cast<float>(num_rays);

        for (int i = 0; i < num_rays; ++i) {
            float angle_deg = static_cast<float>(i) * angle_step;

            RawRay raw;
            raw.angle_deg = angle_deg;

            float rad = angle_deg * DEG2RAD;
            Vec2 dir{std::cos(rad), -std::sin(rad)};   // +X forward, clockwise angle

            // Find the nearest hit among all objects.
            float min_t = LD06_MAX_RANGE_MM;
            float best_incidence = 90.0f;
            int   best_id        = -1;

            for (const auto& obj : objects) {
                // Quick bounding-radius cull.
                Vec2  to_obj = obj->centre() - user_pos;
                float dist_to_centre = to_obj.length();
                if (dist_to_centre - obj->bounding_radius() > LD06_MAX_RANGE_MM) continue;

                float incidence = 90.0f;
                float t = obj->ray_intersect(user_pos, dir, incidence);

                if (t > LD06_MIN_RANGE_MM && t < min_t) {
                    min_t          = t;
                    best_incidence = incidence;
                    best_id        = obj->id;
                }
            }

            if (best_id >= 0) {
                raw.hit           = true;
                raw.true_dist_mm  = min_t;
                raw.incidence_deg = best_incidence;
                raw.object_id     = best_id;
            }

            // Apply material-specific noise modifiers.
            NoiseModel ray_noise = noise;  // copy base noise
            if (best_id >= 0) {
                for (const auto& obj : objects) {
                    if (obj->id == best_id) {
                        float dm = material_dropout_multiplier(obj->material);
                        ray_noise.dropout_prob = std::min(0.98f, noise.dropout_prob * dm);
                        break;
                    }
                }
            }

            results.push_back(ray_noise.apply(raw, rng));
        }

        return results;
    }

    // ── Scene builder ─────────────────────────────────────────────────────────
    // Constructs the default outdoor suburban sidewalk scene.
    // Called automatically by the constructor.
    void build_default_scene();

    // Append individual object types for custom scenarios.
    void add_pedestrian(Vec2 pos, Vec2 vel, const std::string& label = "pedestrian");
    void add_cyclist(Vec2 pos, Vec2 vel, const std::string& label = "cyclist");
    void add_parked_car(Vec2 pos, float heading_rad, const std::string& label = "car");
    void add_wall(Vec2 p0, Vec2 p1, Material mat = Material::Brick,
                  const std::string& label = "wall");
    void add_pole(Vec2 pos, float radius_mm = 60.0f, const std::string& label = "pole");

    // Clear all objects and rebuild from scratch.
    void reset(uint32_t new_seed = 0);

    // ── Scene builders (called by SimLidar scene routing) ─────────────────────
    // Each method populates objects with the geometry for that scene.
    // build_default_scene() is called automatically by the constructor and
    // builds the sidewalk scene. The others are called explicitly by SimLidar
    // after clearing objects when a non-default scene is requested.
    void build_scene_crossing();
    void build_scene_hallway();
    void build_scene_parking_lot();
    void build_scene_cyclist_overtake();
    void build_scene_crowd();

private:
    int next_id_ = 1;
    int alloc_id() { return next_id_++; }
};

} // namespace sim
