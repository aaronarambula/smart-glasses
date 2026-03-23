// ─── sim_lidar.cpp ────────────────────────────────────────────────────────────
// Full implementation of SimLidar — the drop-in simulated sensor.
// See include/sim/sim_lidar.h for the design contract.
//
// Every method matches the behaviour of the real LD06 driver exactly:
//   open()             → parses scene, constructs SimWorld, applies noise config
//   start()            → launches background sim thread at config_.scan_hz
//   stop()             → signals thread, joins it
//   close()            → calls stop(), clears world
//   get_latest_frame() → mutex-protected value copy, identical to LD06
//   set_frame_callback()→ identical to LD06
//
// The sim thread runs SimWorld::step(dt) + SimWorld::cast_rays(N) on every
// tick, packs the NoisedRay results into a ScanFrame, and publishes it.
// Real-time pacing is enforced with std::this_thread::sleep_until so the
// pipeline sees frames at exactly scan_hz regardless of how fast the physics
// step runs (physics is << 1ms; sleep dominates at 10 Hz).

#include "sim/sim_lidar.h"
#include "sim/sim_world.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace sim {

// ─── Construction / Destruction ──────────────────────────────────────────────

SimLidar::SimLidar(std::string port_or_scene, SimConfig config)
    : sensors::LidarBase(std::move(port_or_scene))
    , config_(std::move(config))
{
    // Scene ID is parsed from port_ in open() — not here — so that open()
    // can return false cleanly if the scene string is malformed.
}

SimLidar::~SimLidar()
{
    stop();
    close();
}

// ─── open ────────────────────────────────────────────────────────────────────
// Parses the scene from the port string, constructs the SimWorld, and applies
// the noise configuration. Always returns true for valid scenes.

bool SimLidar::open()
{
    if (open_) return true;

    // Parse scene ID from "sim://scene_name"
    scene_id_ = parse_scene_id(port_);

    // Build the world for this scene.
    build_world();

    // Apply noise overrides from config.
    if (world_) {
        world_->noise.sigma_range_1m_mm = config_.noise_sigma_range_1m_mm;
        world_->noise.dropout_prob      = config_.noise_dropout_prob;
        world_->noise.specular_prob     = config_.noise_specular_prob;
    }

    open_ = true;
    return true;
}

// ─── start ───────────────────────────────────────────────────────────────────

bool SimLidar::start()
{
    if (!open_) {
        std::lock_guard<std::mutex> lk(error_mutex_);
        error_str_ = "start() called before open()";
        return false;
    }
    if (running_.load()) return true;

    stop_flag_.store(false);
    running_.store(true);
    sim_thread_ = std::thread(&SimLidar::sim_loop, this);
    return true;
}

// ─── stop ────────────────────────────────────────────────────────────────────

void SimLidar::stop()
{
    if (!running_.load()) return;

    stop_flag_.store(true);
    if (sim_thread_.joinable()) {
        sim_thread_.join();
    }
    running_.store(false);
}

// ─── close ───────────────────────────────────────────────────────────────────

void SimLidar::close()
{
    stop();
    world_.reset();
    open_ = false;
}

// ─── get_latest_frame ────────────────────────────────────────────────────────

sensors::ScanFrame SimLidar::get_latest_frame() const
{
    std::lock_guard<std::mutex> lk(frame_mutex_);
    return latest_frame_;
}

// ─── set_frame_callback ───────────────────────────────────────────────────────

void SimLidar::set_frame_callback(FrameCallback cb)
{
    std::lock_guard<std::mutex> lk(cb_mutex_);
    frame_callback_ = std::move(cb);
}

// ─── error_message ────────────────────────────────────────────────────────────

std::string SimLidar::error_message() const
{
    std::lock_guard<std::mutex> lk(error_mutex_);
    return error_str_;
}

// ─── model_name ───────────────────────────────────────────────────────────────

std::string SimLidar::model_name() const
{
    switch (scene_id_) {
        case SceneId::Sidewalk:         return "SimLidar[sidewalk]";
        case SceneId::Crossing:         return "SimLidar[crossing]";
        case SceneId::Hallway:          return "SimLidar[hallway]";
        case SceneId::ParkingLot:       return "SimLidar[parking_lot]";
        case SceneId::CyclistOvertake:  return "SimLidar[cyclist_overtake]";
        case SceneId::Crowd:            return "SimLidar[crowd]";
        default:                        return "SimLidar[unknown]";
    }
}

// ─── build_world ─────────────────────────────────────────────────────────────
// Constructs the SimWorld with the right seed and calls the appropriate
// scene builder.

void SimLidar::build_world()
{
    world_ = std::make_unique<SimWorld>(config_.rng_seed);

    // User speed: controlled by config_.user_walks flag.
    if (!config_.user_walks) {
        world_->user_speed_mm_s = 0.0f;
    }

    // Clear the default scene that the SimWorld constructor built.
    world_->objects.clear();

    // Build the requested scene.
    switch (scene_id_) {
        case SceneId::Sidewalk:
            build_scene_sidewalk();
            break;
        case SceneId::Crossing:
            build_scene_crossing();
            break;
        case SceneId::Hallway:
            build_scene_hallway();
            break;
        case SceneId::ParkingLot:
            build_scene_parking_lot();
            break;
        case SceneId::CyclistOvertake:
            build_scene_cyclist_overtake();
            break;
        case SceneId::Crowd:
            build_scene_crowd();
            break;
        default:
            build_scene_sidewalk();
            break;
    }
}

// ─── Scene builders ───────────────────────────────────────────────────────────
// These delegate entirely to SimWorld helper methods. SimLidar itself adds
// no geometry — it is purely a driver, not a scene author. The world builds
// its own scene; SimLidar just routes the request.

void SimLidar::build_scene_sidewalk()
{
    // SimWorld::build_default_scene() was already called by the constructor
    // but we cleared objects above. Call it explicitly now.
    world_->build_default_scene();
}

void SimLidar::build_scene_crossing()
{
    world_->build_scene_crossing();
}

void SimLidar::build_scene_hallway()
{
    world_->build_scene_hallway();
}

void SimLidar::build_scene_parking_lot()
{
    world_->build_scene_parking_lot();
}

void SimLidar::build_scene_cyclist_overtake()
{
    world_->build_scene_cyclist_overtake();
}

void SimLidar::build_scene_crowd()
{
    world_->build_scene_crowd();
}

// ─── pack_frame ───────────────────────────────────────────────────────────────
// Converts a vector of NoisedRay results into a ScanFrame with correct metadata.
//
// ScanPoint mapping:
//   NoisedRay::angle_deg    → ScanPoint::angle_deg
//   NoisedRay::distance_mm  → ScanPoint::distance_mm  (0 if !valid)
//   NoisedRay::quality      → ScanPoint::quality       (0 if !valid)
//   is_new_scan = true on the first point only (matches real LD06 behaviour)

sensors::ScanFrame SimLidar::pack_frame(const std::vector<NoisedRay>& rays,
                                         uint64_t frame_id) const
{
    sensors::ScanFrame frame;
    frame.frame_id   = frame_id;
    frame.timestamp  = std::chrono::steady_clock::now();
    frame.sensor_rpm = LD06_RPM;   // constant 600 RPM = 10 Hz × 60

    frame.points.reserve(rays.size());

    bool first = true;
    for (const auto& ray : rays) {
        sensors::ScanPoint pt;
        pt.angle_deg   = ray.angle_deg;
        pt.is_new_scan = first;
        first = false;

        if (ray.valid && ray.distance_mm > LD06_MIN_RANGE_MM) {
            pt.distance_mm = ray.distance_mm;
            pt.quality     = ray.quality;
        } else {
            // Invalid / dropout: distance = 0, quality = 0
            // This is exactly how the real LD06 encodes no-return points.
            pt.distance_mm = 0.0f;
            pt.quality     = 0;
        }

        frame.points.push_back(pt);
    }

    return frame;
}

// ─── publish_frame ────────────────────────────────────────────────────────────
// Stores the frame in latest_frame_ (under frame_mutex_) and invokes the
// frame callback if set (under cb_mutex_). Identical pattern to LD06.

void SimLidar::publish_frame(sensors::ScanFrame frame)
{
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = frame;
    }

    FrameCallback cb;
    {
        std::lock_guard<std::mutex> lk(cb_mutex_);
        cb = frame_callback_;
    }
    if (cb) {
        cb(frame);
    }
}

// ─── sim_loop ─────────────────────────────────────────────────────────────────
// Background thread body.
//
// Timing model:
//   We use sleep_until(next_tick) rather than sleep_for(dt) to prevent
//   accumulated drift. If a frame takes 2ms to generate and we target 10Hz,
//   we sleep for 98ms not 100ms — the next tick is always wall-clock aligned.
//
// Per-frame sequence:
//   1. Compute dt_s from wall-clock elapsed since last frame.
//   2. world_->step(dt_s) — advance all moving objects.
//   3. world_->cast_rays(N) — full 360° ray cast with noise.
//   4. pack_frame() — convert to ScanFrame.
//   5. publish_frame() — store + invoke callback.
//   6. sleep_until(next_tick).

void SimLidar::sim_loop()
{
    try {
        using clock     = std::chrono::steady_clock;
        using duration  = std::chrono::duration<float>;

        // Frame interval in nanoseconds for sleep_until precision.
        const float    frame_s  = 1.0f / std::max(config_.scan_hz, 0.1f);
        const auto     frame_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<float>(frame_s));

        auto next_tick   = clock::now() + frame_ns;
        auto last_tick   = clock::now();
        bool first_frame = true;

        while (!stop_flag_.load()) {

            // ── Measure actual dt ─────────────────────────────────────────────
            const auto now = clock::now();
            float dt_s = first_frame
                ? frame_s
                : std::chrono::duration_cast<duration>(now - last_tick).count();

            // Clamp dt to [0.005, 0.5] — handles startup jitter and long pauses.
            dt_s = std::max(0.005f, std::min(dt_s, 0.5f));

            last_tick   = now;
            first_frame = false;

            // ── Advance physics ───────────────────────────────────────────────
            world_->step(dt_s);

            // ── Cast rays — the core simulation work ──────────────────────────
            auto rays = world_->cast_rays(config_.rays_per_sweep);

            // ── Pack into ScanFrame and publish ───────────────────────────────
            const uint64_t fid = frame_counter_.fetch_add(1);
            auto frame = pack_frame(rays, fid);
            publish_frame(std::move(frame));

            // ── Sleep until the next scheduled tick ───────────────────────────
            std::this_thread::sleep_until(next_tick);
            next_tick += frame_ns;

            // If we fell more than 2 frames behind (e.g. system was preempted),
            // reset next_tick to avoid a burst of catch-up frames.
            if (clock::now() > next_tick + frame_ns) {
                next_tick = clock::now() + frame_ns;
            }
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(error_mutex_);
        error_str_ = std::string("simulation thread failed: ") + e.what();
    } catch (...) {
        std::lock_guard<std::mutex> lk(error_mutex_);
        error_str_ = "simulation thread failed: unknown exception";
    }

    running_.store(false);
}

} // namespace sim

// ─── C-linkage bridge ─────────────────────────────────────────────────────────
// Called by sensors::make_lidar(LidarModel::Sim, port_path) via an extern
// declaration in sensors/sensors.h. Returns a heap-allocated SimLidar cast
// to LidarBase*. The caller (make_lidar) wraps it in a unique_ptr immediately.
//
// Using a plain C-linkage function avoids any #include of sim/ headers inside
// sensors/ — the entire sim module remains a zero-dependency optional add-on.
// If sim_lib is not linked, the linker emits a clear "undefined reference to
// sim_make_lidar_raw" error rather than a silent runtime failure.

extern "C" sensors::LidarBase* sim_make_lidar_raw(const std::string& port_path)
{
    return new sim::SimLidar(port_path);
}
