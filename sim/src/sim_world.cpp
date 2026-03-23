// ─── sim_world.cpp ────────────────────────────────────────────────────────────
// SimWorld scene builders — all six outdoor environments.
// See include/sim/sim_world.h for the full design notes and geometry spec.
//
// Every scene is designed to exercise a different part of the pipeline:
//   sidewalk        → nominal operation, mixed static + dynamic obstacles
//   crossing        → TTC prediction, single high-urgency event
//   hallway         → narrow passage, bilateral wall detection, head-on
//   parking_lot     → rectangular obstacles, slow moving vehicle, CPA
//   cyclist_overtake→ rear-sector detection, high closing speed
//   crowd           → many objects, Hungarian assignment stress, glass dropout

#include "sim/sim_world.h"

#include <cmath>
#include <stdexcept>

namespace sim {

// ─── SimWorld helpers ─────────────────────────────────────────────────────────

void SimWorld::add_pedestrian(Vec2 pos, Vec2 vel, const std::string& label)
{
    objects.push_back(std::make_unique<MovingPedestrian>(
        alloc_id(), pos, vel, Material::Clothing, label));
}

void SimWorld::add_cyclist(Vec2 pos, Vec2 vel, const std::string& label)
{
    objects.push_back(std::make_unique<MovingCyclist>(
        alloc_id(), pos, vel, label));
}

void SimWorld::add_parked_car(Vec2 pos, float heading_rad, const std::string& label)
{
    // Standard car: ~4500mm long × 1900mm wide
    objects.push_back(std::make_unique<RectObject>(
        alloc_id(), pos,
        /*width_mm=*/4500.0f, /*depth_mm=*/1900.0f,
        heading_rad,
        Material::MetalShiny, label));
}

void SimWorld::add_wall(Vec2 p0, Vec2 p1, Material mat, const std::string& label)
{
    objects.push_back(std::make_unique<LineSegmentObject>(
        alloc_id(), p0, p1, mat, label));
}

void SimWorld::add_pole(Vec2 pos, float radius_mm, const std::string& label)
{
    objects.push_back(std::make_unique<CircleObject>(
        alloc_id(), pos, radius_mm, Material::MetalMatte, label));
}

void SimWorld::reset(uint32_t new_seed)
{
    objects.clear();
    next_id_ = 1;
    if (new_seed != 0) {
        rng = std::mt19937(new_seed);
    }
    build_default_scene();
}

// ─── build_default_scene ─────────────────────────────────────────────────────
// Delegates to the sidewalk scene (the most representative default).

void SimWorld::build_default_scene()
{
    // Sidewalk is the default — the most representative outdoor environment.
    // Scenes built once at construction; dynamic objects evolve via step().

    // ── Building facade — left side of the sidewalk ───────────────────────────
    // Runs along Y = +1800mm (1.8m to the left), extending far in X direction.
    // The user walks along the sidewalk parallel to this wall.
    add_wall(Vec2{-8000.0f,  1800.0f},
             Vec2{ 20000.0f, 1800.0f},
             Material::Brick, "building_facade");

    // ── Curb / kerb — right side of the sidewalk ─────────────────────────────
    // Low concrete barrier at Y = -1600mm (1.6m to the right).
    add_wall(Vec2{-8000.0f, -1600.0f},
             Vec2{ 20000.0f,-1600.0f},
             Material::Concrete, "curb");

    // ── Parked cars along the right kerb ─────────────────────────────────────
    // Three cars parked end-to-end, rear wheel at the kerb.
    // Car centre is ~950mm from the kerb (half of 1900mm depth).
    add_parked_car(Vec2{2000.0f, -2550.0f},  0.0f, "parked_car_1");
    add_parked_car(Vec2{6700.0f, -2550.0f},  0.0f, "parked_car_2");
    add_parked_car(Vec2{11400.0f,-2550.0f},  0.0f, "parked_car_3");

    // ── Utility / lamp poles — left side, recessed into building setback ──────
    // 120mm diameter steel poles, spaced ~8m apart.
    add_pole(Vec2{0.0f,    1400.0f}, 60.0f, "lamp_pole_1");
    add_pole(Vec2{8000.0f, 1400.0f}, 60.0f, "lamp_pole_2");
    add_pole(Vec2{16000.0f,1400.0f}, 60.0f, "lamp_pole_3");

    // ── Street tree — right side, trunk near kerb ─────────────────────────────
    // 200mm radius trunk (mature urban tree), foliage returns are noisy.
    objects.push_back(std::make_unique<CircleObject>(
        alloc_id(),
        Vec2{5000.0f, -1300.0f},
        200.0f,
        Material::Foliage,
        "tree_trunk_1"));

    // ── Bench — along building facade ─────────────────────────────────────────
    // 1800mm × 500mm concrete bench.
    objects.push_back(std::make_unique<RectObject>(
        alloc_id(),
        Vec2{3500.0f, 1450.0f},
        1800.0f, 500.0f,
        0.0f,
        Material::Concrete, "bench_1"));

    // ── Pedestrians ───────────────────────────────────────────────────────────
    // P1: walking the same direction as the user, slightly slower (0.9 m/s).
    //     Starts 8m ahead, centre of the sidewalk.
    add_pedestrian(Vec2{8000.0f,   0.0f},
                   Vec2{900.0f,    0.0f},
                   "pedestrian_same_dir");

    // P2: walking toward the user (opposite direction, 1.1 m/s).
    //     Starts 15m ahead, slightly to the left.
    add_pedestrian(Vec2{15000.0f, 400.0f},
                   Vec2{-1100.0f, 0.0f},
                   "pedestrian_oncoming");

    // ── Bollard pair flanking a shop entrance ─────────────────────────────────
    // 150mm radius steel bollards at Y ≈ +700mm, blocking direct wall-hugging.
    add_pole(Vec2{4800.0f, 700.0f},  75.0f, "bollard_1");
    add_pole(Vec2{4800.0f, 1100.0f}, 75.0f, "bollard_2");
}

// ─── build_scene_crossing ────────────────────────────────────────────────────
// An intersection approach. The user is walking toward a crossing.
// A pedestrian crosses from right to left at 1.4 m/s.
// A car is parked across the street (far side of the crossing).
// Curb lines transition to a dropped kerb at the crossing point.

void SimWorld::build_scene_crossing()
{
    // Sidewalk this side (approaching the crossing)
    add_wall(Vec2{-5000.0f,  1800.0f},
             Vec2{  5000.0f, 1800.0f},
             Material::Brick, "building_left");

    add_wall(Vec2{-5000.0f, -1600.0f},
             Vec2{  5000.0f,-1600.0f},
             Material::Concrete, "curb_near");

    // Road surface starts at X = 5000mm (end of the sidewalk)
    // Far kerb / pavement edge at X = 12000mm
    add_wall(Vec2{12000.0f, -4000.0f},
             Vec2{12000.0f,  4000.0f},
             Material::Concrete, "far_pavement_edge");

    // Building across the street (far side)
    add_wall(Vec2{12000.0f, -4000.0f},
             Vec2{25000.0f, -4000.0f},
             Material::Brick, "far_building");
    add_wall(Vec2{12000.0f,  4000.0f},
             Vec2{25000.0f,  4000.0f},
             Material::Brick, "far_building_right");

    // Crossing tactile paving strip (low friction surface at the kerb dip)
    // Modelled as a short wall stub at the kerb gap — not physically blocking
    // but affects scan density at that bearing.

    // Parked car far side of the road, blocking some rays.
    add_parked_car(Vec2{16000.0f, -2600.0f}, 0.0f, "parked_car_far");

    // Traffic light pole on the right corner.
    add_pole(Vec2{5200.0f, -1700.0f}, 60.0f, "traffic_light_pole");

    // ── The crossing pedestrian ───────────────────────────────────────────────
    // Starts at X = 7000mm, Y = -3500mm (to the right of the user's path),
    // moving left at 1.4 m/s. Will cross the user's forward path at
    // approximately t = 2.5s (when user closes from 7m at 1.2 m/s and
    // pedestrian reaches Y = 0 from Y = -3500 at 1.4 m/s → t ≈ 2.5s).
    add_pedestrian(Vec2{7000.0f, -3500.0f},
                   Vec2{   0.0f,  1400.0f},
                   "crossing_pedestrian");

    // A second person waiting at the far side (stationary).
    add_pedestrian(Vec2{13000.0f, 800.0f},
                   Vec2{0.0f, 0.0f},
                   "waiting_pedestrian");
}

// ─── build_scene_hallway ─────────────────────────────────────────────────────
// A narrow corridor: 1.4m wide (700mm either side of the user path).
// A pedestrian is walking toward the user from 10m away.
// This scene generates sustained WARNING → DANGER as the gap closes.

void SimWorld::build_scene_hallway()
{
    // Left wall: Y = +700mm, runs the full length.
    add_wall(Vec2{-2000.0f,  700.0f},
             Vec2{ 20000.0f, 700.0f},
             Material::Concrete, "hallway_wall_left");

    // Right wall: Y = -700mm.
    add_wall(Vec2{-2000.0f, -700.0f},
             Vec2{ 20000.0f,-700.0f},
             Material::Concrete, "hallway_wall_right");

    // End wall at X = 20000mm (20m — effectively unreachable in a short sim).
    add_wall(Vec2{20000.0f, -700.0f},
             Vec2{20000.0f,  700.0f},
             Material::Concrete, "hallway_end_wall");

    // ── Oncoming pedestrian ───────────────────────────────────────────────────
    // Starts 10m ahead, walking toward the user at 1.0 m/s.
    // Combined closing speed: 1.2 (user) + 1.0 (ped) = 2.2 m/s
    // → collision in ~4.5 seconds from start.
    add_pedestrian(Vec2{10000.0f, 50.0f},   // slight lateral offset
                   Vec2{-1000.0f,  0.0f},
                   "oncoming_pedestrian");

    // A door recess on the right at X = 6000mm (creates a brief gap in the
    // right wall — simulates a doorway).
    // We model it by NOT placing the wall in that segment; instead we add
    // two wall segments with a gap.
    objects.clear();   // clear the walls we just added, rebuild with gap
    next_id_ = 1;

    // Left wall continuous.
    add_wall(Vec2{-2000.0f,  700.0f},
             Vec2{ 20000.0f, 700.0f},
             Material::Concrete, "hallway_wall_left");

    // Right wall: two segments with a 1200mm doorway gap at X = [5400, 6600].
    add_wall(Vec2{-2000.0f, -700.0f},
             Vec2{  5400.0f,-700.0f},
             Material::Concrete, "right_wall_near");
    add_wall(Vec2{  6600.0f, -700.0f},
             Vec2{ 20000.0f, -700.0f},
             Material::Concrete, "right_wall_far");

    // Door frame (depth of wall = 200mm).
    add_wall(Vec2{5400.0f, -700.0f},
             Vec2{5400.0f, -900.0f},
             Material::Concrete, "door_jamb_near");
    add_wall(Vec2{6600.0f, -700.0f},
             Vec2{6600.0f, -900.0f},
             Material::Concrete, "door_jamb_far");

    // End wall.
    add_wall(Vec2{20000.0f, -700.0f},
             Vec2{20000.0f,  700.0f},
             Material::Concrete, "hallway_end_wall");

    // The oncoming pedestrian.
    add_pedestrian(Vec2{10000.0f, 50.0f},
                   Vec2{-1000.0f,  0.0f},
                   "oncoming_pedestrian");
}

// ─── build_scene_parking_lot ─────────────────────────────────────────────────
// An open parking lot: a 4×2 grid of parked cars with driving lanes.
// A slow-moving car (0.5 m/s) crosses the lane from the right.

void SimWorld::build_scene_parking_lot()
{
    // Perimeter wall (low concrete barrier, 300mm tall — models curb stops).
    // The lot is 30m × 20m.
    add_wall(Vec2{-2000.0f, -10000.0f},
             Vec2{30000.0f, -10000.0f},
             Material::Concrete, "lot_perimeter_bottom");
    add_wall(Vec2{-2000.0f,  10000.0f},
             Vec2{30000.0f,  10000.0f},
             Material::Concrete, "lot_perimeter_top");
    add_wall(Vec2{30000.0f, -10000.0f},
             Vec2{30000.0f,  10000.0f},
             Material::Concrete, "lot_perimeter_far");

    // Row 1 of parked cars: Y = +3500mm (3.5m left), 4 cars spaced 5m apart.
    for (int i = 0; i < 4; ++i) {
        add_parked_car(Vec2{2000.0f + i * 5000.0f, 3500.0f},
                       PI * 0.5f,   // parked perpendicular to lane
                       "row1_car_" + std::to_string(i));
    }

    // Row 2 of parked cars: Y = -3500mm (3.5m right), 4 cars spaced 5m apart.
    for (int i = 0; i < 4; ++i) {
        add_parked_car(Vec2{2000.0f + i * 5000.0f, -3500.0f},
                       PI * 0.5f,
                       "row2_car_" + std::to_string(i));
    }

    // ── Slow-moving car ───────────────────────────────────────────────────────
    // Starts at X = 12000mm, Y = -8000mm (far right of the lot),
    // moving left at 500 mm/s. Will cross the user's forward path at Y ≈ 0
    // around t = 16s. Simulates a car pulling out of a bay.
    // We model the moving car as a RectObject — but RectObject doesn't move.
    // Use a MovingPedestrian with a large radius as a proxy for the moving car.
    // (A proper MovingRect would be added in a future version.)
    // For now: add it as a large circular obstacle (3m radius ≈ car width).
    objects.push_back(std::make_unique<MovingPedestrian>(
        alloc_id(),
        Vec2{12000.0f, -8000.0f},
        Vec2{0.0f, 500.0f},         // moving left at 0.5 m/s
        Material::MetalShiny,
        "slow_car"));

    // Override its radius to something car-sized.
    // (MovingPedestrian::radius_mm is public.)
    static_cast<MovingPedestrian*>(objects.back().get())->radius_mm = 1000.0f;
}

// ─── build_scene_cyclist_overtake ────────────────────────────────────────────
// A shared-use path (cyclists + pedestrians).
// A cyclist approaches from behind-left at 6 m/s and overtakes.
// The user does not see this coming until the cyclist enters the rear sectors.

void SimWorld::build_scene_cyclist_overtake()
{
    // Path edges: 3m wide shared path.
    add_wall(Vec2{-10000.0f,  1500.0f},
             Vec2{  20000.0f, 1500.0f},
             Material::Concrete, "path_left_edge");
    add_wall(Vec2{-10000.0f, -1500.0f},
             Vec2{  20000.0f,-1500.0f},
             Material::Concrete, "path_right_edge");

    // Some hedge on the left side (high dropout — foliage).
    add_wall(Vec2{-10000.0f, 2000.0f},
             Vec2{  20000.0f, 2000.0f},
             Material::Foliage, "hedge");

    // A park bench far ahead (stationary, won't cause alert during overtake).
    objects.push_back(std::make_unique<RectObject>(
        alloc_id(),
        Vec2{14000.0f, 1100.0f},
        1800.0f, 500.0f,
        0.0f,
        Material::Concrete, "bench"));

    // ── Cyclist ───────────────────────────────────────────────────────────────
    // Starts 8m BEHIND the user (X = -8000mm), positioned slightly to the
    // left (Y = +600mm), moving forward at 6 m/s.
    // At 6 m/s vs user at 1.2 m/s, relative speed = 4.8 m/s.
    // The cyclist is 8m behind → will reach the user in 8000/4800 ≈ 1.67s.
    // This tests the rear-sector tracking and DANGER from behind.
    add_cyclist(Vec2{-8000.0f, 600.0f},
                Vec2{ 6000.0f,   0.0f},
                "overtaking_cyclist");

    // A second, slower cyclist ahead (confirms forward detection still works).
    add_cyclist(Vec2{12000.0f, -200.0f},
                Vec2{  800.0f,    0.0f},
                "slow_cyclist_ahead");
}

// ─── build_scene_crowd ────────────────────────────────────────────────────────
// Busy street: 6 pedestrians in different directions, glass storefronts,
// bus stop shelter. Tests Hungarian assignment under high object count.

void SimWorld::build_scene_crowd()
{
    // Building facades both sides — left has glass (storefront), right is brick.
    add_wall(Vec2{-5000.0f,  2200.0f},
             Vec2{ 20000.0f, 2200.0f},
             Material::Glass, "storefront_glass");

    add_wall(Vec2{-5000.0f, -2200.0f},
             Vec2{ 20000.0f,-2200.0f},
             Material::Brick, "right_building");

    // Bus stop shelter: a small RectObject on the right (glass + metal frame).
    // The glass sides cause high dropout — typical real-world challenge.
    objects.push_back(std::make_unique<RectObject>(
        alloc_id(),
        Vec2{5000.0f, -1800.0f},
        2200.0f, 400.0f,   // 2.2m wide, 0.4m deep shelter
        0.0f,
        Material::Glass,
        "bus_shelter"));

    // Metal post at each end of the shelter.
    add_pole(Vec2{3900.0f, -1800.0f}, 40.0f, "shelter_post_near");
    add_pole(Vec2{6100.0f, -1800.0f}, 40.0f, "shelter_post_far");

    // ── Six pedestrians in varied directions ──────────────────────────────────

    // P1: same direction, slow (0.8 m/s), 6m ahead, slightly right.
    add_pedestrian(Vec2{6000.0f, -300.0f},
                   Vec2{800.0f,    0.0f},
                   "ped_same_slow");

    // P2: opposite direction (oncoming), 12m ahead, centre.
    add_pedestrian(Vec2{12000.0f, 100.0f},
                   Vec2{-1200.0f,  0.0f},
                   "ped_oncoming");

    // P3: crossing left to right, 8m ahead.
    add_pedestrian(Vec2{8000.0f, 1800.0f},
                   Vec2{0.0f, -1000.0f},
                   "ped_cross_l2r");

    // P4: crossing right to left, 4m ahead — this one will be urgent.
    add_pedestrian(Vec2{4000.0f, -1800.0f},
                   Vec2{0.0f,  1200.0f},
                   "ped_cross_r2l");

    // P5: diagonal — walking forward and slightly left.
    add_pedestrian(Vec2{9000.0f, -600.0f},
                   Vec2{900.0f,  200.0f},
                   "ped_diagonal");

    // P6: stationary pedestrian (waiting for bus), at the shelter.
    add_pedestrian(Vec2{5000.0f, -1600.0f},
                   Vec2{0.0f, 0.0f},
                   "ped_waiting");

    // A lamp post mid-pavement.
    add_pole(Vec2{2500.0f, 0.0f}, 60.0f, "lamp_pole");

    // Rubbish bin near the bus shelter.
    objects.push_back(std::make_unique<CircleObject>(
        alloc_id(),
        Vec2{6400.0f, -1900.0f},
        200.0f,
        Material::MetalMatte,
        "bin"));
}

} // namespace sim