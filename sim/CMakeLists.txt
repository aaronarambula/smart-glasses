cmake_minimum_required(VERSION 3.16)
project(sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── sim_lib (static) ──────────────────────────────────────────────────────────
# Optional simulation module. Enabled by passing -DUSE_SIM=ON to cmake.
#
# Provides:
#   SimWorld    — outdoor environment geometry + LD06 noise model
#   SimLidar    — drop-in LidarBase implementation driven by SimWorld
#   6 scenes    — sidewalk, crossing, hallway, parking_lot,
#                 cyclist_overtake, crowd
#
# Zero coupling to perception/, prediction/, audio/, agent/.
# Only depends on sensors/lidar_base.h (bottom of the dep stack).
#
# Consumers include via:
#   #include "sim/sim.h"
#
# Usage:
#   cmake .. -DUSE_SIM=ON
#   make -j4
#   ./app/smart_glasses --sensor sim --scene crossing --verbose

option(USE_SIM
    "Build the simulation module (SimLidar + SimWorld outdoor scenes)"
    OFF)

if(NOT USE_SIM)
    message(STATUS "sim: disabled (pass -DUSE_SIM=ON to enable)")
    return()
endif()

message(STATUS "sim: enabled — building sim_lib with 6 outdoor scenes")

add_library(sim_lib STATIC
    src/sim_world.cpp
    src/sim_lidar.cpp
)

target_include_directories(sim_lib
    PUBLIC
        # sim's own headers (SimWorld, SimLidar, sim.h umbrella)
        ${CMAKE_CURRENT_SOURCE_DIR}/include

        # sensors/lidar_base.h — the only project header sim depends on
        ${CMAKE_CURRENT_SOURCE_DIR}/../sensors/include
)

target_compile_features(sim_lib PUBLIC cxx_std_17)

# ── Link dependencies ─────────────────────────────────────────────────────────

# pthreads — SimLidar runs a background sim thread
find_package(Threads REQUIRED)
target_link_libraries(sim_lib PUBLIC Threads::Threads)

# Math library — std::cos, std::sin, std::sqrt, std::abs, std::exp, etc.
if(UNIX)
    target_link_libraries(sim_lib PUBLIC m)
endif()

# sensors_lib — SimLidar inherits LidarBase; we need the full sensors_lib
# for ScanFrame / ScanPoint definitions that are in lidar_base.h.
# When building from the root CMakeLists.txt, sensors_lib is already a target.
if(TARGET sensors_lib)
    target_link_libraries(sim_lib PUBLIC sensors_lib)
else()
    # Standalone build.
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sensors
                     ${CMAKE_CURRENT_BINARY_DIR}/sensors)
    target_link_libraries(sim_lib PUBLIC sensors_lib)
endif()

# ── Propagate USE_SIM flag to all consumers ────────────────────────────────────
# Any target that links sim_lib automatically gets -DUSE_SIM=1 so that
# conditional blocks in main.cpp (#ifdef USE_SIM) are compiled in.
target_compile_definitions(sim_lib PUBLIC USE_SIM=1)

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "sim: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "sim: sim_lib configured")
message(STATUS "sim: scenes available: sidewalk, crossing, hallway,")
message(STATUS "sim:                   parking_lot, cyclist_overtake, crowd")
message(STATUS "sim: usage: ./smart_glasses --sensor sim --scene sidewalk")
