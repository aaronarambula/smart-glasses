cmake_minimum_required(VERSION 3.16)
project(perception CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── perception_lib (static) ───────────────────────────────────────────────────
# Compiles the three perception components into one static library:
#   - OccupancyMap  : rolling log-odds 2-D grid with Bresenham ray-tracing
#   - Clusterer     : DBSCAN point clusterer
#   - Tracker       : Kalman filter multi-object tracker + Hungarian assignment
#
# Consumers link against perception_lib and include via:
#   #include "perception/perception.h"

add_library(perception_lib STATIC
    src/occupancy_map.cpp
    src/clusterer.cpp
    src/tracker.cpp
)

target_include_directories(perception_lib
    PUBLIC
        # perception's own headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # needs sensors/lidar_base.h for ScanFrame / ScanPoint
        ${CMAKE_CURRENT_SOURCE_DIR}/../sensors/include
)

target_compile_features(perception_lib PUBLIC cxx_std_17)

# pthreads — tracker uses std::chrono::steady_clock; no explicit thread usage
# in perception itself, but the parent app links threads so we pull it in here
# to keep link order simple on all platforms.
find_package(Threads REQUIRED)
target_link_libraries(perception_lib PUBLIC Threads::Threads)

# Math library — std::sqrt, std::exp, std::atan2, etc.
if(UNIX)
    target_link_libraries(perception_lib PUBLIC m)
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "perception: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
