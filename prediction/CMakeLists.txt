cmake_minimum_required(VERSION 3.16)
project(prediction CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── prediction_lib (static) ───────────────────────────────────────────────────
# Compiles the two prediction components into one static library:
#   - TTCEngine      : quadratic TTC solver, CPA, trajectory projection,
#                      sector threat table
#   - RiskPredictor  : aaronnet MLP featurisation, inference, online Adam
#                      training, pseudo-labelling, checkpoint I/O
#
# Consumers link against prediction_lib and include via:
#   #include "prediction/prediction.h"

add_library(prediction_lib STATIC
    src/ttc_engine.cpp
    src/risk_predictor.cpp
)

target_include_directories(prediction_lib
    PUBLIC
        # prediction's own headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include

        # aaronnet autograd engine headers
        ${CMAKE_CURRENT_SOURCE_DIR}/../autograd/include

        # perception headers (TrackedObject, TrackedObjectList, PerceptionResult)
        ${CMAKE_CURRENT_SOURCE_DIR}/../perception/include

        # sensors headers (ScanFrame — pulled in transitively via perception)
        ${CMAKE_CURRENT_SOURCE_DIR}/../sensors/include
)

target_compile_features(prediction_lib PUBLIC cxx_std_17)

# ── Link dependencies ─────────────────────────────────────────────────────────

# aaronnet static library — provides Tensor, Sequential, Adam, etc.
# When building standalone (not from root CMake), we compile autograd_lib here.
# When building from the root CMake, autograd_lib is already a target.
if(TARGET autograd_lib)
    target_link_libraries(prediction_lib PUBLIC autograd_lib)
else()
    # Standalone build: bring in autograd as a subdirectory.
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../autograd
                     ${CMAKE_CURRENT_BINARY_DIR}/autograd)
    target_link_libraries(prediction_lib PUBLIC autograd_lib)
endif()

# perception_lib — provides TrackedObjectList, PerceptionResult, OccupancyMap.
if(TARGET perception_lib)
    target_link_libraries(prediction_lib PUBLIC perception_lib)
else()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../perception
                     ${CMAKE_CURRENT_BINARY_DIR}/perception)
    target_link_libraries(prediction_lib PUBLIC perception_lib)
endif()

# pthreads — needed transitively (perception + autograd use std::thread/mutex).
find_package(Threads REQUIRED)
target_link_libraries(prediction_lib PUBLIC Threads::Threads)

# Math library — std::sqrt, std::exp, std::log, std::isfinite, etc.
if(UNIX)
    target_link_libraries(prediction_lib PUBLIC m)
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "prediction: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
