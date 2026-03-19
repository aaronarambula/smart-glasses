cmake_minimum_required(VERSION 3.16)
project(smart_glasses CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ── Global compiler warnings ──────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Optional Pi-native optimisation ──────────────────────────────────────────
# Uncomment when building directly on a Raspberry Pi 4 for maximum performance.
# option(PI_NATIVE "Optimise for the host Raspberry Pi CPU" OFF)
# if(PI_NATIVE)
#     add_compile_options(-O3 -march=native -mtune=native)
#     message(STATUS "Pi-native optimisation enabled (-O3 -march=native)")
# endif()

# ── External dependencies (checked once at the top level) ─────────────────────

# pthreads — sensors, perception, audio, agent all use std::thread
find_package(Threads REQUIRED)

# libcurl — OpenAI HTTPS client in agent/
find_package(CURL REQUIRED)

# ── Submodules — built in dependency order ────────────────────────────────────
#
# Dependency graph (each module only sees modules above it):
#
#   autograd/      ← aaronnet C++ autograd engine (no project deps)
#   sensors/       ← LD06 + RPLidar A1 drivers    (no project deps)
#   perception/    ← OccupancyMap, DBSCAN, Kalman  (← sensors)
#   prediction/    ← TTC engine, aaronnet MLP      (← perception, autograd)
#   audio/         ← TTS engine, alert policy      (← prediction)
#   agent/         ← SceneBuilder, OpenAI, loop    (← audio, prediction, perception)
#   app/           ← main.cpp entry point          (← all of the above)

add_subdirectory(autograd)
add_subdirectory(sensors)
add_subdirectory(perception)
add_subdirectory(prediction)
add_subdirectory(audio)
add_subdirectory(agent)
add_subdirectory(app)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║           Smart Glasses — build configured               ║")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  Modules : autograd  sensors  perception                 ║")
message(STATUS "║            prediction  audio  agent  app                 ║")
message(STATUS "║  Target  : smart_glasses                                 ║")
message(STATUS "║  Type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  Build:   cmake --build . --parallel                     ║")
message(STATUS "║  Run:     ./app/smart_glasses --sensor ld06              ║")
message(STATUS "║           ./app/smart_glasses --help                     ║")
message(STATUS "║  Deploy:  sudo make install  →  /usr/local/bin/          ║")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  Deps (Raspberry Pi OS):                                 ║")
message(STATUS "║    sudo apt install libcurl4-openssl-dev espeak-ng       ║")
message(STATUS "║  API key:                                                ║")
message(STATUS "║    export OPENAI_API_KEY=sk-...                          ║")
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "")
