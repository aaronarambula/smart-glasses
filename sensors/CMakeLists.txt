cmake_minimum_required(VERSION 3.16)
project(sensors CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── sensors_lib (static) ──────────────────────────────────────────────────────
# LiDAR sensor drivers compiled into a single static library.
# Consumed by perception_lib → prediction_lib → audio_lib → agent_lib → app.
#
# Provides:
#   LD06       — LDROBOT LD06 / LD19, UART 230400 baud, streams 47-byte packets
#   RPLidarA1  — Slamtec A1M8, USB-serial 115200 baud, request/response protocol
#   SerialPort — POSIX RAII serial port wrapper (termios, 8N1, select-based)
#   ScanPoint  — one distance + angle + quality measurement
#   ScanFrame  — one complete 360° sweep (~460 points at 10 Hz)
#   LidarBase  — abstract interface (open / start / stop / close / get_latest_frame)
#
# Include via:  #include "sensors/sensors.h"

add_library(sensors_lib STATIC
    rplidar_a1.cpp
    ld06.cpp
    LiDAR.cpp
)

target_include_directories(sensors_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(sensors_lib PUBLIC cxx_std_17)

# pthreads — LD06 and RPLidarA1 run background read threads
find_package(Threads REQUIRED)
target_link_libraries(sensors_lib PUBLIC Threads::Threads)

# Math library — std::sqrt, std::cos, std::sin in ScanPoint::to_cartesian
if(UNIX)
    target_link_libraries(sensors_lib PUBLIC m)
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "sensors: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
