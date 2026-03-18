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
# All sensor driver source files compiled into a single static library.
# Consumers link against sensors_lib and include via "sensors/sensors.h".

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

# pthreads — required for std::thread on Linux / Raspberry Pi OS
find_package(Threads REQUIRED)
target_link_libraries(sensors_lib PUBLIC Threads::Threads)

# Math library (std::sqrt, std::cos, etc.) — implicit on most systems but
# explicitly linked on some older ARM Linux toolchains.
if(UNIX)
    target_link_libraries(sensors_lib PUBLIC m)
endif()

# ── Optional: standalone test executable ─────────────────────────────────────
# Build a self-contained lidar_test binary that exercises the sensor directly
# without the rest of the smart-glasses pipeline.
#
# Enable with:
#   cmake .. -DLIDAR_BUILD_STANDALONE=ON
#
# Usage after building:
#   ./lidar_test rplidar /dev/ttyUSB0
#   ./lidar_test ld06    /dev/ttyAMA0

option(LIDAR_BUILD_STANDALONE
    "Build the standalone lidar_test executable for sensor debugging"
    OFF)

if(LIDAR_BUILD_STANDALONE)
    add_executable(lidar_test
        rplidar_a1.cpp
        ld06.cpp
        LiDAR.cpp
    )

    target_include_directories(lidar_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_compile_features(lidar_test PRIVATE cxx_std_17)

    # LIDAR_STANDALONE enables the main() in LiDAR.cpp
    target_compile_definitions(lidar_test PRIVATE LIDAR_STANDALONE)

    target_link_libraries(lidar_test PRIVATE Threads::Threads)

    if(UNIX)
        target_link_libraries(lidar_test PRIVATE m)
    endif()

    message(STATUS "sensors: lidar_test standalone executable enabled")
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "sensors: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
