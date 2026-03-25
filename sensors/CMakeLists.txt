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
    ultrasonic_fallback.cpp
    camera_fallback.cpp
    camera_classifier.cpp
    LiDAR.cpp
)

target_include_directories(sensors_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(sensors_lib PUBLIC cxx_std_17)

if(TARGET autograd_lib)
    target_link_libraries(sensors_lib PRIVATE autograd_lib)
endif()

# pthreads — LD06 and RPLidarA1 run background read threads
find_package(Threads REQUIRED)
target_link_libraries(sensors_lib PUBLIC Threads::Threads)

# Math library — std::sqrt, std::cos, std::sin in ScanPoint::to_cartesian
if(UNIX)
    target_link_libraries(sensors_lib PUBLIC m)
endif()

# Prefer libgpiod on Linux for modern Raspberry Pi GPIO access. Fall back to
# legacy sysfs GPIO only when libgpiod is unavailable.
if(UNIX AND NOT APPLE)
    include(CheckSymbolExists)
    find_path(GPIOD_INCLUDE_DIR gpiod.h)
    find_library(GPIOD_LIBRARY gpiod)
    if(GPIOD_INCLUDE_DIR AND GPIOD_LIBRARY)
        set(CMAKE_REQUIRED_INCLUDES ${GPIOD_INCLUDE_DIR})
        set(CMAKE_REQUIRED_LIBRARIES ${GPIOD_LIBRARY})
        check_symbol_exists(gpiod_chip_request_lines "gpiod.h" HAVE_LIBGPIOD_V2_API)
        check_symbol_exists(gpiod_chip_get_line "gpiod.h" HAVE_LIBGPIOD_V1_API)

        if(HAVE_LIBGPIOD_V2_API OR HAVE_LIBGPIOD_V1_API)
            target_include_directories(sensors_lib PRIVATE ${GPIOD_INCLUDE_DIR})
            target_link_libraries(sensors_lib PRIVATE ${GPIOD_LIBRARY})
            if(HAVE_LIBGPIOD_V2_API)
                target_compile_definitions(sensors_lib PRIVATE HAVE_LIBGPIOD_V2=1)
                message(STATUS "sensors: libgpiod v2 API found — ultrasonic fallback will use GPIO character device API")
            else()
                target_compile_definitions(sensors_lib PRIVATE HAVE_LIBGPIOD_V1=1)
                message(STATUS "sensors: libgpiod v1 API found — ultrasonic fallback will use legacy libgpiod line API")
            endif()
        else()
            message(STATUS "sensors: libgpiod found but usable GPIO API symbols were not detected; falling back to legacy sysfs GPIO")
        endif()
    else()
        message(STATUS "sensors: libgpiod not found — ultrasonic fallback will use legacy sysfs GPIO if available")
    endif()
endif()

# Optional OpenCV for the camera fallback sensor.
find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
if(OpenCV_FOUND)
    target_include_directories(sensors_lib PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(sensors_lib PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(sensors_lib PRIVATE HAVE_OPENCV=1)
    message(STATUS "sensors: OpenCV found — camera fallback sensor enabled")
else()
    message(STATUS "sensors: OpenCV not found — camera fallback will report a clear runtime error if selected")
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "sensors: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
