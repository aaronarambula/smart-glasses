cmake_minimum_required(VERSION 3.16)
project(audio CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── audio_lib (static) ────────────────────────────────────────────────────────
# Compiles the two audio components into one static library:
#   - TtsEngine   : non-blocking espeak-ng wrapper with priority queue,
#                   POSIX fork/exec subprocess management, deduplication
#   - AlertPolicy : rate-limited alert policy consuming FullPrediction,
#                   natural-language utterance builder, escalation bypass
#
# Consumers link against audio_lib and include via:
#   #include "audio/audio.h"

add_library(audio_lib STATIC
    src/tts_engine.cpp
    src/alert_policy.cpp
)

target_include_directories(audio_lib
    PUBLIC
        # audio's own headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include

        # prediction headers (FullPrediction, RiskLevel, TTCResult, etc.)
        ${CMAKE_CURRENT_SOURCE_DIR}/../prediction/include

        # perception headers (pulled in transitively via prediction)
        ${CMAKE_CURRENT_SOURCE_DIR}/../perception/include

        # sensors headers (pulled in transitively via perception)
        ${CMAKE_CURRENT_SOURCE_DIR}/../sensors/include

        # autograd headers (pulled in transitively via prediction)
        ${CMAKE_CURRENT_SOURCE_DIR}/../autograd/include
)

target_compile_features(audio_lib PUBLIC cxx_std_17)

# ── Link dependencies ─────────────────────────────────────────────────────────

# prediction_lib — provides FullPrediction, RiskLevel, TTCFrame, etc.
if(TARGET prediction_lib)
    target_link_libraries(audio_lib PUBLIC prediction_lib)
else()
    # Standalone build: bring in the full dependency chain.
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../autograd
                     ${CMAKE_CURRENT_BINARY_DIR}/autograd)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sensors
                     ${CMAKE_CURRENT_BINARY_DIR}/sensors)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../perception
                     ${CMAKE_CURRENT_BINARY_DIR}/perception)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../prediction
                     ${CMAKE_CURRENT_BINARY_DIR}/prediction)
    target_link_libraries(audio_lib PUBLIC prediction_lib)
endif()

# pthreads — TtsEngine uses std::thread + std::mutex + std::condition_variable.
find_package(Threads REQUIRED)
target_link_libraries(audio_lib PUBLIC Threads::Threads)

# Math library — std::ceil, std::round, std::floor, std::isfinite, etc.
if(UNIX)
    target_link_libraries(audio_lib PUBLIC m)
endif()

# ── espeak-ng runtime check (informational only) ──────────────────────────────
# We don't link against espeak-ng — we invoke it as a subprocess via fork/exec.
# This find_program call just emits a warning at configure time if espeak-ng
# is not found on the build host, so the developer knows to install it.
find_program(ESPEAK_NG_EXECUTABLE espeak-ng)
if(ESPEAK_NG_EXECUTABLE)
    message(STATUS "audio: espeak-ng found at ${ESPEAK_NG_EXECUTABLE}")
else()
    message(WARNING
        "audio: espeak-ng not found on PATH. "
        "Install it on the Raspberry Pi with: sudo apt install espeak-ng")
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "audio: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
