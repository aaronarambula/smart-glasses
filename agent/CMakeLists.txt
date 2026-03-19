cmake_minimum_required(VERSION 3.16)
project(agent CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── agent_lib (static) ────────────────────────────────────────────────────────
# Compiles the three agent components into one static library:
#   - SceneBuilder   : FullPrediction → compact JSON for GPT context
#   - OpenAIClient   : async HTTPS POST to GPT-4o via libcurl, retry, timeout
#   - AgentLoop      : background query thread with gating, change detection,
#                      response filtering, and AudioSystem delivery
#
# Consumers link against agent_lib and include via:
#   #include "agent/agent.h"
#
# Required system packages (Raspberry Pi OS):
#   sudo apt install libcurl4-openssl-dev
#
# Required environment variable at runtime:
#   export OPENAI_API_KEY="sk-..."
#   The key is read from the environment — never hardcoded.

add_library(agent_lib STATIC
    src/scene_builder.cpp
    src/openai_client.cpp
    src/agent_loop.cpp
)

target_include_directories(agent_lib
    PUBLIC
        # agent's own headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include

        # audio headers (AudioSystem, TtsEngine — AgentLoop delivers advice here)
        ${CMAKE_CURRENT_SOURCE_DIR}/../audio/include

        # prediction headers (FullPrediction, RiskLevel, TTCFrame, etc.)
        ${CMAKE_CURRENT_SOURCE_DIR}/../prediction/include

        # perception headers (PerceptionResult — forwarded into scene JSON)
        ${CMAKE_CURRENT_SOURCE_DIR}/../perception/include

        # sensors headers (pulled in transitively via perception)
        ${CMAKE_CURRENT_SOURCE_DIR}/../sensors/include

        # autograd headers (pulled in transitively via prediction)
        ${CMAKE_CURRENT_SOURCE_DIR}/../autograd/include
)

target_compile_features(agent_lib PUBLIC cxx_std_17)

# ── libcurl ───────────────────────────────────────────────────────────────────
# Required for HTTPS POST to the OpenAI API.
# find_package uses CMake's built-in FindCURL module.
# On Raspberry Pi: sudo apt install libcurl4-openssl-dev

find_package(CURL REQUIRED)
target_link_libraries(agent_lib PUBLIC CURL::libcurl)

# ── Link dependency chain ─────────────────────────────────────────────────────
# Build the full dependency chain when building standalone.
# When building from the root CMake, all targets are already defined.

if(TARGET audio_lib)
    target_link_libraries(agent_lib PUBLIC audio_lib)
else()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../autograd
                     ${CMAKE_CURRENT_BINARY_DIR}/autograd)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sensors
                     ${CMAKE_CURRENT_BINARY_DIR}/sensors)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../perception
                     ${CMAKE_CURRENT_BINARY_DIR}/perception)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../prediction
                     ${CMAKE_CURRENT_BINARY_DIR}/prediction)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../audio
                     ${CMAKE_CURRENT_BINARY_DIR}/audio)
    target_link_libraries(agent_lib PUBLIC audio_lib)
endif()

# pthreads — AgentLoop uses std::thread + std::mutex + std::condition_variable.
find_package(Threads REQUIRED)
target_link_libraries(agent_lib PUBLIC Threads::Threads)

# Math library — std::abs, std::isfinite, std::transform, etc.
if(UNIX)
    target_link_libraries(agent_lib PUBLIC m)
endif()

# ── API key runtime check (informational only) ────────────────────────────────
# We emit a configure-time note reminding the developer to set the key.
# The key is intentionally NOT checked here — it is read from the environment
# at runtime by OpenAIClient so it is never baked into the build system.
message(STATUS
    "agent: OpenAI API key is read from the environment variable "
    "OPENAI_API_KEY at runtime. "
    "Set it with: export OPENAI_API_KEY=\"sk-...\"")

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "agent: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
