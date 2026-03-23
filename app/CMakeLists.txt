cmake_minimum_required(VERSION 3.16)
project(app CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── smart_glasses executable ──────────────────────────────────────────────────
# The single binary that wires every module together:
#   sensors_lib    — LD06 / RPLidar A1 driver + serial port
#   perception_lib — OccupancyMap + DBSCAN clusterer + Kalman tracker
#   prediction_lib — TTC engine + aaronnet MLP risk predictor
#   audio_lib      — TTS engine (espeak-ng) + alert policy
#   agent_lib      — SceneBuilder + OpenAI client + AgentLoop
#   autograd_lib   — aaronnet C++ autograd engine (pulled in via prediction_lib)

add_executable(smart_glasses
    main.cpp
)

target_include_directories(smart_glasses
    PRIVATE
        # Each module's include directory — the libraries already expose these
        # as PUBLIC include dirs, but listing them here makes IDE indexing work
        # correctly when opening main.cpp standalone.
        ${CMAKE_CURRENT_SOURCE_DIR}/../sensors/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../perception/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../prediction/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../audio/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../agent/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../autograd/include
)

target_compile_features(smart_glasses PRIVATE cxx_std_17)

# ── Link all libraries ────────────────────────────────────────────────────────
# Link order matters on some linkers: list the most-dependent first.
# agent_lib  → audio_lib → prediction_lib → perception_lib → sensors_lib
#                        → autograd_lib

if(TARGET agent_lib AND TARGET audio_lib AND TARGET prediction_lib AND
   TARGET perception_lib AND TARGET sensors_lib AND TARGET autograd_lib)
    # Building from the root CMakeLists.txt — all targets already exist.
    target_link_libraries(smart_glasses
        PRIVATE
            agent_lib
            audio_lib
            prediction_lib
            perception_lib
            sensors_lib
            autograd_lib
    )
else()
    # Standalone build — bring in the full dependency chain via add_subdirectory.
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
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../agent
                     ${CMAKE_CURRENT_BINARY_DIR}/agent)

    target_link_libraries(smart_glasses
        PRIVATE
            agent_lib
            audio_lib
            prediction_lib
            perception_lib
            sensors_lib
            autograd_lib
    )
endif()

# ── Optional sim_lib ──────────────────────────────────────────────────────────
# Linked only when USE_SIM=ON is passed to cmake.
# sim_lib propagates -DUSE_SIM=1 to all consumers via its PUBLIC compile
# definition, which activates the #ifdef USE_SIM blocks in main.cpp.
# Real hardware builds (USE_SIM=OFF, the default) are completely unaffected —
# sim/ is not compiled, not linked, and contributes zero bytes to the binary.
#
# Enable:
#   cmake .. -DUSE_SIM=ON
#   make -j4
#   ./smart_glasses --sensor sim --scene sidewalk --verbose

if(TARGET sim_lib)
    target_link_libraries(smart_glasses PRIVATE sim_lib)
    target_include_directories(smart_glasses PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../sim/include
    )
    message(STATUS "app: sim_lib linked — simulation mode available")
    message(STATUS "app: run with: --sensor sim --scene <name>")
    message(STATUS "app: scenes: sidewalk crossing hallway parking_lot cyclist_overtake crowd")
endif()

# pthreads — required by TtsEngine, AgentLoop, sensor drivers.
find_package(Threads REQUIRED)
target_link_libraries(smart_glasses PRIVATE Threads::Threads)

# libcurl — required by OpenAIClient (pulled in via agent_lib PUBLIC link,
# but listed explicitly here so the linker sees it in all configurations).
find_package(CURL REQUIRED)
target_link_libraries(smart_glasses PRIVATE CURL::libcurl)

# Math library — std::sqrt, std::exp, std::log, etc.
if(UNIX)
    target_link_libraries(smart_glasses PRIVATE m)
endif()

# ── Release optimisations ─────────────────────────────────────────────────────
# On the Raspberry Pi, -O2 is the right tradeoff between compile time and
# runtime performance. -O3 with -march=native is safe if you only deploy to
# one Pi model.
#
# Uncomment the line below when deploying to a known Pi 4 target:
# target_compile_options(smart_glasses PRIVATE -O3 -march=native)

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "app: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ── Install target (optional) ─────────────────────────────────────────────────
# Installs the binary to /usr/local/bin when the user runs `make install`.
# On the Pi: sudo make install → sudo smart_glasses
install(TARGETS smart_glasses
    RUNTIME DESTINATION bin
)

# ── Summary message ───────────────────────────────────────────────────────────
message(STATUS
    "app: smart_glasses executable configured. "
    "Build with: cmake --build . --target smart_glasses")
