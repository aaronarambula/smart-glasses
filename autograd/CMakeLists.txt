cmake_minimum_required(VERSION 3.16)
project(autograd CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── autograd_lib (static) ─────────────────────────────────────────────────────
# All engine source files compiled into a single static library.

add_library(autograd_lib STATIC
    src/ops.cpp
    src/tensor.cpp
    src/layers.cpp
    src/optimizer.cpp
)

target_include_directories(autograd_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# C++17 features used: std::optional, structured bindings, inline variables
# (thread_local inline in no_grad.h), if-constexpr, etc.
target_compile_features(autograd_lib PUBLIC cxx_std_17)

# ── example executable ────────────────────────────────────────────────────────

add_executable(example example.cpp)

target_link_libraries(example PRIVATE autograd_lib)

# On Linux/macOS, link math library explicitly just in case.
if(UNIX)
    target_link_libraries(example PRIVATE m)
endif()

# ── Optional: Release build defaults ─────────────────────────────────────────
# Set a default build type if none was specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
