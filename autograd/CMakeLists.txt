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
# aaronnet — our custom C++ autograd engine.
# Compiled into a single static library consumed by prediction_lib.
#
# Provides:
#   Tensor / TensorPtr  — shared_ptr-based graph node with flat float storage
#   Linear, ReLU        — neural network layers with He initialisation
#   Sequential          — ordered layer container
#   Adam                — adaptive moment optimiser with gradient clipping
#   NoGradGuard         — RAII context that disables graph construction
#
# Include via:  #include "autograd/autograd.h"

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

target_compile_features(autograd_lib PUBLIC cxx_std_17)

# Math library — std::sqrt, std::exp, std::pow, etc.
if(UNIX)
    target_link_libraries(autograd_lib PUBLIC m)
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "autograd: defaulting build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
