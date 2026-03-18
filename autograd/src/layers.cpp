// ─── layers.cpp ──────────────────────────────────────────────────────────────
// All layer logic (Linear, ReLU, Softmax, Sequential) is defined inline in
// include/autograd/layers.h — no separate compilation unit is needed.
// This file exists solely to satisfy the CMake target sources list and to
// provide a natural place for any future non-trivial layer implementations
// (e.g. BatchNorm, Dropout, Conv2d) that are too large to live in a header.

#include "autograd/layers.h"

// Nothing to define here yet — see layers.h for all implementations.