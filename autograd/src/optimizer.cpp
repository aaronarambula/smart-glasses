// ─── optimizer.cpp ───────────────────────────────────────────────────────────
// The Adam optimizer is fully defined inline in include/autograd/optimizer.h.
// This file exists to satisfy the CMake target sources list and to provide a
// natural place for future optimizers (SGD, AdaGrad, RMSProp, etc.) that are
// too large to live in a header.

#include "autograd/optimizer.h"

// Nothing to define here yet — see optimizer.h for the full Adam implementation.