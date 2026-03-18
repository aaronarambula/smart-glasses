#pragma once

// ─── no_grad.h ───────────────────────────────────────────────────────────────
// Thread-local gradient-enable flag and RAII guard matching Python's
// @contextmanager no_grad() pattern.

namespace autograd {

// One flag per thread — safe for multi-threaded data loaders.
inline thread_local bool enable_grad = true;

// RAII guard: disables gradient tracking for its lifetime, then restores.
//
//   {
//       NoGradGuard g;
//       auto out = model.forward(x);   // no graph is built
//   }
//   // gradient tracking restored here
struct NoGradGuard {
    bool prev;

    NoGradGuard() : prev(enable_grad) {
        enable_grad = false;
    }

    ~NoGradGuard() {
        enable_grad = prev;
    }

    // Non-copyable, non-movable.
    NoGradGuard(const NoGradGuard&)            = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&)                 = delete;
    NoGradGuard& operator=(NoGradGuard&&)      = delete;
};

} // namespace autograd