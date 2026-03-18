#pragma once

// ─── tensor.h ────────────────────────────────────────────────────────────────
// Core Tensor type for the autograd engine.
// Every tensor is heap-allocated and reference-counted via shared_ptr so that
// the backward graph can hold onto inputs that might otherwise go out of scope.

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <cstddef>

namespace autograd {

// Forward declaration so TensorPtr can be used inside the class body.
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

// ─── Tensor ──────────────────────────────────────────────────────────────────

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // ── Storage ──────────────────────────────────────────────────────────────

    std::vector<float>  data;   // flat, row-major
    std::vector<size_t> shape;  // e.g. {rows, cols} or {cols} for 1-D bias

    // ── Gradient ─────────────────────────────────────────────────────────────

    bool                requires_grad;
    std::vector<float>  grad;   // same size as data; empty == "no grad yet"

    // ── Graph bookkeeping ─────────────────────────────────────────────────────

    std::vector<TensorPtr>   _prev;      // inputs that produced this tensor
    std::function<void()>    _backward;  // closure that accumulates grads into _prev

    // ── Constructors ──────────────────────────────────────────────────────────

    // Scalar (0-D stored as 1-element, shape = {1})
    explicit Tensor(float scalar, bool requires_grad = false);

    // From flat data + shape
    Tensor(std::vector<float> data, std::vector<size_t> shape,
           bool requires_grad = false);

    // Convenience: 2-D matrix
    Tensor(std::vector<float> data, size_t rows, size_t cols,
           bool requires_grad = false);

    // Non-copyable (graphs are always accessed through shared_ptr)
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&)                 = default;
    Tensor& operator=(Tensor&&)      = default;

    ~Tensor() = default;

    // ── Shape helpers ─────────────────────────────────────────────────────────

    size_t ndim()    const { return shape.size(); }
    size_t numel()   const { return data.size(); }
    size_t rows()    const { return shape.size() >= 2 ? shape[shape.size()-2] : 1; }
    size_t cols()    const { return shape.empty() ? 1 : shape.back(); }

    std::string shape_str() const;

    // ── Gradient helpers ──────────────────────────────────────────────────────

    // Returns true if grad has been allocated.
    bool has_grad() const { return !grad.empty(); }

    // Lazily allocates grad to zeros if not yet present; then adds `delta` into it.
    void accumulate_grad(const std::vector<float>& delta);

    // Zero out grad (keeps allocation).
    void zero_grad();

    // ── Backward ──────────────────────────────────────────────────────────────

    // Runs reverse-mode autodiff from this tensor as the root.
    // `initial_grad`: if empty, uses a vector of ones (suitable for scalar loss).
    void backward(std::vector<float> initial_grad = {});

    // ── Arithmetic ops (return new TensorPtr, build graph) ────────────────────

    // Element-wise add: supports bias broadcast (other.shape = {cols})
    TensorPtr operator+(const TensorPtr& other) const;
    TensorPtr operator+(float scalar)           const;

    // Element-wise multiply: supports broadcast
    TensorPtr operator*(const TensorPtr& other) const;
    TensorPtr operator*(float scalar)           const;

    // Matrix multiply: (M,K) @ (K,N) → (M,N)
    TensorPtr matmul(const TensorPtr& other) const;

    // Activations
    TensorPtr relu()    const;
    TensorPtr softmax() const;  // row-wise

    // Reduce all elements to a scalar
    TensorPtr sum() const;

    // ── Loss ─────────────────────────────────────────────────────────────────

    // Numerically stable cross-entropy loss.
    // logits: (M, C),  targets: integer class indices in [0, C)
    // Returns a scalar TensorPtr.
    static TensorPtr cross_entropy(const TensorPtr& logits,
                                   const std::vector<int>& targets);

    // ── Debug ─────────────────────────────────────────────────────────────────

    std::string repr() const;
};

// ─── Factory helpers ──────────────────────────────────────────────────────────

// Create a tensor and return a shared_ptr — use these instead of `new` /
// `make_shared` at call sites for brevity.

inline TensorPtr make_tensor(std::vector<float> data,
                              std::vector<size_t> shape,
                              bool requires_grad = false)
{
    return std::make_shared<Tensor>(std::move(data), std::move(shape),
                                    requires_grad);
}

inline TensorPtr make_tensor(std::vector<float> data,
                              size_t rows, size_t cols,
                              bool requires_grad = false)
{
    return std::make_shared<Tensor>(std::move(data), rows, cols, requires_grad);
}

inline TensorPtr make_scalar(float value, bool requires_grad = false)
{
    return std::make_shared<Tensor>(value, requires_grad);
}

// ─── Free operator overloads ──────────────────────────────────────────────────
// Allow natural syntax: a + b, a * b, a @ b (via matmul free fn)
// when both sides are TensorPtr.

inline TensorPtr operator+(const TensorPtr& a, const TensorPtr& b)
{
    return (*a) + b;
}

inline TensorPtr operator+(const TensorPtr& a, float s)
{
    return (*a) + s;
}

inline TensorPtr operator*(const TensorPtr& a, const TensorPtr& b)
{
    return (*a) * b;
}

inline TensorPtr operator*(const TensorPtr& a, float s)
{
    return (*a) * s;
}

// Named function for matmul so callers can write: matmul(x, w)
inline TensorPtr matmul(const TensorPtr& a, const TensorPtr& b)
{
    return a->matmul(b);
}

} // namespace autograd