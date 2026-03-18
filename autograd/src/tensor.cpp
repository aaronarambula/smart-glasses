// ─── tensor.cpp ──────────────────────────────────────────────────────────────
// Tensor constructors, backward (reverse-mode autodiff with iterative topo
// sort), and all operator implementations that build the computation graph.

#include "autograd/tensor.h"
#include "autograd/ops.h"
#include "autograd/no_grad.h"

#include <cassert>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <stack>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace autograd {

// ─── Constructors ─────────────────────────────────────────────────────────────

// Scalar: stored as a 1-element tensor with shape {1}
Tensor::Tensor(float scalar, bool requires_grad)
    : data({scalar})
    , shape({1})
    , requires_grad(requires_grad && enable_grad)
    , _backward([](){})
{}

// From flat data + explicit shape vector
Tensor::Tensor(std::vector<float> data_, std::vector<size_t> shape_,
               bool requires_grad)
    : data(std::move(data_))
    , shape(std::move(shape_))
    , requires_grad(requires_grad && enable_grad)
    , _backward([](){})
{
    // Sanity check: product of shape dims must match data size.
    size_t expected = 1;
    for (size_t d : shape) expected *= d;
    if (data.size() != expected) {
        throw std::invalid_argument(
            "Tensor: data.size() (" + std::to_string(data.size()) +
            ") does not match product of shape (" + std::to_string(expected) + ")");
    }
}

// Convenience 2-D constructor
Tensor::Tensor(std::vector<float> data_, size_t rows_, size_t cols_,
               bool requires_grad)
    : Tensor(std::move(data_), std::vector<size_t>{rows_, cols_}, requires_grad)
{}

// ─── Shape helpers ────────────────────────────────────────────────────────────

std::string Tensor::shape_str() const
{
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size()) ss << ", ";
    }
    ss << ")";
    return ss.str();
}

std::string Tensor::repr() const
{
    std::ostringstream ss;
    ss << "Tensor(shape=" << shape_str()
       << ", requires_grad=" << (requires_grad ? "true" : "false");
    if (has_grad()) ss << ", has_grad";
    ss << ")";
    return ss.str();
}

// ─── Gradient helpers ─────────────────────────────────────────────────────────

void Tensor::accumulate_grad(const std::vector<float>& delta)
{
    if (grad.empty()) {
        // Lazily allocate grad buffer (zeros) then add delta.
        grad.assign(data.size(), 0.0f);
    }
    assert(grad.size() == delta.size());
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] += delta[i];
    }
}

void Tensor::zero_grad()
{
    std::fill(grad.begin(), grad.end(), 0.0f);
}

// ─── Backward ────────────────────────────────────────────────────────────────
//
// Runs reverse-mode autodiff from this tensor as the root.
// Uses an iterative post-order DFS to build the topological order, avoiding
// potential stack overflow on very deep computation graphs.
//
// Algorithm:
//   1. Iterative DFS → collect nodes in reverse topological order.
//   2. Seed this node's grad with `initial_grad` (or all-ones for a scalar loss).
//   3. Walk the topo list in reverse (leaf → root already reversed, so forward).
//      Actually: topo is built leaves-first, so reversed(topo) is root-first,
//      which is what backward needs.

void Tensor::backward(std::vector<float> initial_grad)
{
    if (!requires_grad) return;

    // ── Build topological order (iterative post-order DFS) ───────────────────
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;

    // Stack entries: (node, iterator into its _prev children)
    // We use a two-stack approach: push to explore, pop after children done.
    std::stack<std::pair<Tensor*, size_t>> stk;
    stk.push({this, 0});
    visited.insert(this);

    while (!stk.empty()) {
        auto& [node, child_idx] = stk.top();

        if (child_idx < node->_prev.size()) {
            // Visit next unvisited child.
            Tensor* child = node->_prev[child_idx].get();
            ++child_idx;
            if (visited.find(child) == visited.end()) {
                visited.insert(child);
                stk.push({child, 0});
            }
        } else {
            // All children visited — this node is "done", add to topo.
            topo.push_back(node);
            stk.pop();
        }
    }

    // ── Seed gradient ─────────────────────────────────────────────────────────
    if (initial_grad.empty()) {
        // Default: gradient of a scalar loss = 1.
        grad.assign(data.size(), 1.0f);
    } else {
        grad = std::move(initial_grad);
    }

    // ── Propagate gradients in reverse topological order ─────────────────────
    // topo is leaves-first (post-order), so reversed is root-first.
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

// ─── operator+ ───────────────────────────────────────────────────────────────
//
// Handles:
//   (B, N) + (B, N)  — same shape
//   (B, N) + (N,)    — bias broadcast (most common case in Linear layers)
//   scalar + tensor  — trivially broadcast

TensorPtr Tensor::operator+(const TensorPtr& other) const
{
    // Determine output shape dimensions.
    size_t a_rows = rows(), a_cols = cols();
    size_t b_rows = other->rows(), b_cols = other->cols();

    auto result_data = ops::add(data, a_rows, a_cols,
                                other->data, b_rows, b_cols);

    size_t out_rows = std::max(a_rows, b_rows);
    size_t out_cols = std::max(a_cols, b_cols);

    auto out = std::make_shared<Tensor>(
        std::move(result_data),
        std::vector<size_t>{out_rows, out_cols},
        requires_grad || other->requires_grad
    );

    // Capture by shared_ptr to keep inputs alive for the backward closure.
    auto self_ptr  = std::shared_ptr<const Tensor>(shared_from_this(), this);
    auto other_ptr = other;

    out->_prev = {
        std::const_pointer_cast<Tensor>(self_ptr),
        other_ptr
    };

    out->_backward = [self_ptr, other_ptr,
                      a_rows, a_cols, b_rows, b_cols,
                      out_rows, out_cols,
                      out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        const auto& gout = out_sp->grad;

        // Gradient flows through add unchanged, reduced if broadcast occurred.
        if (self_ptr->requires_grad) {
            auto gs = ops::add_backward_reduce(
                gout, out_rows, out_cols, a_rows, a_cols);
            const_cast<Tensor*>(self_ptr.get())->accumulate_grad(gs);
        }
        if (other_ptr->requires_grad) {
            auto go = ops::add_backward_reduce(
                gout, out_rows, out_cols, b_rows, b_cols);
            other_ptr->accumulate_grad(go);
        }
    };

    return out;
}

TensorPtr Tensor::operator+(float scalar) const
{
    auto scalar_t = std::make_shared<Tensor>(scalar, /*requires_grad=*/false);
    return (*this) + scalar_t;
}

// ─── operator* ───────────────────────────────────────────────────────────────
//
// Element-wise multiply with broadcast, same semantics as operator+.

TensorPtr Tensor::operator*(const TensorPtr& other) const
{
    size_t a_rows = rows(), a_cols = cols();
    size_t b_rows = other->rows(), b_cols = other->cols();

    auto result_data = ops::mul(data, a_rows, a_cols,
                                other->data, b_rows, b_cols);

    size_t out_rows = std::max(a_rows, b_rows);
    size_t out_cols = std::max(a_cols, b_cols);

    auto out = std::make_shared<Tensor>(
        std::move(result_data),
        std::vector<size_t>{out_rows, out_cols},
        requires_grad || other->requires_grad
    );

    auto self_ptr  = std::shared_ptr<const Tensor>(shared_from_this(), this);
    auto other_ptr = other;

    out->_prev = {
        std::const_pointer_cast<Tensor>(self_ptr),
        other_ptr
    };

    out->_backward = [self_ptr, other_ptr,
                      a_rows, a_cols, b_rows, b_cols,
                      out_rows, out_cols,
                      out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        const auto& gout = out_sp->grad;

        // d/da (a*b) = b  →  reduce if a was broadcast
        if (self_ptr->requires_grad) {
            auto gs = ops::mul_backward_reduce(
                gout, other_ptr->data,
                out_rows, out_cols, a_rows, a_cols);
            const_cast<Tensor*>(self_ptr.get())->accumulate_grad(gs);
        }
        // d/db (a*b) = a  →  reduce if b was broadcast
        if (other_ptr->requires_grad) {
            auto go = ops::mul_backward_reduce(
                gout, self_ptr->data,
                out_rows, out_cols, b_rows, b_cols);
            other_ptr->accumulate_grad(go);
        }
    };

    return out;
}

TensorPtr Tensor::operator*(float scalar) const
{
    auto scalar_t = std::make_shared<Tensor>(scalar, /*requires_grad=*/false);
    return (*this) * scalar_t;
}

// ─── matmul ──────────────────────────────────────────────────────────────────
//
// (M, K) @ (K, N) → (M, N)

TensorPtr Tensor::matmul(const TensorPtr& other) const
{
    // Validate shapes.
    if (shape.size() < 2 || other->shape.size() < 2) {
        throw std::invalid_argument("matmul requires at least 2-D tensors");
    }
    size_t M = rows(), K = cols();
    size_t K2 = other->rows(), N = other->cols();
    if (K != K2) {
        throw std::invalid_argument(
            "matmul shape mismatch: (" + std::to_string(M) + "," +
            std::to_string(K) + ") @ (" + std::to_string(K2) + "," +
            std::to_string(N) + ")");
    }

    auto result_data = ops::matmul(data, M, K, other->data, N);

    auto out = std::make_shared<Tensor>(
        std::move(result_data), M, N,
        requires_grad || other->requires_grad
    );

    auto self_ptr  = std::shared_ptr<const Tensor>(shared_from_this(), this);
    auto other_ptr = other;

    out->_prev = {
        std::const_pointer_cast<Tensor>(self_ptr),
        other_ptr
    };

    out->_backward = [self_ptr, other_ptr, M, K, N,
                      out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        const auto& gout = out_sp->grad;

        std::vector<float> ga, gb;
        ops::matmul_backward(
            self_ptr->data, M, K,
            other_ptr->data, N,
            gout, ga, gb);

        if (self_ptr->requires_grad) {
            const_cast<Tensor*>(self_ptr.get())->accumulate_grad(ga);
        }
        if (other_ptr->requires_grad) {
            other_ptr->accumulate_grad(gb);
        }
    };

    return out;
}

// ─── relu ─────────────────────────────────────────────────────────────────────

TensorPtr Tensor::relu() const
{
    auto result_data = ops::relu_forward(data);

    auto out = std::make_shared<Tensor>(
        std::move(result_data), shape, requires_grad
    );

    auto self_ptr = std::shared_ptr<const Tensor>(shared_from_this(), this);
    out->_prev = { std::const_pointer_cast<Tensor>(self_ptr) };

    out->_backward = [self_ptr, out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        if (!self_ptr->requires_grad) return;

        // grad_in[i] = (x[i] > 0) ? gout[i] : 0
        auto g = ops::relu_backward(self_ptr->data, out_sp->grad);
        const_cast<Tensor*>(self_ptr.get())->accumulate_grad(g);
    };

    return out;
}

// ─── softmax ──────────────────────────────────────────────────────────────────
//
// Row-wise numerically stable softmax.

TensorPtr Tensor::softmax() const
{
    size_t r = rows(), c = cols();
    auto result_data = ops::softmax_forward(data, r, c);

    auto out = std::make_shared<Tensor>(
        std::move(result_data), shape, requires_grad
    );

    auto self_ptr = std::shared_ptr<const Tensor>(shared_from_this(), this);
    out->_prev = { std::const_pointer_cast<Tensor>(self_ptr) };

    out->_backward = [self_ptr, r, c, out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        if (!self_ptr->requires_grad) return;

        // Jacobian contraction: dx = y * (dy - sum(y*dy, row))
        auto g = ops::softmax_backward(out_sp->data, out_sp->grad, r, c);
        const_cast<Tensor*>(self_ptr.get())->accumulate_grad(g);
    };

    return out;
}

// ─── sum ──────────────────────────────────────────────────────────────────────
//
// Reduces all elements to a scalar.

TensorPtr Tensor::sum() const
{
    float total = 0.0f;
    for (float v : data) total += v;

    auto out = std::make_shared<Tensor>(total, requires_grad);
    auto self_ptr = std::shared_ptr<const Tensor>(shared_from_this(), this);
    out->_prev = { std::const_pointer_cast<Tensor>(self_ptr) };

    out->_backward = [self_ptr, out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        if (!self_ptr->requires_grad) return;

        // Gradient of sum w.r.t. each input element is the upstream scalar.
        const float upstream = out_sp->grad[0];
        std::vector<float> g(self_ptr->data.size(), upstream);
        const_cast<Tensor*>(self_ptr.get())->accumulate_grad(g);
    };

    return out;
}

// ─── cross_entropy ────────────────────────────────────────────────────────────
//
// Numerically stable cross-entropy loss.
// logits: (M, C)   targets: integer indices in [0, C)
// Returns a scalar TensorPtr.

TensorPtr Tensor::cross_entropy(const TensorPtr& logits,
                                const std::vector<int>& targets)
{
    if (logits->shape.size() < 2) {
        throw std::invalid_argument("cross_entropy: logits must be at least 2-D");
    }
    size_t M = logits->rows();
    size_t C = logits->cols();

    if (targets.size() != M) {
        throw std::invalid_argument(
            "cross_entropy: targets.size() (" + std::to_string(targets.size()) +
            ") != batch size (" + std::to_string(M) + ")");
    }

    // Compute log-softmax and scalar loss.
    auto log_probs = ops::log_softmax(logits->data, M, C);
    float loss_val = ops::nll_loss(log_probs, targets, M, C);

    auto out = std::make_shared<Tensor>(loss_val, logits->requires_grad);
    out->_prev = { logits };

    out->_backward = [logits, log_probs, targets, M, C,
                      out_wk = std::weak_ptr<Tensor>(out)]()
    {
        auto out_sp = out_wk.lock();
        if (!out_sp || out_sp->grad.empty()) return;
        if (!logits->requires_grad) return;

        // Upstream gradient is a scalar (the loss grad, typically 1.0).
        const float upstream = out_sp->grad[0];

        auto g = ops::cross_entropy_backward(log_probs, targets, M, C, upstream);
        logits->accumulate_grad(g);
    };

    return out;
}

} // namespace autograd