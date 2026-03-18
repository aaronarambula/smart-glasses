#pragma once

// ─── ops.h ───────────────────────────────────────────────────────────────────
// Declarations for all primitive kernel functions used by the autograd engine.
// All kernels operate on flat row-major float buffers with explicit shape args.

#include <vector>
#include <cstddef>

namespace autograd {
namespace ops {

// ─── Matrix Multiply ─────────────────────────────────────────────────────────

// Forward: C = A @ B
// A: (M, K)  B: (K, N)  →  C: (M, N)
std::vector<float> matmul(
    const std::vector<float>& a, size_t M, size_t K,
    const std::vector<float>& b,             size_t N);

// Backward: given upstream gradient gout (M, N),
//   ga = gout @ B^T  →  shape (M, K)
//   gb = A^T @ gout  →  shape (K, N)
void matmul_backward(
    const std::vector<float>& a,    size_t M, size_t K,
    const std::vector<float>& b,              size_t N,
    const std::vector<float>& gout,
    std::vector<float>& ga,
    std::vector<float>& gb);

// ─── ReLU ────────────────────────────────────────────────────────────────────

// Forward: out[i] = max(0, x[i])
std::vector<float> relu_forward(const std::vector<float>& x);

// Backward: grad_in[i] = (x[i] > 0) ? gout[i] : 0
std::vector<float> relu_backward(
    const std::vector<float>& x,
    const std::vector<float>& gout);

// ─── Softmax ─────────────────────────────────────────────────────────────────

// Forward: numerically stable row-wise softmax.
// x: (rows, cols)  →  out: (rows, cols)
std::vector<float> softmax_forward(
    const std::vector<float>& x, size_t rows, size_t cols);

// Backward: Jacobian contraction.
// y = softmax output (rows, cols), dy = upstream grad (rows, cols)
// dx[i] = y[i] * (dy[i] - sum_j(y[i,j] * dy[i,j]))   (row-wise)
std::vector<float> softmax_backward(
    const std::vector<float>& y,
    const std::vector<float>& dy,
    size_t rows, size_t cols);

// ─── Element-wise Add (broadcast-aware) ──────────────────────────────────────

// Adds a to b with broadcasting support for the common case of
// a bias vector of shape (cols,) being added to a matrix (rows, cols).
// Both full-size (same shape) and bias-broadcast cases are handled.
std::vector<float> add(
    const std::vector<float>& a, size_t a_rows, size_t a_cols,
    const std::vector<float>& b, size_t b_rows, size_t b_cols);

// Reduce gradient back to original shape by summing over broadcast axes.
// out_grad has shape (rows, cols); original had shape (orig_rows, orig_cols).
// If orig is (1, cols) or (cols,), sums over rows.
std::vector<float> add_backward_reduce(
    const std::vector<float>& gout,
    size_t out_rows, size_t out_cols,
    size_t orig_rows, size_t orig_cols);

// ─── Element-wise Multiply (broadcast-aware) ─────────────────────────────────

// out[i] = a[i] * b[i], with same broadcast rules as add.
std::vector<float> mul(
    const std::vector<float>& a, size_t a_rows, size_t a_cols,
    const std::vector<float>& b, size_t b_rows, size_t b_cols);

// Gradient for one operand of mul: g_a = gout * b (then reduce if broadcast).
std::vector<float> mul_backward_reduce(
    const std::vector<float>& gout,
    const std::vector<float>& other,
    size_t out_rows, size_t out_cols,
    size_t orig_rows, size_t orig_cols);

// ─── Cross-Entropy helpers ────────────────────────────────────────────────────

// Numerically stable log-softmax: log_probs[i,j] = x[i,j] - logsumexp(x[i,:])
// logits: (M, C)  →  log_probs: (M, C)
std::vector<float> log_softmax(
    const std::vector<float>& logits, size_t M, size_t C);

// Scalar NLL loss: -mean( log_probs[i, targets[i]] )
float nll_loss(
    const std::vector<float>& log_probs,
    const std::vector<int>& targets,
    size_t M, size_t C);

// Gradient of cross-entropy w.r.t. logits:
//   grad[i,j] = softmax(logits)[i,j] - (j == targets[i])  then / M
// Scaled by upstream scalar grad `upstream`.
std::vector<float> cross_entropy_backward(
    const std::vector<float>& log_probs,
    const std::vector<int>& targets,
    size_t M, size_t C,
    float upstream);

// ─── Adam ────────────────────────────────────────────────────────────────────

// In-place Adam parameter update.
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g^2
//   p -= lr * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + eps)
void adam_step(
    std::vector<float>& p,
    const std::vector<float>& g,
    std::vector<float>& m,
    std::vector<float>& v,
    float beta1, float beta2,
    float lr, float eps,
    int t);

} // namespace ops
} // namespace autograd