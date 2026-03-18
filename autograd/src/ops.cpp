// ─── ops.cpp ─────────────────────────────────────────────────────────────────
// Implementations of all primitive kernel functions for the autograd engine.
// All operations work on flat row-major float buffers with explicit shape args.
// No external BLAS dependency — pure C++ loops throughout.

#include "autograd/ops.h"

#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>

namespace autograd {
namespace ops {

// ─── Matrix Multiply ─────────────────────────────────────────────────────────

// C = A @ B   where A:(M,K), B:(K,N) → C:(M,N)
// Classic triple loop — O(M*K*N).
std::vector<float> matmul(
    const std::vector<float>& a, size_t M, size_t K,
    const std::vector<float>& b,             size_t N)
{
    std::vector<float> c(M * N, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float aik = a[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                // c[i,j] += a[i,k] * b[k,j]
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }

    return c;
}

// Backward pass for matmul:
//   ga = gout @ B^T   shape (M,K)
//   gb = A^T  @ gout  shape (K,N)
void matmul_backward(
    const std::vector<float>& a,    size_t M, size_t K,
    const std::vector<float>& b,              size_t N,
    const std::vector<float>& gout,
    std::vector<float>& ga,
    std::vector<float>& gb)
{
    ga.assign(M * K, 0.0f);
    gb.assign(K * N, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const float g = gout[i * N + j];
            for (size_t k = 0; k < K; ++k) {
                // ga[i,k] += gout[i,j] * b[k,j]   (gout @ B^T)
                ga[i * K + k] += g * b[k * N + j];
                // gb[k,j] += a[i,k] * gout[i,j]   (A^T @ gout)
                gb[k * N + j] += a[i * K + k] * g;
            }
        }
    }
}

// ─── ReLU ────────────────────────────────────────────────────────────────────

// out[i] = max(0, x[i])
std::vector<float> relu_forward(const std::vector<float>& x)
{
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return out;
}

// grad_in[i] = (x[i] > 0) ? gout[i] : 0
std::vector<float> relu_backward(
    const std::vector<float>& x,
    const std::vector<float>& gout)
{
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] > 0.0f ? gout[i] : 0.0f;
    }
    return out;
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

// Numerically stable row-wise softmax.
// For each row: subtract row_max before exp to avoid overflow.
//   out[i,j] = exp(x[i,j] - max(x[i,:])) / sum_j(exp(x[i,j] - max(x[i,:])))
std::vector<float> softmax_forward(
    const std::vector<float>& x, size_t rows, size_t cols)
{
    std::vector<float> out(rows * cols);

    for (size_t i = 0; i < rows; ++i) {
        const float* row_in  = x.data()   + i * cols;
        float*       row_out = out.data()  + i * cols;

        // 1. Find row max for numerical stability.
        float row_max = *std::max_element(row_in, row_in + cols);

        // 2. Compute exp(x - max) and accumulate sum.
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            row_out[j] = std::exp(row_in[j] - row_max);
            sum += row_out[j];
        }

        // 3. Normalize.
        for (size_t j = 0; j < cols; ++j) {
            row_out[j] /= sum;
        }
    }

    return out;
}

// Jacobian contraction for softmax backward.
// Given y = softmax(x) and upstream grad dy:
//   dx[i,j] = y[i,j] * (dy[i,j] - sum_k(y[i,k] * dy[i,k]))
// This is the efficient O(N) form of the full Jacobian product.
std::vector<float> softmax_backward(
    const std::vector<float>& y,
    const std::vector<float>& dy,
    size_t rows, size_t cols)
{
    std::vector<float> dx(rows * cols);

    for (size_t i = 0; i < rows; ++i) {
        const float* yi  = y.data()  + i * cols;
        const float* dyi = dy.data() + i * cols;
        float*       dxi = dx.data() + i * cols;

        // dot product s = sum_k( y[i,k] * dy[i,k] )
        float s = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            s += yi[j] * dyi[j];
        }

        // dx[i,j] = y[i,j] * (dy[i,j] - s)
        for (size_t j = 0; j < cols; ++j) {
            dxi[j] = yi[j] * (dyi[j] - s);
        }
    }

    return dx;
}

// ─── Broadcast helpers ────────────────────────────────────────────────────────

// Resolves flat index for (a_rows x a_cols) tensor with broadcasting.
// If a_rows == 1, all rows map to row 0.
// If a_cols == 1, all cols map to col 0 (not used currently but kept general).
static inline float broadcast_get(
    const std::vector<float>& buf,
    size_t a_rows, size_t a_cols,
    size_t i, size_t j)
{
    size_t ri = (a_rows == 1) ? 0 : i;
    size_t ci = (a_cols == 1) ? 0 : j;
    return buf[ri * a_cols + ci];
}

// ─── Add (broadcast-aware) ────────────────────────────────────────────────────

// Adds two tensors with broadcasting support.
// Handles the common neural network case of adding a bias vector {cols} to a
// matrix {rows, cols}: each row gets the same bias added to it.
std::vector<float> add(
    const std::vector<float>& a, size_t a_rows, size_t a_cols,
    const std::vector<float>& b, size_t b_rows, size_t b_cols)
{
    // Output shape is the broadcasted shape.
    size_t out_rows = std::max(a_rows, b_rows);
    size_t out_cols = std::max(a_cols, b_cols);

    std::vector<float> out(out_rows * out_cols);

    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            float av = broadcast_get(a, a_rows, a_cols, i, j);
            float bv = broadcast_get(b, b_rows, b_cols, i, j);
            out[i * out_cols + j] = av + bv;
        }
    }

    return out;
}

// Reduces an upstream gradient back to the original shape by summing over
// the axes that were broadcast.
// out_grad shape: (out_rows, out_cols)
// original shape: (orig_rows, orig_cols)
// If orig_rows == 1 (was broadcast across rows), sum over the row axis.
// The result always has the same number of elements as the original tensor.
std::vector<float> add_backward_reduce(
    const std::vector<float>& gout,
    size_t out_rows, size_t out_cols,
    size_t orig_rows, size_t orig_cols)
{
    // Fast path: shapes already match, no reduction needed.
    if (orig_rows == out_rows && orig_cols == out_cols) {
        return gout;
    }

    std::vector<float> reduced(orig_rows * orig_cols, 0.0f);

    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            // Map the output index back to the original (possibly size-1) index.
            size_t ri = (orig_rows == 1) ? 0 : i;
            size_t ci = (orig_cols == 1) ? 0 : j;
            reduced[ri * orig_cols + ci] += gout[i * out_cols + j];
        }
    }

    return reduced;
}

// ─── Mul (broadcast-aware) ────────────────────────────────────────────────────

// Element-wise product with broadcasting.
std::vector<float> mul(
    const std::vector<float>& a, size_t a_rows, size_t a_cols,
    const std::vector<float>& b, size_t b_rows, size_t b_cols)
{
    size_t out_rows = std::max(a_rows, b_rows);
    size_t out_cols = std::max(a_cols, b_cols);

    std::vector<float> out(out_rows * out_cols);

    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            float av = broadcast_get(a, a_rows, a_cols, i, j);
            float bv = broadcast_get(b, b_rows, b_cols, i, j);
            out[i * out_cols + j] = av * bv;
        }
    }

    return out;
}

// Gradient for one operand of mul:
//   g_a[i,j] = gout[i,j] * other[i,j]   (element-wise, then reduce if broadcast)
std::vector<float> mul_backward_reduce(
    const std::vector<float>& gout,
    const std::vector<float>& other,
    size_t out_rows, size_t out_cols,
    size_t orig_rows, size_t orig_cols)
{
    // First compute element-wise product of upstream grad and other operand,
    // with broadcasting applied to `other`.
    std::vector<float> g(out_rows * out_cols);
    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            float ov = broadcast_get(other, orig_rows, orig_cols, i, j);
            g[i * out_cols + j] = gout[i * out_cols + j] * ov;
        }
    }

    // Then reduce back to original shape if broadcasting occurred.
    return add_backward_reduce(g, out_rows, out_cols, orig_rows, orig_cols);
}

// ─── Cross-Entropy helpers ────────────────────────────────────────────────────

// Numerically stable log-softmax, row-wise.
// log_probs[i,j] = x[i,j] - log(sum_k exp(x[i,k]))
//                = x[i,j] - (max(x[i,:]) + log(sum_k exp(x[i,k] - max(x[i,:]))))
std::vector<float> log_softmax(
    const std::vector<float>& logits, size_t M, size_t C)
{
    std::vector<float> log_probs(M * C);

    for (size_t i = 0; i < M; ++i) {
        const float* row = logits.data() + i * C;
        float*       out = log_probs.data() + i * C;

        // 1. Row max for stability.
        float row_max = *std::max_element(row, row + C);

        // 2. sum( exp(x - max) )
        float sum_exp = 0.0f;
        for (size_t j = 0; j < C; ++j) {
            sum_exp += std::exp(row[j] - row_max);
        }

        // 3. log-sum-exp = max + log(sum_exp)
        float lse = row_max + std::log(sum_exp);

        // 4. log_probs[i,j] = x[i,j] - lse
        for (size_t j = 0; j < C; ++j) {
            out[j] = row[j] - lse;
        }
    }

    return log_probs;
}

// Scalar NLL loss: -mean( log_probs[i, targets[i]] )
float nll_loss(
    const std::vector<float>& log_probs,
    const std::vector<int>& targets,
    size_t M, size_t C)
{
    float loss = 0.0f;
    for (size_t i = 0; i < M; ++i) {
        int t = targets[i];
        loss -= log_probs[i * C + t];
    }
    return loss / static_cast<float>(M);
}

// Gradient of cross-entropy w.r.t. logits.
// The gradient of NLL(log_softmax(z), y) w.r.t. z[i,j] is:
//   (softmax(z)[i,j] - 1_{j == targets[i]}) / M
// Then scaled by the upstream scalar grad `upstream`.
std::vector<float> cross_entropy_backward(
    const std::vector<float>& log_probs,
    const std::vector<int>& targets,
    size_t M, size_t C,
    float upstream)
{
    std::vector<float> grad(M * C);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < C; ++j) {
            // softmax = exp(log_probs)
            grad[i * C + j] = std::exp(log_probs[i * C + j]);
        }
        // Subtract 1 from the true class column.
        grad[i * C + targets[i]] -= 1.0f;
    }

    // Average over batch and scale by upstream gradient.
    const float scale = upstream / static_cast<float>(M);
    for (float& g : grad) {
        g *= scale;
    }

    return grad;
}

// ─── Adam ────────────────────────────────────────────────────────────────────

// In-place Adam update for a single parameter tensor.
//
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
//   mh  = m_t / (1 - beta1^t)      ← bias-corrected first moment
//   vh  = v_t / (1 - beta2^t)      ← bias-corrected second moment
//   p   = p - lr * mh / (sqrt(vh) + eps)
void adam_step(
    std::vector<float>& p,
    const std::vector<float>& g,
    std::vector<float>& m,
    std::vector<float>& v,
    float beta1, float beta2,
    float lr, float eps,
    int t)
{
    // Bias-correction denominators.
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    const size_t n = p.size();
    for (size_t i = 0; i < n; ++i) {
        // Update biased first moment estimate.
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        // Update biased second raw moment estimate.
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];

        // Compute bias-corrected moments.
        const float mh = m[i] / bc1;
        const float vh = v[i] / bc2;

        // Apply update.
        p[i] -= lr * mh / (std::sqrt(vh) + eps);
    }
}

} // namespace ops
} // namespace autograd