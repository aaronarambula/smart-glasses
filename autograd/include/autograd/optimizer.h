#pragma once

// ─── optimizer.h ─────────────────────────────────────────────────────────────
// Adam optimizer with optional gradient clipping.
// Mirrors the Python Adam class exactly, including:
//   - per-parameter first/second moment buffers (m, v)
//   - bias-correction via timestep t
//   - optional global gradient norm clipping before the update

#include "tensor.h"
#include "ops.h"

#include <vector>
#include <cmath>
#include <stdexcept>

namespace autograd {

// ─── Adam ─────────────────────────────────────────────────────────────────────
//
// Implements the Adam update rule (Kingma & Ba, 2015):
//
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   p_t = p_{t-1} - lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
//
// Usage:
//   Adam opt(model.parameters(), /*lr=*/1e-3f);
//   opt.zero_grad();
//   loss->backward();
//   opt.step();

class Adam {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    // params : all learnable TensorPtrs (typically from model.parameters())
    // lr     : learning rate (default 1e-3)
    // beta1  : exponential decay for first moment  (default 0.9)
    // beta2  : exponential decay for second moment (default 0.999)
    // eps    : numerical stability term            (default 1e-8)
    Adam(std::vector<TensorPtr> params,
         float lr    = 1e-3f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps   = 1e-8f)
        : params_(std::move(params))
        , lr_(lr)
        , beta1_(beta1)
        , beta2_(beta2)
        , eps_(eps)
        , t_(0)
    {
        // Allocate zeroed moment buffers for every parameter.
        m_.reserve(params_.size());
        v_.reserve(params_.size());
        for (const auto& p : params_) {
            m_.emplace_back(p->numel(), 0.0f);
            v_.emplace_back(p->numel(), 0.0f);
        }
    }

    // ── zero_grad ─────────────────────────────────────────────────────────────
    // Clears accumulated gradients from all tracked parameters.
    // Call this at the beginning of each training step.
    void zero_grad() {
        for (auto& p : params_) {
            p->grad.clear();   // empty == "no grad" (see Tensor::has_grad())
        }
    }

    // ── step ──────────────────────────────────────────────────────────────────
    // Performs one Adam update across all parameters.
    //
    // clip_norm: if > 0, clips the global gradient norm to this value before
    //            applying the update. Equivalent to Python's clip_norm arg.
    //            Set to -1.0f (default) to disable clipping.
    void step(float clip_norm = -1.0f) {
        // ── Optional gradient clipping ────────────────────────────────────────
        // Compute the global L2 norm across all parameter gradients,
        // then scale all gradients down if the norm exceeds clip_norm.
        if (clip_norm > 0.0f) {
            float total_sq = 0.0f;
            for (const auto& p : params_) {
                if (!p->has_grad()) continue;
                for (float g : p->grad) {
                    total_sq += g * g;
                }
            }
            const float total_norm = std::sqrt(total_sq);
            const float scale = clip_norm / (total_norm + 1e-6f);
            if (scale < 1.0f) {
                for (auto& p : params_) {
                    if (!p->has_grad()) continue;
                    for (float& g : p->grad) {
                        g *= scale;
                    }
                }
            }
        }

        // ── Adam update ───────────────────────────────────────────────────────
        ++t_;  // increment timestep before bias correction (1-indexed like Python)

        for (size_t i = 0; i < params_.size(); ++i) {
            auto& p = params_[i];
            if (!p->has_grad()) continue;   // skip frozen / unused params

            ops::adam_step(
                p->data,   // parameter buffer (updated in-place)
                p->grad,   // gradient
                m_[i],     // first moment buffer
                v_[i],     // second moment buffer
                beta1_, beta2_,
                lr_, eps_,
                t_
            );
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    float lr()    const { return lr_;    }
    float beta1() const { return beta1_; }
    float beta2() const { return beta2_; }
    float eps()   const { return eps_;   }
    int   timestep() const { return t_; }

    // Allow learning rate scheduling from outside.
    void set_lr(float lr) { lr_ = lr; }

    const std::vector<TensorPtr>& params() const { return params_; }

private:
    std::vector<TensorPtr>         params_;  // tracked parameters
    float                          lr_;
    float                          beta1_;
    float                          beta2_;
    float                          eps_;
    int                            t_;       // current timestep (bias-correction)
    std::vector<std::vector<float>> m_;      // first moment  (one per param)
    std::vector<std::vector<float>> v_;      // second moment (one per param)
};

} // namespace autograd