#pragma once

// ─── layers.h ────────────────────────────────────────────────────────────────
// Neural network layer abstractions built on top of the autograd Tensor.
//
//   Layer        – pure-virtual base (forward + parameters)
//   Linear       – affine transform with He-initialized weights
//   ReLU         – element-wise rectified linear unit
//   Sequential   – ordered container of layers

#include "tensor.h"

#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <numeric>

namespace autograd {

// ─── Layer (abstract base) ────────────────────────────────────────────────────
//
// All layers implement:
//   forward()    – runs the layer and returns a new TensorPtr
//   parameters() – returns all learnable TensorPtrs (empty for stateless layers)

class Layer {
public:
    virtual ~Layer() = default;

    virtual TensorPtr forward(const TensorPtr& x) = 0;

    // Convenience: makes layers callable like Python's __call__
    TensorPtr operator()(const TensorPtr& x) { return forward(x); }

    virtual std::vector<TensorPtr> parameters() { return {}; }

    // Human-readable name for debugging
    virtual std::string name() const = 0;
};

// ─── Linear ───────────────────────────────────────────────────────────────────
//
// Affine layer: out = x @ weight + bias
//
//   weight : shape (in_features, out_features)  — He-initialized
//   bias   : shape (out_features,)              — zero-initialized
//
// He initialization: limit = sqrt(2 / in_features),
//                    weight ~ Uniform(-limit, limit)
// (matches the Python `np.random.randn * limit` variant used in the original.)

class Linear : public Layer {
public:
    TensorPtr weight;   // (in_features, out_features)
    TensorPtr bias;     // (out_features,)

    Linear(size_t in_features, size_t out_features,
           unsigned int seed = 42)
        : in_(in_features), out_(out_features)
    {
        // ── He initialization ────────────────────────────────────────────────
        // scale = sqrt(2 / fan_in)
        const float limit = std::sqrt(2.0f / static_cast<float>(in_features));

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> w_data(in_features * out_features);
        for (auto& v : w_data) {
            v = dist(rng) * limit;
        }

        weight = make_tensor(std::move(w_data), in_features, out_features,
                             /*requires_grad=*/true);

        // ── Zero bias ────────────────────────────────────────────────────────
        bias = make_tensor(std::vector<float>(out_features, 0.0f),
                           std::vector<size_t>{out_features},
                           /*requires_grad=*/true);
    }

    // forward: x (B, in) @ weight (in, out) + bias (out,) → (B, out)
    TensorPtr forward(const TensorPtr& x) override {
        // matmul then broadcast-add bias
        return x->matmul(weight) + bias;
    }

    std::vector<TensorPtr> parameters() override {
        return {weight, bias};
    }

    std::string name() const override {
        return "Linear(" + std::to_string(in_) + ", " + std::to_string(out_) + ")";
    }

private:
    size_t in_, out_;
};

// ─── ReLU ─────────────────────────────────────────────────────────────────────
//
// Stateless activation: out = max(0, x)
// No learnable parameters.

class ReLU : public Layer {
public:
    TensorPtr forward(const TensorPtr& x) override {
        return x->relu();
    }

    std::vector<TensorPtr> parameters() override { return {}; }

    std::string name() const override { return "ReLU"; }
};

// ─── Conv2d ──────────────────────────────────────────────────────────────────
//
// Valid 2-D convolution over NCHW tensors with square kernels and stride.
// Input  shape: (N, C_in, H, W)
// Weight shape: (C_out, C_in, K, K)
// Bias   shape: (C_out)
// Output shape: (N, C_out, (H-K)/stride+1, (W-K)/stride+1)

class Conv2d : public Layer {
public:
    TensorPtr weight;
    TensorPtr bias;

    Conv2d(size_t in_channels,
           size_t out_channels,
           size_t kernel_size,
           size_t stride = 1,
           unsigned int seed = 42)
        : in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_(kernel_size)
        , stride_(stride)
    {
        if (in_channels_ == 0 || out_channels_ == 0 || kernel_size_ == 0 || stride_ == 0) {
            throw std::invalid_argument("Conv2d: channels, kernel_size, and stride must be > 0");
        }

        const float limit = std::sqrt(2.0f /
            static_cast<float>(in_channels_ * kernel_size_ * kernel_size_));

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> w_data(out_channels_ * in_channels_ * kernel_size_ * kernel_size_);
        for (auto& v : w_data) {
            v = dist(rng) * limit;
        }

        weight = make_tensor(std::move(w_data),
                             std::vector<size_t>{out_channels_, in_channels_, kernel_size_, kernel_size_},
                             /*requires_grad=*/true);
        bias = make_tensor(std::vector<float>(out_channels_, 0.0f),
                           std::vector<size_t>{out_channels_},
                           /*requires_grad=*/true);
    }

    TensorPtr forward(const TensorPtr& x) override {
        if (x->shape.size() != 4) {
            throw std::invalid_argument("Conv2d::forward expects input shape (N,C,H,W)");
        }
        const size_t N  = x->shape[0];
        const size_t Ci = x->shape[1];
        const size_t H  = x->shape[2];
        const size_t W  = x->shape[3];
        if (Ci != in_channels_) {
            throw std::invalid_argument("Conv2d::forward input channels mismatch");
        }
        if (H < kernel_size_ || W < kernel_size_) {
            throw std::invalid_argument("Conv2d::forward kernel larger than input");
        }

        const size_t Ho = (H - kernel_size_) / stride_ + 1;
        const size_t Wo = (W - kernel_size_) / stride_ + 1;

        std::vector<float> out_data(N * out_channels_ * Ho * Wo, 0.0f);

        auto x_idx = [=](size_t n, size_t c, size_t h, size_t w) {
            return ((n * Ci + c) * H + h) * W + w;
        };
        auto w_idx = [=](size_t co, size_t ci, size_t kh, size_t kw) {
            return ((co * Ci + ci) * kernel_size_ + kh) * kernel_size_ + kw;
        };
        auto o_idx = [=](size_t n, size_t co, size_t h, size_t w) {
            return ((n * out_channels_ + co) * Ho + h) * Wo + w;
        };

        for (size_t n = 0; n < N; ++n) {
            for (size_t co = 0; co < out_channels_; ++co) {
                for (size_t oh = 0; oh < Ho; ++oh) {
                    for (size_t ow = 0; ow < Wo; ++ow) {
                        float sum = bias->data[co];
                        const size_t ih0 = oh * stride_;
                        const size_t iw0 = ow * stride_;
                        for (size_t ci = 0; ci < Ci; ++ci) {
                            for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                    sum += x->data[x_idx(n, ci, ih0 + kh, iw0 + kw)] *
                                           weight->data[w_idx(co, ci, kh, kw)];
                                }
                            }
                        }
                        out_data[o_idx(n, co, oh, ow)] = sum;
                    }
                }
            }
        }

        auto out = make_tensor(std::move(out_data),
                               std::vector<size_t>{N, out_channels_, Ho, Wo},
                               x->requires_grad || weight->requires_grad || bias->requires_grad);

        auto self_ptr = std::shared_ptr<const Tensor>(x->shared_from_this(), x.get());
        auto weight_ptr = weight;
        auto bias_ptr = bias;

        out->_prev = {
            std::const_pointer_cast<Tensor>(self_ptr),
            weight_ptr,
            bias_ptr
        };

        out->_backward = [self_ptr, weight_ptr, bias_ptr,
                          N, Ci, H, W, Ho, Wo,
                          out_channels = out_channels_,
                          kernel = kernel_size_,
                          stride = stride_,
                          out_wk = std::weak_ptr<Tensor>(out)]()
        {
            auto out_sp = out_wk.lock();
            if (!out_sp || out_sp->grad.empty()) return;

            auto x_idx = [=](size_t n, size_t c, size_t h, size_t w) {
                return ((n * Ci + c) * H + h) * W + w;
            };
            auto w_idx = [=](size_t co, size_t ci, size_t kh, size_t kw) {
                return ((co * Ci + ci) * kernel + kh) * kernel + kw;
            };
            auto o_idx = [=](size_t n, size_t co, size_t h, size_t w) {
                return ((n * out_channels + co) * Ho + h) * Wo + w;
            };

            std::vector<float> gx(self_ptr->numel(), 0.0f);
            std::vector<float> gw(weight_ptr->numel(), 0.0f);
            std::vector<float> gb(bias_ptr->numel(), 0.0f);

            for (size_t n = 0; n < N; ++n) {
                for (size_t co = 0; co < out_channels; ++co) {
                    for (size_t oh = 0; oh < Ho; ++oh) {
                        for (size_t ow = 0; ow < Wo; ++ow) {
                            const float go = out_sp->grad[o_idx(n, co, oh, ow)];
                            gb[co] += go;

                            const size_t ih0 = oh * stride;
                            const size_t iw0 = ow * stride;
                            for (size_t ci = 0; ci < Ci; ++ci) {
                                for (size_t kh = 0; kh < kernel; ++kh) {
                                    for (size_t kw = 0; kw < kernel; ++kw) {
                                        const size_t xi = x_idx(n, ci, ih0 + kh, iw0 + kw);
                                        const size_t wi = w_idx(co, ci, kh, kw);
                                        gx[xi] += go * weight_ptr->data[wi];
                                        gw[wi] += go * self_ptr->data[xi];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (self_ptr->requires_grad) {
                const_cast<Tensor*>(self_ptr.get())->accumulate_grad(gx);
            }
            if (weight_ptr->requires_grad) {
                weight_ptr->accumulate_grad(gw);
            }
            if (bias_ptr->requires_grad) {
                bias_ptr->accumulate_grad(gb);
            }
        };

        return out;
    }

    std::vector<TensorPtr> parameters() override {
        return {weight, bias};
    }

    std::string name() const override {
        return "Conv2d(" + std::to_string(in_channels_) + ", " +
               std::to_string(out_channels_) + ", k=" +
               std::to_string(kernel_size_) + ", s=" +
               std::to_string(stride_) + ")";
    }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
};

// ─── Flatten ─────────────────────────────────────────────────────────────────
//
// Reshapes (N, C, H, W) → (N, C*H*W), preserving batch dimension.

class Flatten : public Layer {
public:
    TensorPtr forward(const TensorPtr& x) override {
        if (x->shape.empty()) {
            throw std::invalid_argument("Flatten::forward expects at least 1-D tensor");
        }
        const size_t batch = x->shape[0];
        const size_t rest = x->numel() / batch;
        auto out = make_tensor(x->data, std::vector<size_t>{batch, rest}, x->requires_grad);

        auto self_ptr = std::shared_ptr<const Tensor>(x->shared_from_this(), x.get());
        out->_prev = { std::const_pointer_cast<Tensor>(self_ptr) };
        out->_backward = [self_ptr, out_wk = std::weak_ptr<Tensor>(out)]() {
            auto out_sp = out_wk.lock();
            if (!out_sp || out_sp->grad.empty()) return;
            if (self_ptr->requires_grad) {
                const_cast<Tensor*>(self_ptr.get())->accumulate_grad(out_sp->grad);
            }
        };
        return out;
    }

    std::string name() const override { return "Flatten"; }
};

// ─── Softmax ──────────────────────────────────────────────────────────────────
//
// Stateless row-wise softmax activation.
// Typically used as the final layer for multi-class output probabilities.
// Note: prefer using Tensor::cross_entropy(logits, targets) directly during
// training (it is numerically stabler); use this layer only at inference time.

class Softmax : public Layer {
public:
    TensorPtr forward(const TensorPtr& x) override {
        return x->softmax();
    }

    std::vector<TensorPtr> parameters() override { return {}; }

    std::string name() const override { return "Softmax"; }
};

// ─── Sequential ───────────────────────────────────────────────────────────────
//
// Chains an ordered list of layers: output of layer[i] is input to layer[i+1].
//
// Ownership of each layer is transferred via unique_ptr so Sequential is the
// sole owner — matches Python's implicit ownership in a list.
//
// Usage:
//   Sequential model;
//   model.add<Linear>(4, 8);
//   model.add<ReLU>();
//   model.add<Linear>(8, 3);
//
//   auto out = model.forward(x);

class Sequential : public Layer {
public:
    Sequential() = default;

    // Variadic constructor: Sequential(std::make_unique<Linear>(4,8), ...)
    explicit Sequential(std::vector<std::unique_ptr<Layer>> layers)
        : layers_(std::move(layers)) {}

    // Add a layer, constructing it in-place.
    // Usage: seq.add<Linear>(4, 8);
    template<typename T, typename... Args>
    Sequential& add(Args&&... args) {
        layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *this;  // fluent interface
    }

    // Run each layer in order.
    TensorPtr forward(const TensorPtr& x) override {
        TensorPtr out = x;
        for (auto& layer : layers_) {
            out = layer->forward(out);
        }
        return out;
    }

    // Collect all parameters from all child layers.
    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& layer : layers_) {
            auto lp = layer->parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }

    std::string name() const override { return "Sequential"; }

    size_t size() const { return layers_.size(); }

    // Index access for inspection
    Layer& operator[](size_t i) {
        if (i >= layers_.size()) {
            throw std::out_of_range("Sequential: layer index out of range");
        }
        return *layers_[i];
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace autograd
