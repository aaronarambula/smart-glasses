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