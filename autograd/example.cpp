// ─── example.cpp ─────────────────────────────────────────────────────────────
// Demonstrates the autograd engine end-to-end:
//   1. Build a 2-layer MLP: Linear(4→8) → ReLU → Linear(8→3)
//   2. Run a forward pass with a batch of 2 samples
//   3. Compute cross-entropy loss
//   4. Call backward() to propagate gradients
//   5. Print loss, a few gradient values, then do one Adam step
//
// Build (from smart-glasses/autograd/):
//   mkdir build && cd build
//   cmake .. && make
//   ./example

#include "autograd/autograd.h"

#include <iostream>
#include <iomanip>
#include <vector>

using namespace autograd;

// ─── pretty-print helpers ─────────────────────────────────────────────────────

static void print_vec(const std::string& label,
                      const std::vector<float>& v,
                      size_t max_elems = 8)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << label << " [";
    size_t n = std::min(v.size(), max_elems);
    for (size_t i = 0; i < n; ++i) {
        std::cout << v[i];
        if (i + 1 < n) std::cout << ", ";
    }
    if (v.size() > max_elems) std::cout << ", ... (" << v.size() << " total)";
    std::cout << "]\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "=== Autograd C++ Engine — MLP Demo ===\n\n";

    // ── 1. Build the model ────────────────────────────────────────────────────
    //
    //   Input  : (batch=2, features=4)
    //   Layer 0: Linear(4 → 8)
    //   Layer 1: ReLU
    //   Layer 2: Linear(8 → 3)
    //   Output : (batch=2, classes=3)  ← raw logits

    Sequential model;
    model.add<Linear>(4, 8, /*seed=*/0)
         .add<ReLU>()
         .add<Linear>(8, 3, /*seed=*/1);

    std::cout << "Model architecture:\n";
    for (size_t i = 0; i < model.size(); ++i) {
        std::cout << "  [" << i << "] " << model[i].name() << "\n";
    }
    std::cout << "\n";

    // ── 2. Create input tensor ────────────────────────────────────────────────
    //
    // Two samples, each with 4 features.
    // shape: (2, 4)  —  row-major, so sample 0 = first 4 floats.

    auto x = make_tensor(
        {
            0.5f, -1.2f,  0.8f,  0.3f,   // sample 0
           -0.7f,  0.4f, -0.5f,  1.1f    // sample 1
        },
        /*rows=*/2, /*cols=*/4,
        /*requires_grad=*/false   // inputs don't need gradients
    );

    std::cout << "Input " << x->repr() << "\n";
    print_vec("  data", x->data);
    std::cout << "\n";

    // ── 3. Forward pass ───────────────────────────────────────────────────────

    auto logits = model.forward(x);

    std::cout << "Logits " << logits->repr() << "\n";
    print_vec("  data", logits->data);
    std::cout << "\n";

    // ── 4. Cross-entropy loss ─────────────────────────────────────────────────
    //
    // Ground-truth class indices:
    //   sample 0 → class 2
    //   sample 1 → class 0

    std::vector<int> targets = {2, 0};

    auto loss = Tensor::cross_entropy(logits, targets);

    std::cout << "Loss " << loss->repr() << "\n";
    std::cout << "  value = " << std::fixed << std::setprecision(6)
              << loss->data[0] << "\n\n";

    // ── 5. Backward pass ──────────────────────────────────────────────────────

    loss->backward();   // seeds loss.grad = [1.0], propagates through graph

    std::cout << "Gradients after backward():\n";

    // Retrieve layer parameters for inspection.
    auto params = model.parameters();
    // params order: [L0.weight, L0.bias, L2.weight, L2.bias]

    for (size_t i = 0; i < params.size(); ++i) {
        const auto& p = params[i];
        std::string label = "  param[" + std::to_string(i) + "] grad";
        if (p->has_grad()) {
            print_vec(label, p->grad, 6);
        } else {
            std::cout << label << " <none>\n";
        }
    }
    std::cout << "\n";

    // ── 6. Adam optimizer step ────────────────────────────────────────────────

    Adam opt(model.parameters(), /*lr=*/1e-3f);

    // zero_grad is a no-op here (grads already exist from backward), but
    // we demonstrate the call pattern used in a real training loop.
    // opt.zero_grad() would clear them — skip it so we can print the update.

    opt.step(/*clip_norm=*/1.0f);   // clip global grad norm to 1.0

    std::cout << "After one Adam step (clip_norm=1.0):\n";
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& p = params[i];
        std::string label = "  param[" + std::to_string(i) + "] data";
        print_vec(label, p->data, 6);
    }
    std::cout << "\n";

    // ── 7. Inference-mode forward (no_grad) ───────────────────────────────────
    //
    // NoGradGuard prevents any new graph nodes from being created,
    // saving memory during evaluation / inference.

    std::cout << "Inference forward (NoGradGuard):\n";
    {
        NoGradGuard ng;
        auto logits_eval = model.forward(x);
        std::cout << "  Logits " << logits_eval->repr() << "\n";
        print_vec("  data", logits_eval->data);
    }
    std::cout << "\n";

    // ── 8. Simulate a short training loop ────────────────────────────────────

    std::cout << "Mini training loop (10 steps):\n";

    // Reset optimizer with a fresh one (moment buffers at zero).
    Adam train_opt(model.parameters(), /*lr=*/1e-3f);

    for (int step = 0; step < 10; ++step) {
        // a) Zero out gradients from the previous step.
        train_opt.zero_grad();

        // b) Forward pass.
        auto out    = model.forward(x);
        auto ce     = Tensor::cross_entropy(out, targets);

        // c) Backward pass.
        ce->backward();

        // d) Update parameters.
        train_opt.step();

        std::cout << "  step " << std::setw(2) << step + 1
                  << "  loss = " << std::fixed << std::setprecision(6)
                  << ce->data[0] << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}