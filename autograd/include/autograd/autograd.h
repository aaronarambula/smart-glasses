#pragma once

// ─── autograd.h ──────────────────────────────────────────────────────────────
// Umbrella header — include this single file to pull in the entire autograd
// engine: tensors, ops, layers, optimizer, and the no_grad guard.
//
// Usage:
//   #include "autograd/autograd.h"
//
//   using namespace autograd;
//
//   auto x      = make_tensor({1,2,3,4}, 1, 4);
//   Sequential model;
//   model.add<Linear>(4, 8).add<ReLU>().add<Linear>(8, 3);
//   Adam opt(model.parameters(), 1e-3f);
//
//   auto logits = model.forward(x);
//   auto loss   = Tensor::cross_entropy(logits, {0});
//   loss->backward();
//   opt.step();

#include "no_grad.h"
#include "ops.h"
#include "tensor.h"
#include "layers.h"
#include "optimizer.h"