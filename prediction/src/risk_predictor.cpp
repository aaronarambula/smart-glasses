// ─── risk_predictor.cpp ───────────────────────────────────────────────────────
// Full implementation of the RiskPredictor and PseudoLabeller.
// This is where aaronnet (our custom C++ autograd engine) connects to live
// LiDAR sensor data for the first time.
//
// See include/prediction/risk_predictor.h for the full design notes.

#include "prediction/risk_predictor.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <iostream>

namespace prediction {

namespace {
bool debug_prediction_enabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("SMART_GLASSES_DEBUG_PREDICTION");
        return v && v[0] != '\0' && std::string(v) != "0";
    }();
    return enabled;
}

bool frame_is_empty(const TTCFrame& frame)
{
    if (!frame.results.empty()) return false;
    return std::none_of(frame.sectors.begin(), frame.sectors.end(),
                        [](const SectorThreat& st) { return st.occupied; });
}

bool frame_has_actionable_signal(const TTCFrame& frame)
{
    for (const auto& r : frame.results) {
        if (r.distance_mm < 4000.0f) return true;
        if (r.has_ttc()) return true;
        if (r.cpa.time_s <= 8.0f && r.cpa.distance_mm < 2000.0f) return true;
    }
    return std::any_of(frame.sectors.begin(), frame.sectors.end(),
                       [](const SectorThreat& st) { return st.occupied; });
}

void apply_probability_floor(std::array<float, RiskPredictor::NUM_CLASSES>& probs,
                             RiskLevel floor_label,
                             float min_prob = 0.60f)
{
    const size_t idx = static_cast<size_t>(static_cast<int>(floor_label));
    if (probs[idx] >= min_prob) return;

    float remaining = 1.0f - probs[idx];
    if (remaining <= 1e-6f) {
        probs.fill(0.0f);
        probs[idx] = 1.0f;
        return;
    }

    const float target_remaining = 1.0f - min_prob;
    const float scale = target_remaining / remaining;

    for (size_t i = 0; i < probs.size(); ++i) {
        if (i == idx) continue;
        probs[i] *= scale;
    }
    probs[idx] = min_prob;
}

void log_feature_vector(const FeatureVector& feat)
{
    if (!debug_prediction_enabled()) return;

    std::cout << "[debug-feat] dist=[";
    for (int i = 0; i < NUM_SECTORS; ++i) {
        if (i) std::cout << ",";
        std::cout << feat.sector_dist(i);
    }
    std::cout << "] ttc=[";
    for (int i = 0; i < NUM_SECTORS; ++i) {
        if (i) std::cout << ",";
        std::cout << feat.sector_ttc(i);
    }
    std::cout << "] dyn=["
              << feat.max_closing_speed() << ","
              << feat.num_confirmed_tracks() << ","
              << feat.global_min_ttc() << ","
              << feat.local_occ_density()
              << "]\n";
}
} // namespace

// ─── PredictionResult::summary ───────────────────────────────────────────────

std::string PredictionResult::summary() const
{
    std::ostringstream ss;
    ss << std::fixed;

    ss << "PredictionResult{"
       << "frame=" << frame_id
       << " risk=" << risk_name(risk_level)
       << std::setprecision(2)
       << " conf=" << confidence()
       << " entropy=" << entropy()
       << " probs=[";

    for (size_t i = 0; i < 4; ++i) {
        ss << std::setprecision(2) << probabilities[i];
        if (i < 3) ss << ", ";
    }

    ss << "]"
       << " pseudo=" << risk_name(pseudo_label);

    if (trained_this_frame) {
        ss << std::setprecision(4)
           << " loss=" << training_loss;
    }

    ss << "}";
    return ss.str();
}

// ─── PseudoLabeller::label_result ────────────────────────────────────────────
//
// Assigns a RiskLevel to one TTCResult based on explicit threshold rules.
// Priority order: DANGER > WARNING > CAUTION > CLEAR.
// Both distance-based and TTC-based rules fire independently — whichever
// gives the higher risk level wins.

RiskLevel PseudoLabeller::label_result(const TTCResult& r) const
{
    RiskLevel level = RiskLevel::CLEAR;

    // ── Distance-based rules ──────────────────────────────────────────────────
    if      (r.distance_mm < danger_dist_mm)  level = RiskLevel::DANGER;
    else if (r.distance_mm < warning_dist_mm) level = max_risk(level, RiskLevel::WARNING);
    else if (r.distance_mm < caution_dist_mm) level = max_risk(level, RiskLevel::CAUTION);

    // ── TTC-based rules ───────────────────────────────────────────────────────
    // TTC rules only apply if we have a valid, reliable TTC value.
    if (r.has_ttc()) {
        if      (r.ttc_s < danger_ttc_s)  level = max_risk(level, RiskLevel::DANGER);
        else if (r.ttc_s < warning_ttc_s)  level = max_risk(level, RiskLevel::WARNING);
        else if (r.ttc_s < caution_ttc_s)  level = max_risk(level, RiskLevel::CAUTION);
    }

    // ── CPA-based rule ────────────────────────────────────────────────────────
    // Crossing-path objects often have TTC = ∞ but still pass close enough to
    // matter. Use CPA distance + CPA time as a secondary risk signal.
    if (r.cpa.time_s <= danger_ttc_s && r.cpa.distance_mm < warning_dist_mm) {
        level = max_risk(level, RiskLevel::DANGER);
    } else if (r.cpa.time_s <= warning_ttc_s && r.cpa.distance_mm < warning_dist_mm) {
        level = max_risk(level, RiskLevel::WARNING);
    } else if (r.cpa.time_s <= caution_ttc_s && r.cpa.distance_mm < caution_dist_mm) {
        level = max_risk(level, RiskLevel::CAUTION);
    } else if (r.cpa.is_dangerous()) {
        level = max_risk(level, RiskLevel::WARNING);
    }

    return level;
}

// ─── PseudoLabeller::label ────────────────────────────────────────────────────
//
// Assigns a RiskLevel to a full TTCFrame by taking the maximum (worst-case)
// label across all objects. This matches the ground-truth semantics: the frame
// risk is determined by the single most dangerous object, not an average.
//
// Only confirmed-track objects are considered by default (allow_tentative=false)
// to avoid false alarms from newly spawned tracks whose velocity is unreliable.

RiskLevel PseudoLabeller::label(const TTCFrame& frame, bool allow_tentative) const
{
    RiskLevel worst = RiskLevel::CLEAR;
    const bool has_reliable = std::any_of(
        frame.results.begin(), frame.results.end(),
        [](const TTCResult& r) { return r.velocity_reliable; });

    for (const auto& r : frame.results) {
        // Prefer reliable tracks once available, but don't suppress all risk
        // when only early/tentative tracks exist.
        if (!allow_tentative && has_reliable && !r.velocity_reliable) continue;

        worst = max_risk(worst, label_result(r));

        // Short-circuit: can't get worse than DANGER.
        if (worst == RiskLevel::DANGER) break;
    }

    return worst;
}

// ─── RiskPredictor construction ───────────────────────────────────────────────
//
// Architecture: Linear(24→64) → ReLU → Linear(64→32) → ReLU → Linear(32→4)
//
// Seed offset by layer index to prevent identical weight initialisation
// across layers (He init with different seeds produces different random matrices).
//
// Adam: lr=5e-4, betas=(0.9, 0.999), eps=1e-8, clip_norm=1.0
// These are conservative settings appropriate for online single-sample training.
// Lower LR than the example (1e-3) because we're doing persistent online
// learning — we don't want to catastrophically forget earlier training.

RiskPredictor::RiskPredictor(std::string checkpoint_path,
                               unsigned int seed,
                               bool online_training)
    : online_training_(online_training)
    , checkpoint_path_(std::move(checkpoint_path))
{
    // ── Build the aaronnet MLP ────────────────────────────────────────────────
    model_.add<autograd::Linear>(INPUT_DIM,   HIDDEN1_DIM, seed)
          .add<autograd::ReLU>()
          .add<autograd::Linear>(HIDDEN1_DIM, HIDDEN2_DIM, seed + 1)
          .add<autograd::ReLU>()
          .add<autograd::Linear>(HIDDEN2_DIM, NUM_CLASSES,  seed + 2);

    // ── Adam optimiser ────────────────────────────────────────────────────────
    optimizer_ = std::make_unique<autograd::Adam>(
        model_.parameters(),
        LEARNING_RATE,
        0.9f,    // beta1
        0.999f,  // beta2
        1e-8f    // eps
    );

    // ── Label distribution counters ───────────────────────────────────────────
    label_counts_.fill(0);

    // ── Load checkpoint if it exists ──────────────────────────────────────────
    if (!checkpoint_path_.empty()) {
        // Silently ignore load failure on first run (file doesn't exist yet).
        try {
            loaded_checkpoint_ = load_weights(checkpoint_path_);
        } catch (const std::exception&) {
            // First run — start from He-initialised weights.
            loaded_checkpoint_ = false;
        }
    }
}

// ─── featurise ────────────────────────────────────────────────────────────────
//
// Converts a TTCFrame into a 24-element FeatureVector.
//
// The TTCFrame's sector threat table (already built by TTCEngine) is the
// primary data source. This keeps featurisation O(8) — independent of the
// number of tracked objects — and ensures consistent semantics between
// the sector table and the MLP input.
//
// Feature groups:
//   [0..7]  : normalised sector distances   (1.0 = empty, 0.0 = touching)
//   [8..15] : normalised sector TTCs        (1.0 = no threat, 0.0 = now)
//   [16..19]: named-sector scan density     (fraction of expected points seen)
//   [20..23]: global dynamic stats

FeatureVector RiskPredictor::featurise(const TTCFrame& frame,
                                        float local_density) const
{
    FeatureVector feat;

    // ── Groups A and B: sector distances and TTCs ─────────────────────────────
    for (int s = 0; s < NUM_SECTORS; ++s) {
        const auto& st = frame.sectors[static_cast<size_t>(s)];
        feat.sector_dist(s) = st.normalised_distance();  // [0,1], 1=empty
        feat.sector_ttc(s)  = st.normalised_ttc();       // [0,1], 1=safe
    }

    // ── Group C: named-sector density ────────────────────────────────────────
    // "Density" here is defined as how many of the 8 sectors within each
    // named quadrant are occupied (fraction of occupied sectors).
    // Forward = sectors 0 (0°–45°) and 7 (315°–360°) → 2 sectors
    // Right   = sectors 1 (45°–90°) and 2 (90°–135°) → 2 sectors
    // Rear    = sectors 3 (135°–180°) and 4 (180°–225°) → 2 sectors
    // Left    = sectors 5 (225°–270°) and 6 (270°–315°) → 2 sectors

    auto occupied_frac = [&](std::initializer_list<int> idxs) -> float {
        int occ = 0;
        for (int s : idxs) {
            if (frame.sectors[static_cast<size_t>(s)].occupied) ++occ;
        }
        return static_cast<float>(occ) / static_cast<float>(idxs.size());
    };

    feat.density_forward() = occupied_frac({0, 7});
    feat.density_right()   = occupied_frac({1, 2});
    feat.density_rear()    = occupied_frac({3, 4});
    feat.density_left()    = occupied_frac({5, 6});

    // ── Group D: global dynamic statistics ───────────────────────────────────

    // [20] Max closing speed across all results, normalised.
    float max_cs = 0.0f;
    int   confirmed_count = 0;
    for (const auto& r : frame.results) {
        if (r.closing_speed_mm_s > max_cs) max_cs = r.closing_speed_mm_s;
        if (r.velocity_reliable) ++confirmed_count;
    }
    feat.max_closing_speed() = std::clamp(max_cs / MAX_SPEED_MM_S, 0.0f, 1.0f);

    // [21] Number of confirmed (velocity-reliable) tracks, normalised.
    feat.num_confirmed_tracks() =
        std::clamp(static_cast<float>(confirmed_count) / MAX_TRACKS, 0.0f, 1.0f);

    // [22] Global minimum TTC, normalised.
    float min_ttc = frame.min_ttc();
    if (std::isfinite(min_ttc)) {
        feat.global_min_ttc() = 1.0f - std::clamp(min_ttc / MAX_TTC_S, 0.0f, 1.0f);
        // Inverted: 1.0 = imminent, 0.0 = no threat — matches risk semantics.
    } else {
        feat.global_min_ttc() = 0.0f;  // no collision predicted
    }

    // [23] Local occupancy density from the OccupancyMap (passed in externally).
    // Captures walls and surfaces that the tracker doesn't see as distinct objects.
    feat.local_occ_density() = std::clamp(local_density, 0.0f, 1.0f);

    return feat;
}

// ─── forward_inference ────────────────────────────────────────────────────────
//
// Runs the aaronnet MLP in inference mode.
// NoGradGuard ensures no computation graph is built — zero memory allocation
// for the backward pass, which is critical at 10 Hz on a Pi.
//
// Returns the (logits_tensor, softmax_probabilities) pair.
// The logits tensor is retained so the caller can pass it to cross_entropy
// during a training step (though the gradient graph won't exist — training
// re-runs forward with grad enabled).

std::pair<autograd::TensorPtr, std::array<float, RiskPredictor::NUM_CLASSES>>
RiskPredictor::forward_inference(const FeatureVector& features)
{
    // Pack the 24-element feature array into a (1, 24) row tensor.
    std::vector<float> feat_vec(features.data.begin(), features.data.end());
    auto x = autograd::make_tensor(std::move(feat_vec),
                                   /*rows=*/1,
                                   /*cols=*/INPUT_DIM,
                                   /*requires_grad=*/false);

    // Forward pass — no gradient tracking.
    autograd::TensorPtr logits;
    {
        autograd::NoGradGuard ng;
        logits = model_.forward(x);
    }

    // Softmax over the 4 class logits to get probabilities.
    autograd::TensorPtr probs_tensor;
    {
        autograd::NoGradGuard ng;
        probs_tensor = logits->softmax();
    }

    // Copy probabilities into a plain array.
    std::array<float, NUM_CLASSES> probs{};
    assert(probs_tensor->numel() == NUM_CLASSES);
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        probs[i] = probs_tensor->data[i];
    }

    return {logits, probs};
}

// ─── train_step ───────────────────────────────────────────────────────────────
//
// One full supervised training step:
//   1. Forward pass WITH gradient tracking (requires_grad=true on input not
//      needed — only the parameters carry gradients; the input is just data).
//   2. Cross-entropy loss against the pseudo-label.
//   3. Backward pass through the aaronnet graph.
//   4. Adam step with gradient clipping (clip_norm=1.0).
//
// The forward pass here re-runs through the model with gradient tracking
// enabled (enable_grad = true by default). This is separate from
// forward_inference() which uses NoGradGuard.
//
// Returns the scalar loss value.

float RiskPredictor::train_step(const FeatureVector& features, RiskLevel label)
{
    // Pack features into input tensor. The input itself doesn't need grad —
    // only the model's weight and bias tensors carry requires_grad=true.
    std::vector<float> feat_vec(features.data.begin(), features.data.end());
    auto x = autograd::make_tensor(std::move(feat_vec),
                                   /*rows=*/1,
                                   /*cols=*/INPUT_DIM,
                                   /*requires_grad=*/false);

    // Zero previous gradients.
    optimizer_->zero_grad();

    // Forward pass — gradient graph IS built here.
    auto logits = model_.forward(x);

    // Cross-entropy loss (numerically stable log-softmax internally).
    auto loss = autograd::Tensor::cross_entropy(
        logits,
        {static_cast<int>(label)}   // integer class index
    );

    // Backward pass — propagates gradient through all Linear layers.
    loss->backward();

    // Adam update with gradient norm clipping.
    optimizer_->step(CLIP_NORM);

    return loss->data[0];
}

// ─── update_smoothed_loss ────────────────────────────────────────────────────
//
// Exponential moving average of training loss.
// α = 0.05: slow-moving average that tracks long-term convergence trends.
// Displayed in the verbose output so the user can see if training is stable.

void RiskPredictor::update_smoothed_loss(float loss)
{
    constexpr float ALPHA = 0.05f;
    if (training_steps_ <= 1) {
        smoothed_loss_ = loss;
    } else {
        smoothed_loss_ = ALPHA * loss + (1.0f - ALPHA) * smoothed_loss_;
    }
}

// ─── predict ──────────────────────────────────────────────────────────────────
//
// Main entry point. Called once per ScanFrame at ~10 Hz.
//
// Inference is always done first (NoGradGuard), then training runs every
// TRAIN_EVERY_N frames with gradient enabled. This ordering matters:
//   - The prediction the user hears is always based on the current weights.
//   - The training update uses the same frame's features and pseudo-label.
//   - Because we train AFTER predicting, the prediction is never corrupted
//     by a partially-updated gradient state.
//
// Checkpoint is saved every SAVE_EVERY_N training steps to avoid excessive
// I/O (each save is ~10 KB, negligible, but file writes on SD card have
// unpredictable latency — we don't want them on the hot path every frame).

PredictionResult RiskPredictor::predict(const TTCFrame& frame,
                                         float local_density)
{
    ++frames_processed_;

    // ── 1. Featurise ──────────────────────────────────────────────────────────
    FeatureVector features = featurise(frame, local_density);
    log_feature_vector(features);

    // ── 2. Inference (no grad) ────────────────────────────────────────────────
    auto [logits, probs] = forward_inference(features);

    // ── 3. Argmax → predicted class ───────────────────────────────────────────
    int pred_idx = static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin());
    RiskLevel predicted = static_cast<RiskLevel>(pred_idx);

    // ── 4. Pseudo-label ───────────────────────────────────────────────────────
    // Use allow_tentative=true for the first 30 frames (300 ms at 10 Hz)
    // so training starts immediately even before Kalman velocities settle.
    const bool empty_scene = frame_is_empty(frame);
    const bool has_reliable = std::any_of(
        frame.results.begin(), frame.results.end(),
        [](const TTCResult& r) { return r.velocity_reliable; });
    bool allow_tentative = (frames_processed_ < 30) || !has_reliable;
    RiskLevel pseudo = labeller_.label(frame, allow_tentative);

    if (empty_scene) {
        pseudo = RiskLevel::CLEAR;
    }

    // Update label distribution counters.
    label_counts_[static_cast<size_t>(static_cast<int>(pseudo))]++;

    // ── 5. Online training (every TRAIN_EVERY_N frames) ───────────────────────
    float loss = 0.0f;
    bool trained = false;

    const bool actionable_scene = !empty_scene && frame_has_actionable_signal(frame);

    if (online_training_ &&
        actionable_scene &&
        (frames_processed_ % TRAIN_EVERY_N == 0)) {
        loss = train_step(features, pseudo);
        last_loss_ = loss;
        ++training_steps_;
        update_smoothed_loss(loss);
        trained = true;

        // ── 6. Checkpoint save ────────────────────────────────────────────────
        if (!checkpoint_path_.empty() &&
            training_steps_ % SAVE_EVERY_N == 0) {
            try {
                save_weights(checkpoint_path_);
            } catch (const std::exception&) {
                // Non-fatal — log is available via last_loss_ / smoothed_loss_.
            }
        }
    }

    // ── 7. Select user-facing label ───────────────────────────────────────────
    // Fresh models start from random He-initialised weights, which can produce
    // pathological outputs (for example always predicting DANGER) before enough
    // online updates have happened. When we did not load a checkpoint, use the
    // heuristic pseudo-label during an initial bootstrap window.
    const bool bootstrap_mode =
        !loaded_checkpoint_ &&
        training_steps_ < BOOTSTRAP_TRAINING_STEPS;

    if (bootstrap_mode) {
        predicted = pseudo;
        probs.fill(0.0f);
        probs[static_cast<size_t>(static_cast<int>(pseudo))] = 1.0f;
    } else if (pseudo > predicted) {
        predicted = pseudo;
        apply_probability_floor(probs, pseudo);
    }

    if (debug_prediction_enabled()) {
        std::cout << "[debug-pred] frame=" << frame.frame_id
                  << " pred=" << risk_name(predicted)
                  << " pseudo=" << risk_name(pseudo)
                  << " probs=["
                  << probs[0] << "," << probs[1] << ","
                  << probs[2] << "," << probs[3] << "]"
                  << " trained=" << trained
                  << " empty=" << empty_scene
                  << " actionable=" << actionable_scene
                  << "\n";
    }

    // ── 8. Assemble result ────────────────────────────────────────────────────
    PredictionResult result;
    result.risk_level          = predicted;
    result.probabilities       = probs;
    result.features            = features;
    result.pseudo_label        = pseudo;
    result.training_loss       = loss;
    result.trained_this_frame  = trained;
    result.frame_id            = frame.frame_id;

    return result;
}

// ─── save_weights ─────────────────────────────────────────────────────────────
//
// Binary checkpoint format:
//
//   Offset  Size  Field
//   ──────  ────  ─────────────────────────────────────────────────────────────
//   0        4    Magic number: 0x4E4E5241 ("ARNN" little-endian)
//   4        4    Version:      2
//   8        4    num_params:   number of parameter tensors in the model
//   12       4    INPUT_DIM:    sanity check (must match at load time)
//   16       4    training_steps: restore training step counter
//   20       4    smoothed_loss:  restore EMA loss
//   24+      ...  For each parameter tensor:
//                   4   numel: number of float elements
//                   4*numel   float data (little-endian, IEEE 754)
//
// The format is intentionally simple — no compression, no endian conversion
// beyond what the host CPU provides (Pi is always little-endian).

void RiskPredictor::save_weights(const std::string& path) const
{
    if (path.empty()) return;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw std::runtime_error("RiskPredictor::save_weights: cannot open '"
                                 + path + "' for writing");
    }

    auto write_u32 = [&](uint32_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto write_f32 = [&](float v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };

    // Header
    write_u32(CHECKPOINT_MAGIC);
    write_u32(CHECKPOINT_VERSION);

    auto params = const_cast<autograd::Sequential&>(model_).parameters();
    write_u32(static_cast<uint32_t>(params.size()));
    write_u32(static_cast<uint32_t>(INPUT_DIM));
    write_u32(static_cast<uint32_t>(training_steps_));
    write_f32(smoothed_loss_);

    // Parameter tensors
    for (const auto& p : params) {
        uint32_t numel = static_cast<uint32_t>(p->numel());
        write_u32(numel);
        f.write(reinterpret_cast<const char*>(p->data.data()),
                numel * sizeof(float));
    }

    if (!f) {
        throw std::runtime_error("RiskPredictor::save_weights: write error on '"
                                 + path + "'");
    }
}

// ─── load_weights ─────────────────────────────────────────────────────────────
//
// Loads a checkpoint written by save_weights().
// Validates magic number, version, param count, and INPUT_DIM before
// overwriting any model weights — partial loads are never committed.
// Returns false (without throwing) if path is empty.
// Throws std::runtime_error on any corruption or shape mismatch.

bool RiskPredictor::load_weights(const std::string& path)
{
    if (path.empty()) return false;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        // File doesn't exist yet — first run.
        return false;
    }

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    auto read_f32 = [&]() -> float {
        float v = 0.0f;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };

    // ── Validate header ───────────────────────────────────────────────────────
    uint32_t magic   = read_u32();
    uint32_t version = read_u32();

    if (magic != CHECKPOINT_MAGIC) {
        throw std::runtime_error(
            "RiskPredictor::load_weights: bad magic in '" + path + "'");
    }
    if (version != CHECKPOINT_VERSION) {
        throw std::runtime_error(
            "RiskPredictor::load_weights: version mismatch in '" + path +
            "' (got " + std::to_string(version) +
            ", expected " + std::to_string(CHECKPOINT_VERSION) + ")");
    }

    uint32_t num_params  = read_u32();
    uint32_t saved_dim   = read_u32();
    uint32_t saved_steps = read_u32();
    float    saved_loss  = read_f32();

    if (saved_dim != static_cast<uint32_t>(INPUT_DIM)) {
        throw std::runtime_error(
            "RiskPredictor::load_weights: INPUT_DIM mismatch in '" + path + "'");
    }

    auto params = model_.parameters();
    if (num_params != static_cast<uint32_t>(params.size())) {
        throw std::runtime_error(
            "RiskPredictor::load_weights: param count mismatch in '" + path +
            "' (got " + std::to_string(num_params) +
            ", model has " + std::to_string(params.size()) + ")");
    }

    // ── Load parameter tensors ────────────────────────────────────────────────
    // Read into temporary buffers first — only commit on full success.
    std::vector<std::vector<float>> loaded_data(num_params);

    for (uint32_t i = 0; i < num_params; ++i) {
        uint32_t numel = read_u32();

        if (numel != static_cast<uint32_t>(params[i]->numel())) {
            throw std::runtime_error(
                "RiskPredictor::load_weights: shape mismatch for param " +
                std::to_string(i) + " in '" + path + "'");
        }

        loaded_data[i].resize(numel);
        f.read(reinterpret_cast<char*>(loaded_data[i].data()),
               numel * sizeof(float));

        if (!f) {
            throw std::runtime_error(
                "RiskPredictor::load_weights: read error for param " +
                std::to_string(i) + " in '" + path + "'");
        }
    }

    // ── Commit: overwrite model weights ───────────────────────────────────────
    for (uint32_t i = 0; i < num_params; ++i) {
        params[i]->data = std::move(loaded_data[i]);
        // Clear any stale gradient from a previous session.
        params[i]->grad.clear();
    }

    // Restore training state counters.
    training_steps_ = static_cast<int>(saved_steps);
    smoothed_loss_  = saved_loss;
    last_loss_      = saved_loss;

    // Reset Adam moment buffers — they are not saved because they are
    // session-specific (learning rate schedule, data distribution).
    // Starting with zeroed moments is equivalent to a warm restart.
    optimizer_ = std::make_unique<autograd::Adam>(
        model_.parameters(),
        LEARNING_RATE,
        0.9f,
        0.999f,
        1e-8f
    );
    // Restore the timestep so bias-correction stays correct.
    // We do this by advancing the internal step counter.
    for (int s = 0; s < training_steps_; ++s) {
        // We can't directly set t_ in Adam (it's private), so we advance
        // it via a no-op step with zero-grad parameters.
        // Instead, we simply re-set it via set_lr which doesn't touch t_,
        // and accept that bias correction resets to step 1 after a reload.
        // This is a minor inaccuracy that self-corrects after ~10 steps.
        (void)s;
        break;
    }

    return true;
}

} // namespace prediction
