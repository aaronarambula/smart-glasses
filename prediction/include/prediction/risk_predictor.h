#pragma once

// perception.h is included here so that PredictionPipeline::process() can
// accept a perception::PerceptionResult by const reference.

// ─── risk_predictor.h ────────────────────────────────────────────────────────
// The bridge between aaronnet (our custom autograd engine) and the real-time
// LiDAR perception pipeline.
//
// This is where the autograd engine stops being a standalone demo and becomes
// a live component of an embedded AI system running on a Raspberry Pi.
//
// Design
// ──────
// The RiskPredictor owns a aaronnet Sequential MLP that maps a fixed-length
// feature vector extracted from a TTCFrame → a 4-class risk probability
// distribution over {CLEAR, CAUTION, WARNING, DANGER}.
//
// Architecture:  Linear(24→64) → ReLU → Linear(64→32) → ReLU → Linear(32→4)
//
// Why these numbers:
//   24 inputs  : 8 sector min-distances + 8 sector min-TTCs +
//                4 named-sector densities + 4 velocity/dynamic stats
//   64 hidden  : wide enough to learn sector interaction patterns
//                (e.g. "fast object on left + wall ahead = worse than either alone")
//   32 hidden  : compression layer — forces the network to distil to abstract risk
//   4 outputs  : one logit per RiskLevel class
//
// Feature vector (24 floats, all normalised to [0, 1])
// ─────────────────────────────────────────────────────
// Group A — Sector minimum distances (8 features)
//   [0..7]  normalised_distance() for each of the 8 × 45° sectors.
//           1.0 = nothing detected (max range), 0.0 = right on top of user.
//           This is the primary spatial awareness signal.
//
// Group B — Sector minimum TTCs (8 features)
//   [8..15] normalised_ttc() for each of the 8 × 45° sectors.
//           1.0 = no imminent collision, 0.0 = collision now.
//           This makes the network time-aware, not just distance-aware.
//
// Group C — Named-sector density (4 features)
//   [16]  fraction of valid scan points in forward sector  (0° ± 30°)
//   [17]  fraction of valid scan points in right sector    (30°–150°)
//   [18]  fraction of valid scan points in rear sector     (150°–210°)
//   [19]  fraction of valid scan points in left sector     (210°–330°)
//   Density captures how "cluttered" each zone is, independent of distance.
//   A narrow gap between two walls registers as two close clusters; density
//   captures that the whole forward area is full.
//
// Group D — Dynamic / global stats (4 features)
//   [20]  max closing speed across all confirmed objects  (÷ 3000 mm/s, clamp 1)
//         Distinguishes a fast-moving person from a stationary wall at same dist.
//   [21]  number of confirmed tracks                      (÷ 10, clamp 1)
//         Scene complexity — more objects = more uncertainty.
//   [22]  min TTC across all objects                      (normalised, 1=none)
//         Global urgency signal independent of sector layout.
//   [23]  local occupancy density from OccupancyMap       (fraction [0,1])
//         Grid-based measure — captures walls/surfaces the tracker doesn't see
//         as distinct objects (e.g. a long continuous wall is one cluster but
//         many occupied cells).
//
// Pseudo-label heuristic (bootstrap training)
// ───────────────────────────────────────────
// Real labelled data doesn't exist on day one. The PseudoLabeller converts
// a TTCFrame into a RiskLevel using explicit threshold rules — the same logic
// as the old classify_distance() function but richer. These pseudo-labels let
// the MLP train from the very first frame, learning a smooth generalisation of
// the hard threshold rules. After real labels accumulate (e.g. operator
// feedback via a button press), the pseudo-labels are replaced.
//
// Heuristic label rules (in priority order):
//   DANGER  : any object with TTC < 2s,  OR distance < 500mm
//   WARNING : any object with TTC < 4s,  OR distance < 1000mm
//   CAUTION : any object with TTC < 8s,  OR distance < 2000mm
//   CLEAR   : otherwise
//
// Online training
// ───────────────
// The predictor can fine-tune its weights on every N-th frame using
// cross-entropy loss + Adam. This is safe to do on the inference thread
// because: (a) the model is tiny, (b) we use a small LR, (c) we only train
// every TRAIN_EVERY_N frames to amortise the cost.
//
// Training costs ~0.5ms per step on a Pi 4 (24-input MLP, batch size 1).
// At 10 Hz with N=5, that's one training step every 0.5s → negligible.
//
// Weight persistence
// ──────────────────
// Weights are saved to a binary checkpoint after every SAVE_EVERY_N training
// steps, and loaded at construction if the file exists. This means the model
// improves run-to-run — it gets better every time the glasses are worn.
//
// Thread safety
// ─────────────
// RiskPredictor is NOT thread-safe. It must be called from a single thread
// (the pipeline thread). The audio and agent modules receive copies of the
// output (PredictionResult is a value type), not references into the predictor.

#include "ttc_engine.h"

#include "perception/perception.h"

#include "autograd/autograd.h"

#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cassert>

namespace prediction {

// ─── RiskLevel ────────────────────────────────────────────────────────────────
//
// Four-class ordinal risk taxonomy.
// Ordinal: DANGER > WARNING > CAUTION > CLEAR in severity.
// The integer values are the class indices used by the MLP cross-entropy loss.

enum class RiskLevel : int {
    CLEAR   = 0,
    CAUTION = 1,
    WARNING = 2,
    DANGER  = 3,
};

inline const char* risk_name(RiskLevel r) {
    switch (r) {
        case RiskLevel::CLEAR:   return "CLEAR";
        case RiskLevel::CAUTION: return "CAUTION";
        case RiskLevel::WARNING: return "WARNING";
        case RiskLevel::DANGER:  return "DANGER";
        default:                 return "UNKNOWN";
    }
}

// Ordinal comparison helpers.
inline bool operator>(RiskLevel a, RiskLevel b) {
    return static_cast<int>(a) > static_cast<int>(b);
}
inline bool operator>=(RiskLevel a, RiskLevel b) {
    return static_cast<int>(a) >= static_cast<int>(b);
}
inline RiskLevel max_risk(RiskLevel a, RiskLevel b) {
    return (a > b) ? a : b;
}

// ─── FeatureVector ────────────────────────────────────────────────────────────
//
// The 24-float input to the MLP. Stored as a named struct so callers can
// inspect individual features for debugging without magic index offsets.

struct FeatureVector {
    static constexpr size_t DIM = 24;

    std::array<float, DIM> data{};

    // Named accessors for each feature group.
    // Group A: sector distances [0..7]
    float& sector_dist(int s)       { return data[s]; }
    float  sector_dist(int s) const { return data[s]; }

    // Group B: sector TTCs [8..15]
    float& sector_ttc(int s)        { return data[8 + s]; }
    float  sector_ttc(int s)  const { return data[8 + s]; }

    // Group C: named-sector densities [16..19]
    float& density_forward()        { return data[16]; }
    float  density_forward()  const { return data[16]; }
    float& density_right()          { return data[17]; }
    float  density_right()    const { return data[17]; }
    float& density_rear()           { return data[18]; }
    float  density_rear()     const { return data[18]; }
    float& density_left()           { return data[19]; }
    float  density_left()     const { return data[19]; }

    // Group D: dynamic stats [20..23]
    float& max_closing_speed()      { return data[20]; }
    float  max_closing_speed() const{ return data[20]; }
    float& num_confirmed_tracks()   { return data[21]; }
    float  num_confirmed_tracks() const { return data[21]; }
    float& global_min_ttc()         { return data[22]; }
    float  global_min_ttc()   const { return data[22]; }
    float& local_occ_density()      { return data[23]; }
    float  local_occ_density() const{ return data[23]; }

    // Flat pointer for passing to make_tensor.
    const float* ptr() const { return data.data(); }
    float*       ptr()       { return data.data(); }
};

// ─── PredictionResult ────────────────────────────────────────────────────────
//
// Output of one RiskPredictor::predict() call.
// Value type — safe to copy and move across thread boundaries.

struct PredictionResult {
    // ── Classification ────────────────────────────────────────────────────────
    RiskLevel risk_level = RiskLevel::CLEAR;

    // Softmax probabilities for all 4 classes [CLEAR, CAUTION, WARNING, DANGER].
    std::array<float, 4> probabilities{};

    // Confidence = probability of the predicted class.
    float confidence() const {
        return probabilities[static_cast<int>(risk_level)];
    }

    // Entropy of the probability distribution [0, log(4)].
    // High entropy → model is uncertain.
    float entropy() const {
        float h = 0.0f;
        for (float p : probabilities) {
            if (p > 1e-9f) h -= p * std::log(p);
        }
        return h;
    }

    // ── Feature snapshot ──────────────────────────────────────────────────────
    FeatureVector features;   // the input that produced this prediction

    // ── Training info ─────────────────────────────────────────────────────────
    RiskLevel pseudo_label  = RiskLevel::CLEAR;  // heuristic label used for training
    float     training_loss = 0.0f;              // cross-entropy loss (0 if not trained)
    bool      trained_this_frame = false;

    // ── Source frame ──────────────────────────────────────────────────────────
    uint64_t  frame_id = 0;

    // ── Human-readable summary ────────────────────────────────────────────────
    std::string summary() const;
};

// ─── PseudoLabeller ──────────────────────────────────────────────────────────
//
// Converts a TTCFrame into a RiskLevel using explicit threshold rules.
// Used to bootstrap MLP training without any labelled data.
//
// Stateless — all methods are const or static.

class PseudoLabeller {
public:
    // Thresholds (all configurable at construction).
    float danger_ttc_s    = 2.0f;     // TTC < this → DANGER
    float warning_ttc_s   = 4.0f;     // TTC < this → WARNING
    float caution_ttc_s   = 8.0f;     // TTC < this → CAUTION

    float danger_dist_mm  = 500.0f;   // distance < this → DANGER
    float warning_dist_mm = 1000.0f;  // distance < this → WARNING
    float caution_dist_mm = 2000.0f;  // distance < this → CAUTION

    // Assign a RiskLevel to a TTCFrame.
    // Only considers confirmed, velocity-reliable objects (or all objects if
    // allow_tentative = true, for early-boot sensitivity).
    RiskLevel label(const TTCFrame& frame, bool allow_tentative = false) const;

    // Assign a RiskLevel to a single TTCResult.
    RiskLevel label_result(const TTCResult& r) const;
};

// ─── RiskPredictor ────────────────────────────────────────────────────────────

class RiskPredictor {
public:
    // ── Dimensions ────────────────────────────────────────────────────────────
    static constexpr size_t INPUT_DIM    = FeatureVector::DIM;   // 24
    static constexpr size_t HIDDEN1_DIM  = 64;
    static constexpr size_t HIDDEN2_DIM  = 32;
    static constexpr size_t NUM_CLASSES  = 4;

    // ── Training hyperparameters ───────────────────────────────────────────────
    static constexpr int   TRAIN_EVERY_N = 5;    // train on every Nth frame
    static constexpr int   SAVE_EVERY_N  = 200;  // save weights every N training steps
    static constexpr int   BOOTSTRAP_TRAINING_STEPS = 20; // heuristic labels until model settles
    static constexpr float LEARNING_RATE = 5e-4f;
    static constexpr float CLIP_NORM     = 1.0f;
    static constexpr float MAX_RANGE_MM  = 6000.0f;
    static constexpr float MAX_SPEED_MM_S = 3000.0f;
    static constexpr float MAX_TRACKS    = 10.0f;

    // ── Construction ──────────────────────────────────────────────────────────

    // checkpoint_path : path to save/load binary weights.
    //                   Pass "" to disable persistence.
    // seed            : RNG seed for He weight init (deterministic on Pi).
    // online_training : if false, inference-only (weights frozen).
    explicit RiskPredictor(std::string checkpoint_path = "aaronnet_risk.bin",
                            unsigned int seed          = 42,
                            bool online_training       = true);

    // Non-copyable (owns heap-allocated MLP weights).
    RiskPredictor(const RiskPredictor&)            = delete;
    RiskPredictor& operator=(const RiskPredictor&) = delete;

    // ── Main entry point ──────────────────────────────────────────────────────

    // Run one predict-then-maybe-train cycle on a TTCFrame.
    //
    // Steps:
    //   1. Featurise the TTCFrame into a 24-element FeatureVector.
    //   2. Forward pass through the MLP (NoGradGuard — no graph built).
    //   3. Softmax → argmax → predicted RiskLevel.
    //   4. Generate pseudo-label via PseudoLabeller.
    //   5. Every TRAIN_EVERY_N frames: Adam update with cross-entropy loss.
    //   6. Every SAVE_EVERY_N training steps: save checkpoint.
    //   7. Return PredictionResult.
    //
    // local_density: the OccupancyMap local density value for feature [23].
    //                Pass 0.0f if the map is unavailable.
    PredictionResult predict(const TTCFrame& frame,
                              float local_density = 0.0f);

    // ── Featurisation (public for logging / debugging) ─────────────────────────

    // Extract the 24-element feature vector from a TTCFrame.
    // Uses sector threat table from TTCFrame::sectors directly —
    // no re-scanning of the object list needed.
    FeatureVector featurise(const TTCFrame& frame,
                            float local_density = 0.0f) const;

    // ── Weight persistence ────────────────────────────────────────────────────

    // Save all MLP parameter tensors to a flat binary file.
    // Format: magic(4B) | version(4B) | num_params(4B) |
    //         for each param: { numel(4B) | floats(numel×4B) }
    void save_weights(const std::string& path) const;

    // Load weights from a file written by save_weights().
    // Throws std::runtime_error if the file is missing or corrupted.
    // Returns false (without throwing) if the path is empty string.
    bool load_weights(const std::string& path);

    // ── Diagnostics ───────────────────────────────────────────────────────────

    int      frames_processed()  const { return frames_processed_; }
    int      training_steps()    const { return training_steps_; }
    float    last_loss()         const { return last_loss_; }
    bool     online_training()   const { return online_training_; }

    // Running exponential moving average of the training loss (α=0.05).
    float    smoothed_loss()     const { return smoothed_loss_; }

    // Distribution of pseudo-labels seen so far (for class balance monitoring).
    const std::array<int, NUM_CLASSES>& label_counts() const {
        return label_counts_;
    }

    // Access to the underlying model (for external inspection / export).
    autograd::Sequential&       model()       { return model_; }
    const autograd::Sequential& model() const { return model_; }

    // Access to the pseudo-labeller (to override thresholds at runtime).
    PseudoLabeller&       labeller()       { return labeller_; }
    const PseudoLabeller& labeller() const { return labeller_; }

private:
    // ── aaronnet model ────────────────────────────────────────────────────────
    autograd::Sequential              model_;
    std::unique_ptr<autograd::Adam>   optimizer_;

    // ── Training state ────────────────────────────────────────────────────────
    bool         online_training_;
    int          frames_processed_   = 0;
    int          training_steps_     = 0;
    float        last_loss_          = 0.0f;
    float        smoothed_loss_      = 0.0f;

    // Label distribution counters for class-balance logging.
    std::array<int, NUM_CLASSES> label_counts_{};

    // ── Persistence ───────────────────────────────────────────────────────────
    std::string checkpoint_path_;
    bool        loaded_checkpoint_ = false;

    // ── Pseudo-labeller ───────────────────────────────────────────────────────
    PseudoLabeller labeller_;

    // ── File format constants ─────────────────────────────────────────────────
    static constexpr uint32_t CHECKPOINT_MAGIC   = 0x4E4E5241;  // "ARNN"
    static constexpr uint32_t CHECKPOINT_VERSION = 2;

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Run one forward pass and return the (logits_tensor, probs_array) pair.
    // Uses NoGradGuard — no computation graph is built.
    std::pair<autograd::TensorPtr, std::array<float, NUM_CLASSES>>
    forward_inference(const FeatureVector& features);

    // Run one Adam training step given a feature vector and a target label.
    // Returns the scalar cross-entropy loss.
    float train_step(const FeatureVector& features, RiskLevel label);

    // Update the smoothed loss EMA.
    void update_smoothed_loss(float loss);
};

// ─── PredictionPipeline ───────────────────────────────────────────────────────
//
// Convenience wrapper that owns both the TTCEngine and the RiskPredictor and
// runs them in sequence from a PerceptionResult.
//
// This is the single object the app module instantiates for the prediction layer.
// Call process() once per PerceptionResult (i.e., once per ScanFrame at 10 Hz).

class PredictionPipeline {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // checkpoint_path : passed through to RiskPredictor.
    //                   Set "" to start fresh every run (useful during testing).
    // online_training : enable/disable MLP fine-tuning.
    explicit PredictionPipeline(std::string checkpoint_path = "aaronnet_risk.bin",
                                 bool online_training       = true)
        : risk_predictor_(std::move(checkpoint_path),
                          /*seed=*/42,
                          online_training)
    {}

    // Non-copyable.
    PredictionPipeline(const PredictionPipeline&)            = delete;
    PredictionPipeline& operator=(const PredictionPipeline&) = delete;

    // ── Main entry point ──────────────────────────────────────────────────────

    // Run the full prediction stack on one PerceptionResult.
    //
    //   1. TTCEngine::compute()       → TTCFrame
    //   2. RiskPredictor::predict()   → PredictionResult
    //
    // Returns both so the app module can forward each to the appropriate sink:
    //   TTCFrame        → agent/scene_builder
    //   PredictionResult → audio/alert_policy
    //
    // local_density: from OccupancyMap::local_density(1500.0f) — the fraction
    //               of cells within 1.5m of the user that are occupied.
    struct Output {
        TTCFrame         ttc;
        PredictionResult prediction;
    };

    Output process(const perception::PerceptionResult& perc,
                   float local_density = 0.0f) {
        Output out;
        out.ttc        = ttc_engine_.compute(perc.objects,
                                              perc.frame_id,
                                              perc.dt_s);
        out.prediction = risk_predictor_.predict(out.ttc, local_density);
        return out;
    }

    // ── Component access ──────────────────────────────────────────────────────

    TTCEngine&       ttc_engine()       { return ttc_engine_; }
    const TTCEngine& ttc_engine() const { return ttc_engine_; }

    RiskPredictor&       risk_predictor()       { return risk_predictor_; }
    const RiskPredictor& risk_predictor() const { return risk_predictor_; }

private:
    TTCEngine     ttc_engine_;
    RiskPredictor risk_predictor_;
};

} // namespace prediction
