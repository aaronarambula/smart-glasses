#pragma once

// ─── prediction.h ────────────────────────────────────────────────────────────
// Umbrella header for the prediction module.
//
// Include this single file to pull in:
//   - TTCEngine       : time-to-collision computation (quadratic solver + CPA)
//   - TTCFrame        : per-frame TTC results + sector threat table
//   - RiskPredictor   : aaronnet MLP → RiskLevel classification + online training
//   - PredictionPipeline : convenience wrapper (TTCEngine + RiskPredictor)
//
// Dependency graph (no cycles):
//
//   autograd/autograd.h
//         │
//         ▼
//   perception/tracker.h  ←  perception/clusterer.h  ←  sensors/lidar_base.h
//         │
//         ▼
//   ttc_engine.h           (TTCEngine, TTCFrame, SectorThreat, TTCResult)
//         │
//         ▼
//   risk_predictor.h       (RiskPredictor, PseudoLabeller, PredictionResult,
//         │                  FeatureVector, RiskLevel)
//         │
//         ▼
//   prediction.h           ← this file
//
// Typical usage in the app module:
//
//   #include "prediction/prediction.h"
//   #include "perception/perception.h"
//
//   prediction::PredictionPipeline pred("aaronnet_risk.bin");
//
//   // Called once per ScanFrame (10 Hz):
//   void on_perception(const perception::PerceptionResult& perc) {
//       float density = perc.grid.probability(200, 200);  // local density proxy
//       auto [ttc, result] = pred.process(perc, density);
//
//       // ttc    → agent/scene_builder  (full trajectory context for GPT)
//       // result → audio/alert_policy   (risk level + confidence → TTS)
//   }

#include "ttc_engine.h"
#include "risk_predictor.h"

namespace prediction {

// ─── RiskLevel utilities ──────────────────────────────────────────────────────
//
// These free functions operate on RiskLevel without needing a predictor
// instance. Useful in the audio and agent modules which receive a RiskLevel
// from the pipeline but don't include the full predictor header.

// Returns a short human-readable name: "CLEAR", "CAUTION", "WARNING", "DANGER".
// (Duplicated from risk_predictor.h so callers only need this umbrella header.)
inline const char* risk_level_name(RiskLevel r) {
    return risk_name(r);
}

// Returns true if risk >= WARNING (i.e. a vocal alert is warranted).
inline bool should_alert(RiskLevel r) {
    return r >= RiskLevel::WARNING;
}

// Returns true if risk == DANGER (highest priority — interrupt any ongoing TTS).
inline bool is_danger(RiskLevel r) {
    return r == RiskLevel::DANGER;
}

// Converts a RiskLevel to a vibration/LED intensity in [0.0, 1.0].
// CLEAR=0.0, CAUTION=0.33, WARNING=0.67, DANGER=1.0.
inline float risk_to_intensity(RiskLevel r) {
    return static_cast<float>(static_cast<int>(r)) / 3.0f;
}

// ─── Sector utilities ─────────────────────────────────────────────────────────

// Returns the human-readable name of a 45° sector by index [0,7].
//   0 = "ahead"        (0°–45°)
//   1 = "ahead-right"  (45°–90°)
//   2 = "right"        (90°–135°)
//   3 = "behind-right" (135°–180°)
//   4 = "behind"       (180°–225°)
//   5 = "behind-left"  (225°–270°)
//   6 = "left"         (270°–315°)
//   7 = "ahead-left"   (315°–360°)
inline const char* sector_name(int sector) {
    switch (sector) {
        case 0: return "ahead";
        case 1: return "ahead-right";
        case 2: return "right";
        case 3: return "behind-right";
        case 4: return "behind";
        case 5: return "behind-left";
        case 6: return "left";
        case 7: return "ahead-left";
        default: return "unknown";
    }
}

// Returns the centre bearing of a sector in degrees.
inline float sector_centre_deg(int sector) {
    return sector * 45.0f + 22.5f;
}

// ─── FullPrediction ───────────────────────────────────────────────────────────
//
// Convenience aggregate combining the TTCFrame and PredictionResult into a
// single value that can be forwarded to both the audio and agent modules
// without either module needing to know about the other's inputs.
//
// Produced by PredictionPipeline::process() via structured binding, and stored
// here as a named type for when structured bindings are inconvenient (e.g.
// passing through a std::function callback).

struct FullPrediction {
    TTCFrame         ttc;
    PredictionResult prediction;

    // ── Quick-access helpers (delegates to members) ───────────────────────────

    RiskLevel risk_level()  const { return prediction.risk_level; }
    float     confidence()  const { return prediction.confidence(); }
    float     min_ttc_s()   const { return ttc.min_ttc(); }
    bool      is_danger()   const { return prediction.risk_level == RiskLevel::DANGER; }
    uint64_t  frame_id()    const { return prediction.frame_id; }

    // The most urgent forward-sector result (nullptr if no objects ahead).
    const TTCResult* forward_threat() const {
        return ttc.most_urgent_forward();
    }

    // True if any object will collide within threshold_s seconds.
    bool imminent_collision(float threshold_s = 2.5f) const {
        return ttc.has_collision_within(threshold_s);
    }

    // Builds a compact one-line log string, e.g.:
    // "[frame 42 | DANGER | conf=0.91 | TTC=1.8s | 0.7m ahead]"
    std::string log_str() const {
        std::string s = "[frame ";
        s += std::to_string(prediction.frame_id);
        s += " | ";
        s += risk_name(prediction.risk_level);
        s += " | conf=";
        // Format confidence to 2 decimal places without <sstream> overhead.
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", prediction.confidence());
        s += buf;

        const TTCResult* fwd = forward_threat();
        if (fwd) {
            if (fwd->has_ttc()) {
                std::snprintf(buf, sizeof(buf), " | TTC=%.1fs", fwd->ttc_s);
                s += buf;
            }
            std::snprintf(buf, sizeof(buf), " | %.1fm %s",
                          fwd->distance_mm / 1000.0f, fwd->size_label);
            s += buf;
        }
        s += "]";
        return s;
    }
};

} // namespace prediction