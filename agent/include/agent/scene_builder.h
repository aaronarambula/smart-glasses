#pragma once

// ─── scene_builder.h ─────────────────────────────────────────────────────────
// Converts the current perception + prediction state into a compact JSON
// string that is sent as the user message to the OpenAI Chat Completions API.
//
// Design
// ──────
// GPT-4o has a context window of 128k tokens, but we must be stingy: each
// agent query runs every 5 seconds and we want latency under 3 seconds on a
// cheap Raspberry Pi internet connection. The JSON payload must stay under
// ~400 tokens so the model can respond in one short sentence within budget.
//
// The scene JSON has this shape (all distances in metres, speeds in m/s):
//
//   {
//     "frame": 1042,
//     "risk": "WARNING",
//     "confidence": 0.87,
//     "min_ttc_s": 3.1,
//     "objects": [
//       {
//         "id": 3,
//         "dir": "ahead",
//         "dist_m": 1.2,
//         "size": "medium",
//         "speed_m_s": 0.4,
//         "closing_m_s": 0.4,
//         "ttc_s": 3.1,
//         "moving": true
//       },
//       ...  (max MAX_OBJECTS entries, sorted by urgency)
//     ],
//     "sectors": {
//       "ahead":       { "dist_m": 1.2, "ttc_s": 3.1 },
//       "ahead-right": { "dist_m": 4.1, "ttc_s": null },
//       ...  (only occupied sectors emitted)
//     },
//     "local_density": 0.12,
//     "training": { "steps": 42, "loss": 0.31 }
//   }
//
// "training" block is omitted when training_steps == 0.
// Null TTC values are emitted as JSON null (not infinity — JSON has no Inf).
// Objects beyond MAX_OBJECTS are silently truncated (they are already sorted
// by urgency descending so we keep the most important ones).
//
// Token budget analysis (approximate):
//   Structural keys + values (3 objects, 5 sectors): ~220 tokens
//   System prompt:                                   ~120 tokens
//   Response budget:                                 ~ 60 tokens
//   Total:                                           ~400 tokens  ✓
//
// Thread safety
// ─────────────
// SceneBuilder is stateless — build() is a pure function of its inputs.
// Safe to call from any thread.

#include "prediction/prediction.h"
#include "perception/perception.h"

#include <string>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <array>

namespace agent {

// ─── SceneBuilderConfig ───────────────────────────────────────────────────────

struct SceneBuilderConfig {
    // Maximum number of tracked objects to include in the JSON.
    // Objects are already sorted by urgency — we take the first MAX_OBJECTS.
    size_t max_objects = 4;

    // Maximum number of occupied sectors to emit in the "sectors" block.
    // Sectors are sorted by urgency (min TTC / min distance).
    size_t max_sectors = 5;

    // If true, include the "training" diagnostics block.
    bool include_training_info = true;

    // If true, include sector detail block.
    bool include_sectors = true;

    // Decimal precision for distance and speed values in the JSON.
    int float_precision = 2;
};

// ─── SceneBuilder ─────────────────────────────────────────────────────────────

class SceneBuilder {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    explicit SceneBuilder(SceneBuilderConfig config = SceneBuilderConfig{})
        : config_(std::move(config))
    {}

    // ── Main entry point ──────────────────────────────────────────────────────

    // Build a compact JSON scene string from a FullPrediction and optional
    // PerceptionResult. The JSON is suitable for use as the user-turn content
    // in a Chat Completions request.
    //
    // perc_ptr: pointer to the most recent PerceptionResult. May be nullptr
    //           if the perception pipeline has not yet produced a frame.
    //
    // training_steps / training_loss: forwarded from RiskPredictor diagnostics
    //           so the agent can acknowledge the model's learning state.
    std::string build(const prediction::FullPrediction& pred,
                      const perception::PerceptionResult* perc_ptr = nullptr,
                      int   training_steps = 0,
                      float training_loss  = 0.0f) const;

    // ── Component builders (public for unit testing) ───────────────────────────

    // Serialise one TTCResult as a JSON object (without outer braces).
    std::string build_object_json(const prediction::TTCResult& r) const;

    // Serialise the 8-sector threat table as a JSON object.
    // Only emits occupied sectors, up to config_.max_sectors entries.
    std::string build_sectors_json(
        const std::array<prediction::SectorThreat,
                         prediction::NUM_SECTORS>& sectors) const;

    // ── Config access ─────────────────────────────────────────────────────────

    SceneBuilderConfig&       config()       { return config_; }
    const SceneBuilderConfig& config() const { return config_; }

private:
    SceneBuilderConfig config_;

    // ── Formatting helpers ────────────────────────────────────────────────────

    // Format a float to config_.float_precision decimal places.
    std::string fmt(float v) const;

    // Format a float or emit "null" if it is not finite.
    std::string fmt_nullable(float v) const;

    // Emit a JSON string value with double-quote escaping.
    static std::string json_str(const std::string& s);
    static std::string json_str(const char* s);

    // Emit a boolean as "true" or "false".
    static const char* json_bool(bool b);
};

} // namespace agent