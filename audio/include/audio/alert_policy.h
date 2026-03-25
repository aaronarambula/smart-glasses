#pragma once

// ─── alert_policy.h ──────────────────────────────────────────────────────────
// Rate-limited alert policy: decides what to say, when to say it, and at what
// TTS priority — consuming a FullPrediction from the prediction pipeline and
// driving a TtsEngine.
//
// Design
// ──────
// The naive approach — speaking on every frame — would produce 10 utterances
// per second, which is unusable. The alert policy solves this with three
// mechanisms:
//
//   1. Per-level cooldown timers
//      Each RiskLevel has a minimum interval between consecutive alerts.
//      DANGER: 1.5 s   WARNING: 3 s   CAUTION: 6 s   AGENT: 8 s
//      An alert is only spoken if that cooldown has elapsed since the last
//      alert at that level OR HIGHER.
//
//   2. Escalation bypass
//      If the risk level INCREASES (e.g. CAUTION → WARNING), the cooldown is
//      bypassed regardless of how recently the last alert was spoken. The user
//      must always hear the first sign of a worsening situation immediately.
//
//   3. TTC urgency gate
//      Even within cooldown, if TTC drops below an urgency threshold the alert
//      fires anyway. A collision predicted in 1.2 seconds cannot wait for a
//      3-second cooldown to expire.
//
// Utterance templates
// ───────────────────
// The policy builds natural-language sentences from structured prediction data.
// Examples:
//
//   DANGER  (TTC known):   "Obstacle point eight metres ahead — collision in
//                            two seconds"
//   DANGER  (TTC unknown): "Danger — obstacle point five metres to your right"
//   WARNING (moving):      "Warning — medium obstacle one point two metres
//                            ahead and closing"
//   WARNING (stationary):  "Warning — wall one point five metres to your left"
//   CAUTION:               "Caution — small obstacle two metres ahead-right"
//   AGENT:                 verbatim GPT response (passed in directly)
//
// All distances are spoken in metres to one decimal place.
// TTC is rounded to the nearest second (minimum "one second").
//
// State machine
// ─────────────
// The policy tracks the previous RiskLevel so it can detect transitions.
// On transition DOWN (e.g. DANGER → CLEAR), no alert is spoken — silence
// is itself informative (the threat has passed).
// On transition UP, the alert fires immediately (escalation bypass above).
//
// Thread safety
// ─────────────
// AlertPolicy is NOT thread-safe. It must be called from a single thread
// (the pipeline callback thread). TtsEngine IS thread-safe and can be called
// from any thread.

#include "tts_engine.h"
#include "haptics_engine.h"
#include "prediction/prediction.h"

#include <string>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>

namespace audio {

// ─── AlertThresholds ─────────────────────────────────────────────────────────
//
// All timing and distance thresholds used by the policy.
// Grouped into one struct so they can be tuned at runtime without recompiling.

struct AlertThresholds {
    // ── Cooldown periods (seconds between alerts at each level) ───────────────
    float cooldown_danger_s  = 1.5f;
    float cooldown_warning_s = 3.0f;
    float cooldown_caution_s = 6.0f;
    float cooldown_agent_s   = 8.0f;

    // ── TTC urgency override ──────────────────────────────────────────────────
    // If the forward TTC drops below this, bypass the cooldown and speak now.
    float ttc_urgency_override_s = 2.5f;

    // ── Minimum distance to generate any alert (mm) ───────────────────────────
    // Prevents constant chatter about distant objects.
    float min_alert_distance_mm = 4000.0f;

    // ── CAUTION distance gate (mm) ────────────────────────────────────────────
    // CAUTION alerts only fire if the closest object is within this distance.
    float caution_gate_mm = 2500.0f;

    // ── Closing speed minimum for "and closing" annotation (mm/s) ────────────
    float closing_annotation_speed_mm_s = 200.0f;
};

// ─── AlertEvent ──────────────────────────────────────────────────────────────
//
// Record of one alert that was spoken. Useful for logging and testing.

struct AlertEvent {
    std::chrono::steady_clock::time_point time;
    prediction::RiskLevel                 risk_level;
    SpeechPriority                        tts_priority;
    std::string                           text;
    float                                 distance_mm  = 0.0f;
    float                                 ttc_s        = 0.0f;
    bool                                  was_escalation = false;
    bool                                  was_ttc_override = false;
};

// ─── AlertPolicy ─────────────────────────────────────────────────────────────

class AlertPolicy {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // tts      : reference to the TtsEngine that will speak the alerts.
    //            The TtsEngine must outlive the AlertPolicy.
    // thresholds: tunable timing / distance parameters (default = safe values).
    explicit AlertPolicy(TtsEngine& tts,
                         HapticsEngine& haptics,
                         AlertThresholds thresholds = AlertThresholds{});

    // Non-copyable (holds a reference to TtsEngine).
    AlertPolicy(const AlertPolicy&)            = delete;
    AlertPolicy& operator=(const AlertPolicy&) = delete;

    // ── Main entry point ──────────────────────────────────────────────────────

    // Process one FullPrediction from the pipeline.
    // Decides whether to speak, builds the utterance, and calls tts_.speak().
    //
    // Call this once per frame (10 Hz) from the pipeline callback.
    // Returns true if an alert was spoken this call, false if suppressed.
    bool process(const prediction::FullPrediction& pred);

    // Deliver an agent advice string directly (bypasses the risk-level logic).
    // Spoken at SpeechPriority::AGENT — only if the queue is otherwise quiet.
    // The string is trimmed to MAX_AGENT_CHARS characters to keep it brief.
    void deliver_agent_advice(const std::string& advice);

    // ── State ─────────────────────────────────────────────────────────────────

    // Reset all cooldown timers and state (e.g. on pipeline restart).
    void reset();

    // Force the last-spoken level to CLEAR so the next alert at any level
    // fires immediately regardless of cooldown. Useful after a long silence.
    void reset_cooldowns();

    // ── Accessors ─────────────────────────────────────────────────────────────

    prediction::RiskLevel last_risk_level()  const { return last_risk_level_;  }
    uint64_t              alerts_spoken()    const { return alerts_spoken_;     }
    uint64_t              alerts_suppressed() const { return alerts_suppressed_; }

    AlertThresholds&       thresholds()       { return thresholds_; }
    const AlertThresholds& thresholds() const { return thresholds_; }

    // Most recent alert event (empty text if none yet).
    const AlertEvent& last_event() const { return last_event_; }

    // ── Utterance building (public for unit testing) ───────────────────────────

    // Build the TTS utterance string for a DANGER alert.
    static std::string build_danger_text(const prediction::FullPrediction& pred);

    // Build the TTS utterance string for a WARNING alert.
    static std::string build_warning_text(const prediction::FullPrediction& pred);

    // Build the TTS utterance string for a CAUTION alert.
    static std::string build_caution_text(const prediction::FullPrediction& pred);

private:
    // ── Internal helpers ──────────────────────────────────────────────────────

    // Returns true if the cooldown for `level` has elapsed since the last
    // alert at that level or higher.
    bool cooldown_elapsed(prediction::RiskLevel level) const;

    // Returns the cooldown duration for a given level.
    std::chrono::duration<float> cooldown_for(prediction::RiskLevel level) const;

    // Returns true if TTC is below the urgency override threshold.
    bool ttc_urgency_override(const prediction::FullPrediction& pred) const;

    // Fires an alert: builds text, enqueues in TTS, records the event.
    void fire_alert(prediction::RiskLevel level,
                    SpeechPriority        priority,
                    const std::string&    text,
                    const prediction::FullPrediction& pred,
                    bool was_escalation,
                    bool was_ttc_override);

    // Formats a distance in mm as a spoken string, e.g. 1200 → "one point two metres".
    // Falls back to numeric string for distances outside the lookup table range.
    static std::string format_distance(float distance_mm);

    // Formats a TTC in seconds as a spoken string, e.g. 2.3 → "two seconds".
    static std::string format_ttc(float ttc_s);

    // ── References ────────────────────────────────────────────────────────────

    TtsEngine&      tts_;
    HapticsEngine&  haptics_;
    AlertThresholds thresholds_;

    // ── Per-level last-alert timestamps ───────────────────────────────────────
    // Index = static_cast<int>(RiskLevel): [0]=CLEAR, [1]=CAUTION, [2]=WARNING, [3]=DANGER
    // +1 for AGENT at index 4.
    static constexpr size_t NUM_LEVELS = 5;
    std::array<std::chrono::steady_clock::time_point, NUM_LEVELS> last_alert_time_;

    // ── State tracking ────────────────────────────────────────────────────────

    prediction::RiskLevel last_risk_level_ = prediction::RiskLevel::CLEAR;

    // ── Counters ──────────────────────────────────────────────────────────────

    uint64_t alerts_spoken_    = 0;
    uint64_t alerts_suppressed_ = 0;

    // ── Last event record ─────────────────────────────────────────────────────

    AlertEvent last_event_;

    // ── Constants ─────────────────────────────────────────────────────────────

    // Agent advice is clipped to this many characters to keep it brief.
    static constexpr size_t MAX_AGENT_CHARS = 200;
};

} // namespace audio
