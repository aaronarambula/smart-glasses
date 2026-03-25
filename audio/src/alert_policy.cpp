// ─── alert_policy.cpp ────────────────────────────────────────────────────────
// Full implementation of the rate-limited alert policy.
// See include/audio/alert_policy.h for the full design notes.

#include "audio/alert_policy.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace audio {

// ─── Construction ─────────────────────────────────────────────────────────────

AlertPolicy::AlertPolicy(TtsEngine& tts,
                         HapticsEngine& haptics,
                         AlertThresholds thresholds)
    : tts_(tts)
    , haptics_(haptics)
    , thresholds_(std::move(thresholds))
{
    // Initialise all last-alert timestamps to epoch so every level fires
    // immediately on the first call (no artificial warmup delay).
    const auto epoch = std::chrono::steady_clock::time_point{};
    last_alert_time_.fill(epoch);
}

// ─── reset ────────────────────────────────────────────────────────────────────

void AlertPolicy::reset()
{
    const auto epoch = std::chrono::steady_clock::time_point{};
    last_alert_time_.fill(epoch);
    last_risk_level_   = prediction::RiskLevel::CLEAR;
    alerts_spoken_     = 0;
    alerts_suppressed_ = 0;
    last_event_        = AlertEvent{};
}

void AlertPolicy::reset_cooldowns()
{
    const auto epoch = std::chrono::steady_clock::time_point{};
    last_alert_time_.fill(epoch);
}

// ─── cooldown_for ─────────────────────────────────────────────────────────────

std::chrono::duration<float> AlertPolicy::cooldown_for(
    prediction::RiskLevel level) const
{
    switch (level) {
        case prediction::RiskLevel::DANGER:
            return std::chrono::duration<float>(thresholds_.cooldown_danger_s);
        case prediction::RiskLevel::WARNING:
            return std::chrono::duration<float>(thresholds_.cooldown_warning_s);
        case prediction::RiskLevel::CAUTION:
            return std::chrono::duration<float>(thresholds_.cooldown_caution_s);
        default:
            return std::chrono::duration<float>(thresholds_.cooldown_agent_s);
    }
}

// ─── cooldown_elapsed ────────────────────────────────────────────────────────
//
// Returns true if the cooldown for `level` has elapsed.
//
// The cooldown is measured against the last alert at `level` OR HIGHER —
// this prevents WARNING from firing 3 seconds after a DANGER alert.
// If DANGER fired 1 second ago, WARNING should also be suppressed.

bool AlertPolicy::cooldown_elapsed(prediction::RiskLevel level) const
{
    const auto now = std::chrono::steady_clock::now();
    const auto cooldown = cooldown_for(level);

    // Check this level and all higher levels.
    for (int l = static_cast<int>(level);
         l < static_cast<int>(NUM_LEVELS);
         ++l)
    {
        const auto& t = last_alert_time_[static_cast<size_t>(l)];
        // Epoch means never fired — treat as elapsed.
        if (t == std::chrono::steady_clock::time_point{}) continue;

        auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
            now - t);

        if (elapsed < cooldown) {
            return false;
        }
    }

    return true;
}

// ─── ttc_urgency_override ─────────────────────────────────────────────────────
//
// Returns true if TTC is below the urgency threshold, meaning we bypass
// the normal cooldown and speak immediately regardless.

bool AlertPolicy::ttc_urgency_override(
    const prediction::FullPrediction& pred) const
{
    const float min_ttc = pred.min_ttc_s();
    return std::isfinite(min_ttc) &&
           min_ttc < thresholds_.ttc_urgency_override_s;
}

// ─── format_distance ─────────────────────────────────────────────────────────
//
// Converts a distance in mm to a spoken English string.
// Examples:
//   300 mm  → "zero point three metres"
//   800 mm  → "zero point eight metres"
//   1200 mm → "one point two metres"
//   2500 mm → "two point five metres"
//   5000 mm → "five metres"
//
// We use a lookup table for the most common distances encountered in
// obstacle avoidance (0.3 m – 3.0 m in 0.1 m steps) for natural speech,
// and fall back to a formatted float string for others.

std::string AlertPolicy::format_distance(float distance_mm)
{
    // Round to nearest 100 mm for the lookup.
    const int dm = static_cast<int>(std::round(distance_mm / 100.0f));

    // Tenths-of-metre word table [0..30] (indices 0–30 = 0.0m–3.0m).
    static const char* const TENTHS[] = {
        "zero",           // 0
        "zero point one", // 1 → 0.1
        "zero point two", // 2 → 0.2
        "zero point three",
        "zero point four",
        "zero point five",
        "zero point six",
        "zero point seven",
        "zero point eight",
        "zero point nine",
        "one",            // 10 → 1.0
        "one point one",
        "one point two",
        "one point three",
        "one point four",
        "one point five",
        "one point six",
        "one point seven",
        "one point eight",
        "one point nine",
        "two",            // 20 → 2.0
        "two point one",
        "two point two",
        "two point three",
        "two point four",
        "two point five",
        "two point six",
        "two point seven",
        "two point eight",
        "two point nine",
        "three",          // 30 → 3.0
    };

    if (dm >= 0 && dm <= 30) {
        return std::string(TENTHS[static_cast<size_t>(dm)]) + " metres";
    }

    // Beyond the table: round to nearest 0.5 m and use numeric string.
    float metres = std::round(distance_mm / 500.0f) * 0.5f;
    char buf[32];
    if (metres == std::floor(metres)) {
        std::snprintf(buf, sizeof(buf), "%.0f metres", metres);
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f metres", metres);
    }
    return std::string(buf);
}

// ─── format_ttc ───────────────────────────────────────────────────────────────
//
// Converts a TTC in seconds to a spoken English string.
// Examples:
//   0.8 s → "one second"   (always round up for safety — never undercount)
//   1.2 s → "one second"
//   1.8 s → "two seconds"
//   3.4 s → "three seconds"
//   9.0 s → "nine seconds"

std::string AlertPolicy::format_ttc(float ttc_s)
{
    // Always round UP to the nearest whole second (never undercount urgency).
    const int secs = static_cast<int>(std::ceil(std::max(ttc_s, 0.5f)));

    static const char* const WORDS[] = {
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "ten"
    };

    if (secs >= 1 && secs <= 10) {
        std::string s(WORDS[static_cast<size_t>(secs)]);
        s += (secs == 1) ? " second" : " seconds";
        return s;
    }

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d seconds", secs);
    return std::string(buf);
}

// ─── build_danger_text ────────────────────────────────────────────────────────
//
// Builds the TTS utterance for a DANGER-level alert.
//
// Templates:
//   With TTC + moving:   "Danger — [size] obstacle [dist] [dir] — collision in [T]"
//   With TTC < 1.5s:     "Danger — collision imminent [dir]"
//   No TTC:              "Danger — [size] obstacle [dist] [dir]"

std::string AlertPolicy::build_danger_text(const prediction::FullPrediction& pred)
{
    const prediction::TTCResult* fwd = pred.forward_threat();

    // Fall back to the most urgent result overall if nothing is directly forward.
    const prediction::TTCResult* threat = fwd ? fwd : pred.ttc.most_urgent();

    if (!threat) {
        return "Danger — obstacle detected";
    }

    const float dist  = threat->distance_mm;
    const float ttc   = threat->ttc_s;
    const char* dir   = "ahead";

    // Use the bearing to determine direction if it's not squarely forward.
    float b = threat->bearing_deg;
    if      (b > 22.5f  && b <= 67.5f)  dir = "ahead and to your right";
    else if (b > 67.5f  && b <= 112.5f) dir = "to your right";
    else if (b > 112.5f && b <= 157.5f) dir = "behind and to your right";
    else if (b > 157.5f && b <= 202.5f) dir = "behind you";
    else if (b > 202.5f && b <= 247.5f) dir = "behind and to your left";
    else if (b > 247.5f && b <= 292.5f) dir = "to your left";
    else if (b > 292.5f && b <= 337.5f) dir = "ahead and to your left";

    std::string text = "Danger — ";

    if (threat->has_ttc() && ttc < 1.5f) {
        // Imminent: keep it short and urgent.
        text += "collision imminent ";
        text += dir;
        return text;
    }

    // Add size label if informative.
    const char* sz = threat->size_label;
    bool show_size = (sz != nullptr &&
                      std::string(sz) != "unknown" &&
                      std::string(sz) != "wall");
    if (show_size) {
        text += sz;
        text += " ";
    }
    text += "obstacle ";
    text += format_distance(dist);
    text += " ";
    text += dir;

    if (threat->has_ttc()) {
        text += " — collision in ";
        text += format_ttc(ttc);
    }

    return text;
}

// ─── build_warning_text ───────────────────────────────────────────────────────
//
// Templates:
//   Moving toward user:  "Warning — [size] obstacle [dist] [dir] and closing"
//   Stationary:         "Warning — [size] obstacle [dist] [dir]"
//   With TTC:           "Warning — [size] obstacle [dist] [dir] — [T] to impact"

std::string AlertPolicy::build_warning_text(const prediction::FullPrediction& pred)
{
    const prediction::TTCResult* fwd  = pred.forward_threat();
    const prediction::TTCResult* threat = fwd ? fwd : pred.ttc.most_urgent();

    if (!threat) {
        return "Warning — obstacle nearby";
    }

    const float dist  = threat->distance_mm;
    const char* dir   = "ahead";

    float b = threat->bearing_deg;
    if      (b > 22.5f  && b <= 67.5f)  dir = "ahead-right";
    else if (b > 67.5f  && b <= 112.5f) dir = "right";
    else if (b > 112.5f && b <= 157.5f) dir = "behind-right";
    else if (b > 157.5f && b <= 202.5f) dir = "behind";
    else if (b > 202.5f && b <= 247.5f) dir = "behind-left";
    else if (b > 247.5f && b <= 292.5f) dir = "left";
    else if (b > 292.5f && b <= 337.5f) dir = "ahead-left";

    std::string text = "Warning — ";

    const char* sz = threat->size_label;
    bool show_size = (sz != nullptr && std::string(sz) != "unknown");
    if (show_size) {
        text += sz;
        text += " ";
    }
    text += "obstacle ";
    text += format_distance(dist);
    text += " ";
    text += dir;

    // Annotate with TTC or closing speed.
    if (threat->has_ttc() && threat->ttc_s < 6.0f) {
        text += " — ";
        text += format_ttc(threat->ttc_s);
        text += " to impact";
    } else if (!threat->is_stationary &&
               threat->closing_speed_mm_s > 150.0f) {
        text += " and closing";
    }

    return text;
}

// ─── build_caution_text ───────────────────────────────────────────────────────
//
// Template: "Caution — [size] obstacle [dist] [dir]"

std::string AlertPolicy::build_caution_text(const prediction::FullPrediction& pred)
{
    const prediction::TTCResult* fwd    = pred.forward_threat();
    const prediction::TTCResult* threat = fwd ? fwd : pred.ttc.most_urgent();

    if (!threat) {
        return "Caution — obstacle nearby";
    }

    const float dist = threat->distance_mm;
    const char* dir  = "ahead";

    float b = threat->bearing_deg;
    if      (b > 22.5f  && b <= 67.5f)  dir = "ahead-right";
    else if (b > 67.5f  && b <= 112.5f) dir = "right";
    else if (b > 112.5f && b <= 157.5f) dir = "behind-right";
    else if (b > 157.5f && b <= 202.5f) dir = "behind";
    else if (b > 202.5f && b <= 247.5f) dir = "behind-left";
    else if (b > 247.5f && b <= 292.5f) dir = "left";
    else if (b > 292.5f && b <= 337.5f) dir = "ahead-left";

    std::string text = "Caution — ";

    const char* sz = threat->size_label;
    bool show_size = (sz != nullptr && std::string(sz) != "unknown");
    if (show_size) {
        text += sz;
        text += " ";
    }
    text += "obstacle ";
    text += format_distance(dist);
    text += " ";
    text += dir;

    return text;
}

// ─── fire_alert ───────────────────────────────────────────────────────────────
//
// The single point of output — enqueues text in the TTS engine, records the
// event, and updates the last-alert timestamps.

void AlertPolicy::fire_alert(prediction::RiskLevel             level,
                              SpeechPriority                    priority,
                              const std::string&                text,
                              const prediction::FullPrediction& pred,
                              bool                              was_escalation,
                              bool                              was_ttc_override)
{
    if (level == prediction::RiskLevel::CAUTION) {
        haptics_.pulse_caution();
    } else {
        tts_.speak(text, priority);
    }

    const auto now = std::chrono::steady_clock::now();

    // Record the timestamp for this level.
    last_alert_time_[static_cast<size_t>(static_cast<int>(level))] = now;

    // Build the event record.
    AlertEvent ev;
    ev.time             = now;
    ev.risk_level       = level;
    ev.tts_priority     = priority;
    ev.text             = text;
    ev.was_escalation   = was_escalation;
    ev.was_ttc_override = was_ttc_override;

    // Populate distance and TTC from the most urgent forward threat.
    const prediction::TTCResult* fwd = pred.forward_threat();
    if (fwd) {
        ev.distance_mm = fwd->distance_mm;
        ev.ttc_s       = std::isfinite(fwd->ttc_s) ? fwd->ttc_s : 0.0f;
    }

    last_event_ = ev;
    ++alerts_spoken_;
}

// ─── process ─────────────────────────────────────────────────────────────────
//
// Main entry point — called once per frame (10 Hz).
//
// Decision tree:
//
//   1. Extract risk level from prediction.
//   2. If CLEAR → no alert, update state, return.
//   3. Determine if escalation occurred (level > last_risk_level_).
//   4. Determine if TTC override fires (TTC < urgency_threshold).
//   5. Check per-level cooldown.
//   6. Apply distance gate (suppress if farthest closest object > gate).
//   7. If any of (escalation, ttc_override, cooldown_elapsed) → fire alert.
//   8. Update last_risk_level_.

bool AlertPolicy::process(const prediction::FullPrediction& pred)
{
    const prediction::RiskLevel level = pred.risk_level();

    // CLEAR → no alert, but update state so we track the transition.
    if (level == prediction::RiskLevel::CLEAR) {
        last_risk_level_ = level;
        return false;
    }

    // ── Escalation check ─────────────────────────────────────────────────────
    const bool escalation = (level > last_risk_level_);

    // ── TTC urgency override ──────────────────────────────────────────────────
    const bool ttc_override = ttc_urgency_override(pred);

    // ── Distance gate ─────────────────────────────────────────────────────────
    // Don't speak CAUTION if everything is still far away.
    if (level == prediction::RiskLevel::CAUTION) {
        const prediction::TTCResult* threat = pred.ttc.most_urgent();
        if (threat && threat->distance_mm > thresholds_.caution_gate_mm) {
            ++alerts_suppressed_;
            last_risk_level_ = level;
            return false;
        }
    }

    // ── Should we speak? ──────────────────────────────────────────────────────
    const bool should_speak =
        escalation ||
        ttc_override ||
        cooldown_elapsed(level);

    if (!should_speak) {
        ++alerts_suppressed_;
        last_risk_level_ = level;
        return false;
    }

    // ── Build utterance and fire ───────────────────────────────────────────────
    std::string text;
    SpeechPriority priority;

    switch (level) {
        case prediction::RiskLevel::DANGER:
            text     = build_danger_text(pred);
            priority = SpeechPriority::DANGER;
            break;
        case prediction::RiskLevel::WARNING:
            text     = build_warning_text(pred);
            priority = SpeechPriority::WARNING;
            break;
        case prediction::RiskLevel::CAUTION:
            text     = build_caution_text(pred);
            priority = SpeechPriority::CAUTION;
            break;
        default:
            last_risk_level_ = level;
            return false;
    }

    fire_alert(level, priority, text, pred, escalation, ttc_override);
    last_risk_level_ = level;
    return true;
}

// ─── deliver_agent_advice ────────────────────────────────────────────────────
//
// Delivers an OpenAI agent advice string at the lowest TTS priority.
// Trims to MAX_AGENT_CHARS to keep utterances brief.
// Only enqueues if the TTS engine is not currently speaking something more
// important (the TTS priority queue handles this automatically).

void AlertPolicy::deliver_agent_advice(const std::string& advice)
{
    if (advice.empty()) return;

    // Trim to maximum length.
    std::string trimmed = advice;
    if (trimmed.size() > MAX_AGENT_CHARS) {
        trimmed = trimmed.substr(0, MAX_AGENT_CHARS);
        // Trim to the last complete word.
        size_t last_space = trimmed.rfind(' ');
        if (last_space != std::string::npos) {
            trimmed = trimmed.substr(0, last_space);
        }
    }

    // Check the AGENT-level cooldown.
    const auto now = std::chrono::steady_clock::now();
    const auto& last = last_alert_time_[4];   // index 4 = AGENT
    if (last != std::chrono::steady_clock::time_point{}) {
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
            now - last);
        if (elapsed.count() < thresholds_.cooldown_agent_s) {
            ++alerts_suppressed_;
            return;
        }
    }

    tts_.speak_agent(trimmed);
    last_alert_time_[4] = now;

    AlertEvent ev;
    ev.time         = now;
    ev.risk_level   = prediction::RiskLevel::CLEAR;   // agent advice is not a risk alert
    ev.tts_priority = SpeechPriority::AGENT;
    ev.text         = trimmed;
    last_event_ = ev;

    ++alerts_spoken_;
}

} // namespace audio
