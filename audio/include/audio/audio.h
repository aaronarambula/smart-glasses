#pragma once

// ─── audio.h ─────────────────────────────────────────────────────────────────
// Umbrella header for the audio module.
//
// Include this single file to pull in:
//   - TtsEngine    : non-blocking espeak-ng wrapper with priority queue
//   - AlertPolicy  : rate-limited alert policy consuming FullPrediction
//
// Dependency graph (no cycles):
//
//   prediction/prediction.h
//         │
//         ▼
//   tts_engine.h      (TtsEngine, SpeechPriority, SpeechRequest, TtsConfig)
//         │
//         ▼
//   alert_policy.h    (AlertPolicy, AlertThresholds, AlertEvent)
//         │
//         ▼
//   audio.h           ← this file
//
// Typical usage in the app module:
//
//   #include "audio/audio.h"
//   #include "prediction/prediction.h"
//
//   audio::TtsConfig  cfg;
//   cfg.speed_wpm = 150;
//   cfg.verbose   = true;
//
//   audio::TtsEngine   tts(cfg);
//   audio::AlertPolicy policy(tts);
//
//   tts.start();
//
//   // Called once per frame (10 Hz) from the pipeline callback:
//   void on_prediction(const prediction::FullPrediction& pred) {
//       policy.process(pred);
//   }
//
//   // Called from the agent thread when GPT returns advice:
//   void on_agent_advice(const std::string& text) {
//       policy.deliver_agent_advice(text);
//   }
//
//   tts.stop();

#include "tts_engine.h"
#include "alert_policy.h"

namespace audio {

// ─── AudioSystem ─────────────────────────────────────────────────────────────
//
// Convenience owner that constructs and wires TtsEngine + AlertPolicy together.
// This is the single object the app module instantiates for the entire audio
// subsystem.
//
// Lifecycle:
//   1. Construct AudioSystem (config optional).
//   2. Call start() — launches the TTS worker thread.
//   3. Call process() once per frame from the pipeline thread.
//   4. Call deliver_agent_advice() from the agent thread as needed.
//   5. Call stop() before destroying (or let destructor handle it).
//
// Thread safety:
//   process()             — must be called from one thread only (pipeline)
//   deliver_agent_advice()— thread-safe, may be called from any thread
//   start() / stop()      — call once from the owning thread

class AudioSystem {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    explicit AudioSystem(TtsConfig       tts_config  = TtsConfig{},
                         AlertThresholds thresholds  = AlertThresholds{})
        : tts_(std::move(tts_config))
        , policy_(tts_, std::move(thresholds))
    {}

    // Non-copyable, non-movable (owns TtsEngine with a live thread).
    AudioSystem(const AudioSystem&)            = delete;
    AudioSystem& operator=(const AudioSystem&) = delete;
    AudioSystem(AudioSystem&&)                 = delete;
    AudioSystem& operator=(AudioSystem&&)      = delete;

    // Destructor stops the TTS engine gracefully.
    ~AudioSystem() {
        stop();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Starts the TTS worker thread. Must be called before process().
    void start() {
        tts_.start();
    }

    // Stops the TTS worker thread. Blocks until the thread exits.
    void stop() {
        tts_.stop();
    }

    // ── Main pipeline interface ───────────────────────────────────────────────

    // Process one FullPrediction. Delegates to AlertPolicy::process().
    // Call once per frame (10 Hz) from the pipeline callback thread.
    // Returns true if an alert was spoken this call.
    bool process(const prediction::FullPrediction& pred) {
        return policy_.process(pred);
    }

    // Deliver an agent advice string at the lowest TTS priority.
    // Thread-safe — may be called from the agent background thread.
    void deliver_agent_advice(const std::string& advice) {
        policy_.deliver_agent_advice(advice);
    }

    // ── Direct TTS access (for testing / custom utterances) ───────────────────

    // Speak an arbitrary string at a given priority.
    // Thread-safe.
    void speak(const std::string& text, SpeechPriority priority) {
        tts_.speak(text, priority);
    }

    // Immediately interrupt all speech and clear the queue.
    // Thread-safe.
    void interrupt() {
        tts_.interrupt();
    }

    // ── Status ────────────────────────────────────────────────────────────────

    bool is_speaking()        const { return tts_.is_speaking(); }
    bool is_running()         const { return tts_.is_running();  }
    size_t queue_depth()      const { return tts_.queue_depth(); }

    uint64_t utterances_spoken()    const { return tts_.utterances_spoken();    }
    uint64_t utterances_dropped()   const { return tts_.utterances_dropped();   }
    uint64_t alerts_spoken()        const { return policy_.alerts_spoken();     }
    uint64_t alerts_suppressed()    const { return policy_.alerts_suppressed(); }

    prediction::RiskLevel last_risk_level() const {
        return policy_.last_risk_level();
    }

    const AlertEvent& last_event() const { return policy_.last_event(); }

    // ── Component access (for tuning / diagnostics) ───────────────────────────

    TtsEngine&            tts()            { return tts_;    }
    const TtsEngine&      tts()    const   { return tts_;    }
    AlertPolicy&          policy()         { return policy_; }
    const AlertPolicy&    policy() const   { return policy_; }

    // ── Threshold tuning shortcuts ────────────────────────────────────────────

    void set_cooldown_danger_s (float s) { policy_.thresholds().cooldown_danger_s  = s; }
    void set_cooldown_warning_s(float s) { policy_.thresholds().cooldown_warning_s = s; }
    void set_cooldown_caution_s(float s) { policy_.thresholds().cooldown_caution_s = s; }

private:
    TtsEngine   tts_;
    AlertPolicy policy_;
};

} // namespace audio