#pragma once

// ─── agent_loop.h ─────────────────────────────────────────────────────────────
// Background agent thread that periodically fires GPT-4o queries and delivers
// the responses to the AudioSystem as spoken advice.
//
// Design
// ──────
// The pipeline runs at 10 Hz — far too fast to query GPT on every frame.
// The AgentLoop runs on its own thread and wakes up every QUERY_INTERVAL_S
// seconds to decide whether to send a new query. The decision is gated by:
//
//   1. Risk gate       — only query when risk >= MIN_RISK_TO_QUERY (default CAUTION)
//                        No point asking GPT for advice when the scene is clear.
//   2. Cooldown gate   — only query if at least QUERY_INTERVAL_S seconds have
//                        elapsed since the last successful query.
//   3. In-flight gate  — only one request in-flight at a time. If the previous
//                        request is still pending, skip this tick.
//   4. Change gate     — if the scene has not changed meaningfully since the last
//                        query (same risk level, no new objects), skip.
//
// On every pipeline frame the main thread calls push_prediction() which stores
// the latest FullPrediction in an atomic snapshot. The agent thread reads this
// snapshot when it decides to fire a query — no synchronisation overhead on the
// hot path (10 Hz pipeline thread).
//
// Response delivery
// ─────────────────
// When the OpenAI response arrives (on the OpenAIClient's background thread),
// the AgentLoop calls audio_system.deliver_agent_advice(text). The AudioSystem
// enqueues it at SpeechPriority::AGENT — it will be spoken only when nothing
// higher-priority is queued. This means GPT advice is automatically pre-empted
// by any WARNING or DANGER alert without any additional logic.
//
// Lifecycle
// ─────────
//   AgentLoop loop(client, scene_builder, audio);
//   loop.start();
//   // pipeline thread calls loop.push_prediction(pred) at 10 Hz
//   loop.stop();   // blocks until background thread exits
//
// Thread safety
// ─────────────
//   push_prediction() — lock-free (atomic snapshot swap), safe at 10 Hz
//   start() / stop()  — call once from the owning thread
//   All other methods — call from the owning thread only
//
// Memory
// ──────
// The FullPrediction snapshot is heap-allocated and managed via shared_ptr to
// avoid copying 500+ bytes of TTCFrame data on every push. The agent thread
// takes a shared_ptr copy at query time — zero-copy, zero-lock on the hot path.

#include "openai_client.h"
#include "scene_builder.h"

#include "audio/audio.h"
#include "prediction/prediction.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <chrono>
#include <functional>
#include <string>
#include <cstdint>

namespace agent {

// ─── AgentConfig ──────────────────────────────────────────────────────────────

struct AgentConfig {
    // ── Query timing ──────────────────────────────────────────────────────────

    // Seconds between GPT queries (minimum — actual interval may be longer
    // if the previous request is still in-flight or the risk gate is closed).
    float query_interval_s = 5.0f;

    // Minimum risk level required to trigger a query.
    // CLEAR scenes are never sent — saves API cost and avoids noise.
    prediction::RiskLevel min_risk_to_query = prediction::RiskLevel::CAUTION;

    // If risk reaches DANGER, reduce the query interval to this value so
    // GPT provides more frequent guidance in high-risk situations.
    float danger_query_interval_s = 2.5f;

    // ── Change detection ──────────────────────────────────────────────────────

    // If the scene hasn't changed since the last query, skip.
    // "Changed" is defined as: risk level changed OR a new tracked object
    // appeared OR any tracked object's TTC changed by more than ttc_change_threshold_s.
    bool  use_change_gate          = true;
    float ttc_change_threshold_s   = 1.5f;   // TTC delta that counts as a "change"

    // ── Response filtering ────────────────────────────────────────────────────

    // Maximum character length of agent advice delivered to TTS.
    // Responses longer than this are trimmed at a word boundary.
    size_t max_response_chars = 180;

    // If the response contains any of these substrings (case-insensitive),
    // it is silently dropped (safety filter for off-topic responses).
    // The model should never produce these given the system prompt, but
    // we guard anyway.
    // (Populated with known bad patterns in AgentLoop constructor.)
    std::vector<std::string> blocked_phrases;

    // ── Debug ─────────────────────────────────────────────────────────────────
    bool verbose = false;   // log query/response events to stdout
};

// ─── AgentStats ───────────────────────────────────────────────────────────────
//
// Cumulative statistics since start(). All fields are read from the owning
// thread only (not internally locked).

struct AgentStats {
    uint64_t queries_sent       = 0;
    uint64_t queries_skipped    = 0;   // gated out
    uint64_t responses_received = 0;
    uint64_t responses_dropped  = 0;   // filtered or too late
    uint64_t api_errors         = 0;
    float    last_query_time_s  = 0.0f;   // seconds since epoch of last query
    float    last_response_latency_s = 0.0f;
    std::string last_advice;            // most recent advice string spoken
    std::string last_error;             // most recent error string (if any)
};

// ─── AgentLoop ────────────────────────────────────────────────────────────────

class AgentLoop {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // client        : OpenAIClient that performs the HTTPS requests.
    //                 Must outlive this AgentLoop.
    // scene_builder : SceneBuilder that serialises FullPrediction → JSON.
    //                 May be shared — build() is stateless / const.
    // audio         : AudioSystem that delivers advice to the TTS queue.
    //                 Must outlive this AgentLoop.
    // config        : tuneable parameters (default = safe values).
    AgentLoop(OpenAIClient&  client,
              SceneBuilder&  scene_builder,
              audio::AudioSystem& audio,
              AgentConfig    config = AgentConfig{});

    // Destructor: calls stop() if still running.
    ~AgentLoop();

    // Non-copyable, non-movable (owns a thread + mutex).
    AgentLoop(const AgentLoop&)            = delete;
    AgentLoop& operator=(const AgentLoop&) = delete;
    AgentLoop(AgentLoop&&)                 = delete;
    AgentLoop& operator=(AgentLoop&&)      = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Starts the background query thread.
    // Must be called before push_prediction(). Safe to call only once.
    void start();

    // Signals the background thread to stop and blocks until it exits.
    // In-flight requests are allowed to complete (up to request_timeout_s).
    // Safe to call even if start() was never called.
    void stop();

    // ── Hot-path interface (called at 10 Hz from pipeline thread) ─────────────

    // Atomically replaces the stored FullPrediction snapshot with the latest
    // one. Lock-free — uses atomic shared_ptr swap so the pipeline thread
    // is never blocked by the agent thread's query logic.
    //
    // Also forwards the prediction to the perception result snapshot if
    // perc_ptr is non-null, for use by the scene builder.
    void push_prediction(const prediction::FullPrediction& pred,
                         const perception::PerceptionResult* perc_ptr = nullptr);

    // ── Status ────────────────────────────────────────────────────────────────

    bool is_running() const { return running_.load(); }

    // Returns a snapshot of the cumulative statistics.
    // Call from the owning thread only.
    const AgentStats& stats() const { return stats_; }

    // Returns true if the API key is configured and requests can be sent.
    bool is_enabled() const;

    // ── Config access ─────────────────────────────────────────────────────────

    AgentConfig&       config()       { return config_; }
    const AgentConfig& config() const { return config_; }

    // ── RiskPredictor diagnostics forwarding ──────────────────────────────────

    // The scene builder includes training_steps + loss in the JSON context so
    // GPT is aware the model is learning. Set these from the RiskPredictor
    // diagnostics after each training step.
    void set_training_info(int steps, float loss) {
        training_steps_.store(steps);
        // Store as integer millis to avoid atomic<float> portability issues.
        training_loss_millis_.store(static_cast<int>(loss * 1000.0f));
    }

private:
    // ── References ────────────────────────────────────────────────────────────

    OpenAIClient&       client_;
    SceneBuilder&       scene_builder_;
    audio::AudioSystem& audio_;

    // ── Config + stats ────────────────────────────────────────────────────────

    AgentConfig config_;
    AgentStats  stats_;

    // ── Thread state ──────────────────────────────────────────────────────────

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::atomic<bool>       running_   { false };
    std::atomic<bool>       stop_flag_ { false };

    // ── Atomic prediction snapshot ────────────────────────────────────────────
    // Updated by push_prediction() at 10 Hz (pipeline thread).
    // Read by the agent thread at query time.
    // std::atomic<shared_ptr> requires C++20; we guard with a lightweight
    // spinlock-free swap using mutex for the pointer assignment only.

    mutable std::mutex                              snapshot_mutex_;
    std::shared_ptr<prediction::FullPrediction>     pred_snapshot_;
    std::shared_ptr<perception::PerceptionResult>   perc_snapshot_;

    // ── In-flight tracking ────────────────────────────────────────────────────

    std::atomic<int>  in_flight_    { 0 };

    // ── Timing ────────────────────────────────────────────────────────────────

    std::chrono::steady_clock::time_point last_query_time_;
    bool last_query_time_set_ = false;

    // ── Change detection state ────────────────────────────────────────────────

    prediction::RiskLevel last_queried_risk_ = prediction::RiskLevel::CLEAR;
    float                 last_queried_min_ttc_ = std::numeric_limits<float>::infinity();
    size_t                last_queried_object_count_ = 0;

    // ── Training info (set from RiskPredictor diagnostics) ────────────────────

    std::atomic<int> training_steps_       { 0 };
    std::atomic<int> training_loss_millis_ { 0 };   // loss * 1000

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Main body of the background thread.
    void loop_body();

    // Determines whether to fire a query on this tick.
    // Returns true if all gates pass.
    bool should_query(const prediction::FullPrediction& pred) const;

    // Returns the effective query interval for the current risk level.
    float effective_interval_s(prediction::RiskLevel risk) const;

    // Updates change-detection state after a successful query dispatch.
    void record_query(const prediction::FullPrediction& pred);

    // Filters a model response:
    //   - Trims to max_response_chars at a word boundary.
    //   - Checks for blocked_phrases.
    // Returns empty string if the response should be dropped.
    std::string filter_response(const std::string& raw) const;

    // Logs a line to stdout if config_.verbose is true.
    void log(const std::string& msg) const;
};

} // namespace agent