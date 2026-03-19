// ─── agent_loop.cpp ───────────────────────────────────────────────────────────
// Background agent thread that periodically queries GPT-4o and delivers
// navigation advice to the AudioSystem.
// See include/agent/agent_loop.h for the full design notes.

#include "agent/agent_loop.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <chrono>
#include <limits>

namespace agent {

// ─── Construction ─────────────────────────────────────────────────────────────

AgentLoop::AgentLoop(OpenAIClient&       client,
                     SceneBuilder&       scene_builder,
                     audio::AudioSystem& audio,
                     AgentConfig         config)
    : client_(client)
    , scene_builder_(scene_builder)
    , audio_(audio)
    , config_(std::move(config))
{
    // Populate default blocked phrases — catch off-topic or broken responses.
    if (config_.blocked_phrases.empty()) {
        config_.blocked_phrases = {
            "as an ai",
            "i cannot",
            "i'm sorry",
            "error",
            "undefined",
            "null",
        };
    }
}

// ─── Destructor ───────────────────────────────────────────────────────────────

AgentLoop::~AgentLoop()
{
    stop();
}

// ─── start ────────────────────────────────────────────────────────────────────

void AgentLoop::start()
{
    if (running_.load()) return;

    stop_flag_.store(false);
    running_.store(true);

    thread_ = std::thread(&AgentLoop::loop_body, this);
}

// ─── stop ─────────────────────────────────────────────────────────────────────

void AgentLoop::stop()
{
    if (!running_.load()) return;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        stop_flag_.store(true);
    }
    cv_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }

    running_.store(false);
}

// ─── is_enabled ───────────────────────────────────────────────────────────────

bool AgentLoop::is_enabled() const
{
    return client_.has_api_key();
}

// ─── push_prediction ─────────────────────────────────────────────────────────
//
// Called at 10 Hz from the pipeline thread.
// Swaps the stored FullPrediction snapshot atomically (under a lightweight
// mutex — not on the hot Kalman/DBSCAN path, just the pointer assignment).
// Also wakes the agent thread if it is sleeping through the interval, so it
// can immediately notice a transition to DANGER.

void AgentLoop::push_prediction(
    const prediction::FullPrediction&   pred,
    const perception::PerceptionResult* perc_ptr)
{
    // Heap-allocate a copy of the prediction so the agent thread can hold it
    // after the pipeline thread has moved on to the next frame.
    auto pred_copy = std::make_shared<prediction::FullPrediction>(pred);

    std::shared_ptr<perception::PerceptionResult> perc_copy;
    if (perc_ptr) {
        perc_copy = std::make_shared<perception::PerceptionResult>(*perc_ptr);
    }

    {
        std::lock_guard<std::mutex> lk(snapshot_mutex_);
        pred_snapshot_ = std::move(pred_copy);
        perc_snapshot_ = std::move(perc_copy);
    }

    // Wake the agent thread early if DANGER just appeared — don't make the user
    // wait up to query_interval_s seconds for GPT advice when a collision is
    // imminent.
    if (pred.risk_level() == prediction::RiskLevel::DANGER) {
        cv_.notify_one();
    }
}

// ─── effective_interval_s ─────────────────────────────────────────────────────
//
// Returns the query interval for the given risk level.
// DANGER uses a shorter interval for more frequent guidance.

float AgentLoop::effective_interval_s(prediction::RiskLevel risk) const
{
    if (risk == prediction::RiskLevel::DANGER) {
        return config_.danger_query_interval_s;
    }
    return config_.query_interval_s;
}

// ─── should_query ─────────────────────────────────────────────────────────────
//
// All four gates must pass for a query to be fired.
//
// Gate 1 — Risk gate:
//   Risk must be >= min_risk_to_query. Clear scenes are never sent.
//
// Gate 2 — Cooldown gate:
//   At least effective_interval_s() must have elapsed since the last
//   successful query dispatch.
//
// Gate 3 — In-flight gate:
//   Only one request at a time. If the previous request hasn't completed,
//   skip — we don't want to stack up requests and get stale advice.
//
// Gate 4 — Change gate (optional):
//   If the scene hasn't meaningfully changed since the last query, skip.
//   "Changed" = risk level changed OR a new tracked object appeared OR
//   global min TTC changed by more than ttc_change_threshold_s.

bool AgentLoop::should_query(const prediction::FullPrediction& pred) const
{
    // ── Gate 1: risk level ────────────────────────────────────────────────────
    if (pred.risk_level() < config_.min_risk_to_query) {
        return false;
    }

    // ── Gate 2: cooldown ──────────────────────────────────────────────────────
    if (last_query_time_set_) {
        const auto now     = std::chrono::steady_clock::now();
        const float interval = effective_interval_s(pred.risk_level());
        const auto  elapsed  = std::chrono::duration_cast<std::chrono::duration<float>>(
            now - last_query_time_).count();

        if (elapsed < interval) {
            return false;
        }
    }

    // ── Gate 3: in-flight ─────────────────────────────────────────────────────
    if (in_flight_.load() > 0) {
        return false;
    }

    // ── Gate 4: change detection ──────────────────────────────────────────────
    if (config_.use_change_gate) {
        // Risk level changed → always query.
        if (pred.risk_level() != last_queried_risk_) {
            return true;
        }

        // Object count changed → new obstacles appeared or disappeared.
        const size_t obj_count = pred.ttc.results.size();
        if (obj_count != last_queried_object_count_) {
            return true;
        }

        // Minimum TTC changed significantly.
        const float min_ttc = pred.min_ttc_s();
        const float prev    = last_queried_min_ttc_;

        bool ttc_changed =
            (std::isfinite(min_ttc) != std::isfinite(prev)) ||
            (std::isfinite(min_ttc) && std::isfinite(prev) &&
             std::abs(min_ttc - prev) > config_.ttc_change_threshold_s);

        if (!ttc_changed) {
            return false;
        }
    }

    return true;
}

// ─── record_query ─────────────────────────────────────────────────────────────
//
// Updates change-detection state and timing after dispatching a query.

void AgentLoop::record_query(const prediction::FullPrediction& pred)
{
    last_query_time_            = std::chrono::steady_clock::now();
    last_query_time_set_        = true;
    last_queried_risk_          = pred.risk_level();
    last_queried_min_ttc_       = pred.min_ttc_s();
    last_queried_object_count_  = pred.ttc.results.size();
}

// ─── filter_response ─────────────────────────────────────────────────────────
//
// Applies two filters to the model's response text:
//
//   1. Length trim: truncate to max_response_chars at a word boundary.
//   2. Blocked phrase check: silently drop responses containing any blocked
//      phrase (case-insensitive).
//
// Returns the filtered string, or empty string if the response should be dropped.

std::string AgentLoop::filter_response(const std::string& raw) const
{
    if (raw.empty()) return {};

    // ── Blocked phrase check ──────────────────────────────────────────────────
    std::string lower_raw = raw;
    std::transform(lower_raw.begin(), lower_raw.end(),
                   lower_raw.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });

    for (const auto& phrase : config_.blocked_phrases) {
        std::string lower_phrase = phrase;
        std::transform(lower_phrase.begin(), lower_phrase.end(),
                       lower_phrase.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });

        if (lower_raw.find(lower_phrase) != std::string::npos) {
            log("Response blocked (contains: '" + phrase + "')");
            return {};
        }
    }

    // ── Length trim ───────────────────────────────────────────────────────────
    std::string trimmed = raw;

    if (trimmed.size() > config_.max_response_chars) {
        trimmed = trimmed.substr(0, config_.max_response_chars);
        // Trim to last complete word (find last space).
        const size_t last_space = trimmed.rfind(' ');
        if (last_space != std::string::npos) {
            trimmed = trimmed.substr(0, last_space);
        }
        // Add ellipsis only if we actually truncated mid-sentence.
        if (!trimmed.empty() && trimmed.back() != '.') {
            trimmed += '.';
        }
    }

    // Strip trailing whitespace.
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) {
        trimmed.pop_back();
    }

    return trimmed;
}

// ─── log ──────────────────────────────────────────────────────────────────────

void AgentLoop::log(const std::string& msg) const
{
    if (config_.verbose) {
        std::cout << "[agent] " << msg << "\n" << std::flush;
    }
}

// ─── loop_body ────────────────────────────────────────────────────────────────
//
// Main body of the background thread.
//
// The thread sleeps for a short tick interval (200 ms) in a condition variable
// wait. On each wake it:
//   1. Reads the latest prediction snapshot (lock-free copy).
//   2. Runs all gates via should_query().
//   3. If gates pass: builds the scene JSON, fires an async OpenAI request.
//   4. The request callback delivers the response to AudioSystem.
//   5. Loops back to sleep.
//
// The short tick (200 ms) means the cooldown and change gates are checked
// frequently enough that latency between "gate opens" and "query fires" is
// at most 200 ms — imperceptible to the user.
// The DANGER wake-up path (cv_.notify_one() in push_prediction()) means
// a new DANGER frame can trigger a query within ~5 ms of arriving.

void AgentLoop::loop_body()
{
    // Tick interval: how long the thread sleeps between gate checks.
    // Short enough to be responsive, long enough not to spin.
    static constexpr auto TICK = std::chrono::milliseconds(200);

    while (true) {
        // ── Wait for tick or wake-up ──────────────────────────────────────────
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait_for(lk, TICK, [this] {
                return stop_flag_.load();
            });

            if (stop_flag_.load()) break;
        }

        // ── Read latest snapshot (under snapshot_mutex_) ──────────────────────
        std::shared_ptr<prediction::FullPrediction>   pred_snap;
        std::shared_ptr<perception::PerceptionResult> perc_snap;

        {
            std::lock_guard<std::mutex> lk(snapshot_mutex_);
            pred_snap = pred_snapshot_;
            perc_snap = perc_snapshot_;
        }

        // No prediction yet — nothing to do.
        if (!pred_snap) {
            ++stats_.queries_skipped;
            continue;
        }

        // ── Gate check ────────────────────────────────────────────────────────
        if (!should_query(*pred_snap)) {
            ++stats_.queries_skipped;
            continue;
        }

        // ── Build scene JSON ──────────────────────────────────────────────────
        const int   training_steps = training_steps_.load();
        const float training_loss  = static_cast<float>(
            training_loss_millis_.load()) / 1000.0f;

        std::string scene_json = scene_builder_.build(
            *pred_snap,
            perc_snap.get(),
            training_steps,
            training_loss
        );

        log("Querying GPT | risk=" +
            std::string(prediction::risk_level_name(pred_snap->risk_level())) +
            " | scene=" + scene_json);

        // ── Record state before dispatch ─────────────────────────────────────
        record_query(*pred_snap);
        ++stats_.queries_sent;
        ++in_flight_;

        const auto query_dispatched_at = std::chrono::steady_clock::now();

        // ── Capture everything the callback needs ────────────────────────────
        // The callback runs on a detached thread — we must capture by value.
        // We capture the risk level at query time so the callback can decide
        // whether the response is still relevant when it arrives.
        const prediction::RiskLevel risk_at_query = pred_snap->risk_level();

        // ── Fire async request ─────────────────────────────────────────────────
        bool dispatched = client_.request(
            scene_json,
            [this,
             risk_at_query,
             query_dispatched_at](bool success, std::string text)
            {
                // This callback runs on a detached OpenAIClient thread.
                --in_flight_;

                const auto now = std::chrono::steady_clock::now();
                const float latency_s = std::chrono::duration_cast<
                    std::chrono::duration<float>>(now - query_dispatched_at).count();

                stats_.last_response_latency_s = latency_s;

                if (!success) {
                    ++stats_.api_errors;
                    stats_.last_error = text;
                    log("GPT request failed: " + text);
                    return;
                }

                ++stats_.responses_received;

                log("GPT response (latency=" +
                    [&]() {
                        char buf[16];
                        std::snprintf(buf, sizeof(buf), "%.2f", latency_s);
                        return std::string(buf);
                    }() +
                    "s): " + text);

                // ── Staleness check ───────────────────────────────────────────
                // Read the current risk level from the latest snapshot.
                // If the scene has completely cleared since we queried, drop
                // the response — GPT advice about obstacles that no longer
                // exist would confuse the user.
                {
                    std::shared_ptr<prediction::FullPrediction> current_snap;
                    {
                        std::lock_guard<std::mutex> lk(snapshot_mutex_);
                        current_snap = pred_snapshot_;
                    }

                    if (current_snap) {
                        const auto current_risk = current_snap->risk_level();

                        // Drop if: was DANGER/WARNING, now CLEAR.
                        const bool was_serious =
                            risk_at_query >= prediction::RiskLevel::WARNING;
                        const bool now_clear =
                            current_risk == prediction::RiskLevel::CLEAR;

                        if (was_serious && now_clear) {
                            ++stats_.responses_dropped;
                            log("Response dropped — scene cleared while waiting");
                            return;
                        }
                    }
                }

                // ── Filter response ───────────────────────────────────────────
                const std::string filtered = filter_response(text);
                if (filtered.empty()) {
                    ++stats_.responses_dropped;
                    return;
                }

                // ── Deliver to AudioSystem ────────────────────────────────────
                // deliver_agent_advice() is thread-safe — it enqueues at
                // SpeechPriority::AGENT and is automatically pre-empted by
                // any WARNING or DANGER alert in the TTS queue.
                audio_.deliver_agent_advice(filtered);

                stats_.last_advice = filtered;
                log("Delivered advice: " + filtered);
            }
        );

        if (!dispatched) {
            // Client returned false (no API key) — should not happen here
            // because is_enabled() was checked, but handle defensively.
            --in_flight_;
            ++stats_.api_errors;
            log("Request not dispatched (no API key?)");
        }
    }

    running_.store(false);
}

} // namespace agent