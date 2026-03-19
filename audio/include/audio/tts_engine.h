#pragma once

// ─── tts_engine.h ────────────────────────────────────────────────────────────
// Non-blocking text-to-speech engine wrapping espeak-ng.
//
// Design
// ──────
// Speech output must never block the 10 Hz sensor pipeline. This engine runs
// espeak-ng on a dedicated background thread, driven by a priority queue so
// that a DANGER alert always interrupts a lower-priority CAUTION that is
// currently speaking.
//
// Architecture
// ────────────
//
//   Pipeline thread                 TtsEngine internals
//   ───────────────                 ───────────────────
//   speak(msg, DANGER)  ──push──▶  PriorityQueue
//   speak(msg, WARNING) ──push──▶  PriorityQueue
//                                       │
//                                  worker_thread (blocked on condvar)
//                                       │
//                                  pop highest-priority item
//                                       │
//                                  kill ongoing espeak-ng if lower priority
//                                       │
//                                  popen("espeak-ng -s 150 ...", "w")
//                                       │
//                                  wait for process to finish
//                                       │
//                                  pop next item ...
//
// Priority levels
// ───────────────
//   DANGER  (3) — highest. Kills any ongoing speech and speaks immediately.
//   WARNING (2) — preempts CAUTION and AGENT_ADVICE.
//   CAUTION (1) — queued, spoken when nothing higher is pending.
//   AGENT   (0) — lowest. Only spoken when queue is otherwise empty.
//
// Deduplication
// ─────────────
// Identical messages at the same priority within a short time window are
// silently dropped. This prevents the engine from re-queuing "obstacle ahead"
// 10 times per second when the user is walking toward a wall.
//
// espeak-ng invocation
// ────────────────────
//   espeak-ng -s <speed> -p <pitch> -v <voice> "<text>"
//
//   speed  : words per minute (default 150 — fast enough to be timely,
//             slow enough to be intelligible through a small speaker)
//   pitch  : 0–99 (default 55 — slightly lower than default, less alarming)
//   voice  : "en" (English, or configurable)
//
// The process is spawned with popen() so we can write the text to its stdin
// or pass it as a command-line argument. We use the argument form so we don't
// need to manage a pipe write — espeak-ng exits when it finishes speaking.
//
// Killing ongoing speech
// ──────────────────────
// We track the PID of the running espeak-ng process. On preemption, we
// SIGTERM it, wait for it to exit, then immediately start the higher-priority
// utterance. The gap between kill and new speech is < 20 ms on a Pi.
//
// Thread safety
// ─────────────
// All public methods are thread-safe. The priority queue and subprocess state
// are protected by a single mutex + condition variable.
//
// Raspberry Pi note
// ─────────────────
// espeak-ng writes to the default ALSA audio device. On a Pi Zero 2W / Pi 4
// with a USB speaker or 3.5mm jack, this works out of the box.
// Install:  sudo apt install espeak-ng

#include <string>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include <cstdint>

namespace audio {

// ─── SpeechPriority ──────────────────────────────────────────────────────────

enum class SpeechPriority : int {
    AGENT        = 0,   // OpenAI agent advice — lowest, spoken in quiet moments
    CAUTION      = 1,   // CAUTION-level alert
    WARNING      = 2,   // WARNING-level alert
    DANGER       = 3,   // DANGER-level alert — highest, preempts everything
};

inline const char* priority_name(SpeechPriority p) {
    switch (p) {
        case SpeechPriority::AGENT:   return "AGENT";
        case SpeechPriority::CAUTION: return "CAUTION";
        case SpeechPriority::WARNING: return "WARNING";
        case SpeechPriority::DANGER:  return "DANGER";
        default:                      return "UNKNOWN";
    }
}

// ─── SpeechRequest ───────────────────────────────────────────────────────────
//
// One item in the TTS priority queue.

struct SpeechRequest {
    std::string    text;
    SpeechPriority priority  = SpeechPriority::CAUTION;
    uint64_t       sequence  = 0;    // tie-break: lower = older = lower priority
    std::chrono::steady_clock::time_point enqueued_at;

    // For the priority queue comparator: higher priority value = higher urgency.
    // Ties broken by sequence number (newer items come first within same priority).
    bool operator<(const SpeechRequest& o) const {
        if (priority != o.priority)
            return static_cast<int>(priority) < static_cast<int>(o.priority);
        return sequence < o.sequence;   // newer = higher sequence = higher priority
    }
};

// ─── TtsConfig ───────────────────────────────────────────────────────────────

struct TtsConfig {
    // espeak-ng parameters
    int         speed_wpm    = 150;    // words per minute
    int         pitch        = 55;     // 0–99
    std::string voice        = "en";   // language/voice code
    std::string executable   = "espeak-ng";  // full path or name on PATH

    // Queue limits
    size_t      max_queue_depth = 8;   // drop oldest low-priority items if full

    // Deduplication window: identical text at same priority within this
    // duration is silently dropped (prevents flooding).
    std::chrono::milliseconds dedup_window{2000};

    // If true, print each utterance to stdout before speaking.
    bool        verbose = false;
};

// ─── TtsEngine ───────────────────────────────────────────────────────────────

class TtsEngine {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    explicit TtsEngine(TtsConfig config = TtsConfig{});

    // Destructor: drains the queue, stops the worker thread, kills any
    // running espeak-ng process.
    ~TtsEngine();

    // Non-copyable, non-movable (owns a thread + mutex).
    TtsEngine(const TtsEngine&)            = delete;
    TtsEngine& operator=(const TtsEngine&) = delete;
    TtsEngine(TtsEngine&&)                 = delete;
    TtsEngine& operator=(TtsEngine&&)      = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Starts the background worker thread.
    // Must be called before speak(). Safe to call only once.
    void start();

    // Signals the worker thread to stop after finishing the current utterance.
    // Blocks until the thread exits. Safe to call from any thread.
    void stop();

    // ── Speech ────────────────────────────────────────────────────────────────

    // Enqueue a text utterance at the given priority.
    //
    // If priority == DANGER and there is currently a lower-priority utterance
    // being spoken, it is killed immediately and this utterance starts within
    // ~20 ms.
    //
    // If the queue is full and the new item has lower priority than all items
    // already in the queue, it is silently dropped.
    //
    // Duplicate detection: if the same text at the same priority was enqueued
    // within the dedup_window, the call is a no-op.
    //
    // Thread-safe.
    void speak(const std::string& text, SpeechPriority priority);

    // Convenience overloads matching the RiskLevel enum from prediction module.
    // Maps DANGER→DANGER, WARNING→WARNING, CAUTION→CAUTION, CLEAR→no-op.
    void speak_danger (const std::string& text) { speak(text, SpeechPriority::DANGER);  }
    void speak_warning(const std::string& text) { speak(text, SpeechPriority::WARNING); }
    void speak_caution(const std::string& text) { speak(text, SpeechPriority::CAUTION); }
    void speak_agent  (const std::string& text) { speak(text, SpeechPriority::AGENT);   }

    // Cancel all queued utterances and kill the currently speaking process.
    // The engine remains running and can accept new speak() calls immediately.
    // Thread-safe.
    void interrupt();

    // ── Status ────────────────────────────────────────────────────────────────

    bool is_running()  const { return running_.load(); }
    bool is_speaking() const { return speaking_.load(); }

    // Number of items currently in the queue (approximate — not locked).
    size_t queue_depth() const;

    // Total utterances spoken since start().
    uint64_t utterances_spoken() const { return utterances_spoken_.load(); }

    // Total utterances dropped (dedup or queue full) since start().
    uint64_t utterances_dropped() const { return utterances_dropped_.load(); }

    // ── Configuration ─────────────────────────────────────────────────────────

    TtsConfig&       config()       { return config_; }
    const TtsConfig& config() const { return config_; }

private:
    // ── Internal types ────────────────────────────────────────────────────────

    // Priority queue type: max-heap on SpeechRequest::operator<.
    using PQueue = std::priority_queue<SpeechRequest>;

    // ── State ─────────────────────────────────────────────────────────────────

    TtsConfig   config_;

    std::thread             worker_;
    std::mutex              mutex_;
    std::condition_variable cv_;

    std::atomic<bool>     running_   { false };
    std::atomic<bool>     speaking_  { false };
    std::atomic<bool>     stop_flag_ { false };

    PQueue   queue_;
    uint64_t sequence_counter_ = 0;

    // ── Deduplication state ───────────────────────────────────────────────────
    // Tracks the last-enqueued (text, priority, time) for dedup checks.
    struct DedupEntry {
        std::string    text;
        SpeechPriority priority;
        std::chrono::steady_clock::time_point enqueued_at;
    };
    std::vector<DedupEntry> dedup_history_;

    // ── Subprocess state ──────────────────────────────────────────────────────

    // PID of the currently running espeak-ng process.
    // -1 if no process is running.
    pid_t current_pid_ = -1;

    // Priority of the currently running utterance.
    SpeechPriority current_priority_ = SpeechPriority::CAUTION;

    // ── Counters ──────────────────────────────────────────────────────────────

    std::atomic<uint64_t> utterances_spoken_  { 0 };
    std::atomic<uint64_t> utterances_dropped_ { 0 };

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Main worker thread body.
    void worker_loop();

    // Speak one utterance synchronously (blocks until espeak-ng exits).
    // Called from worker_loop(). Updates current_pid_ so interrupt() can kill it.
    void speak_sync(const SpeechRequest& req);

    // Build the espeak-ng command string for a given utterance.
    std::string build_command(const std::string& text) const;

    // Kill the currently running espeak-ng process (if any).
    // Must be called with mutex_ held.
    void kill_current();

    // Returns true if this (text, priority) pair was recently enqueued
    // (within dedup_window). Must be called with mutex_ held.
    bool is_duplicate(const std::string& text, SpeechPriority priority) const;

    // Records an enqueue event for dedup tracking.
    // Must be called with mutex_ held.
    void record_enqueue(const std::string& text, SpeechPriority priority);

    // Prunes stale entries from dedup_history_ (older than dedup_window).
    // Must be called with mutex_ held.
    void prune_dedup_history();

    // Escape a string for safe use as a shell argument (single-quote wrapping).
    static std::string shell_escape(const std::string& s);
};

} // namespace audio