// ─── tts_engine.cpp ──────────────────────────────────────────────────────────
// Full implementation of the non-blocking TTS engine.
// See include/audio/tts_engine.h for the full design notes.
//
// Platform: POSIX (Linux / Raspberry Pi OS).
// Requires: espeak-ng installed on the system PATH.
//           sudo apt install espeak-ng

#include "audio/tts_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cerrno>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <thread>

// POSIX headers — Raspberry Pi / Linux only
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>

namespace audio {

namespace {
constexpr auto kMaxUtteranceDuration = std::chrono::seconds(8);
constexpr auto kWaitPollInterval = std::chrono::milliseconds(20);
}

// ─── Construction ─────────────────────────────────────────────────────────────

TtsEngine::TtsEngine(TtsConfig config)
    : config_(std::move(config))
{
    // Pre-allocate a small dedup history to avoid rehashing at runtime.
    dedup_history_.reserve(16);
}

// ─── Destructor ───────────────────────────────────────────────────────────────

TtsEngine::~TtsEngine()
{
    stop();
}

// ─── start ───────────────────────────────────────────────────────────────────

void TtsEngine::start()
{
    // Guard against double-start.
    if (running_.load()) return;

    stop_flag_.store(false);
    running_.store(true);

    worker_ = std::thread(&TtsEngine::worker_loop, this);
}

// ─── stop ────────────────────────────────────────────────────────────────────

void TtsEngine::stop()
{
    if (!running_.load()) return;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        stop_flag_.store(true);
        // Kill any currently running espeak-ng process so the worker
        // thread doesn't block waiting for it to finish.
        kill_current();
    }

    cv_.notify_all();

    if (worker_.joinable()) {
        worker_.join();
    }

    running_.store(false);
    speaking_.store(false);
}

// ─── speak ───────────────────────────────────────────────────────────────────

void TtsEngine::speak(const std::string& text, SpeechPriority priority)
{
    if (text.empty()) return;
    if (!running_.load()) return;

    std::unique_lock<std::mutex> lk(mutex_);

    // ── Deduplication check ───────────────────────────────────────────────────
    prune_dedup_history();
    if (is_duplicate(text, priority)) {
        ++utterances_dropped_;
        return;
    }

    // ── DANGER preemption ─────────────────────────────────────────────────────
    // If the new item is DANGER and something lower-priority is currently
    // speaking, kill it immediately so DANGER starts within ~20 ms.
    if (priority == SpeechPriority::DANGER &&
        speaking_.load() &&
        current_priority_ < SpeechPriority::DANGER)
    {
        kill_current();
    }

    // ── Queue overflow handling ───────────────────────────────────────────────
    // If the queue is at capacity, drop the lowest-priority item.
    // If the new item itself is the lowest priority and the queue is full,
    // drop the new item.
    if (queue_.size() >= config_.max_queue_depth) {
        // std::priority_queue doesn't expose the underlying container directly,
        // so we rebuild if we need to drop the lowest item.
        // For simplicity: if the new item has AGENT priority and the queue is
        // full, just drop it.
        if (priority == SpeechPriority::AGENT) {
            ++utterances_dropped_;
            return;
        }
        // Otherwise: drain the queue into a temporary vector, drop the lowest,
        // push back the rest plus the new item.
        std::vector<SpeechRequest> items;
        items.reserve(queue_.size());
        while (!queue_.empty()) {
            items.push_back(queue_.top());
            queue_.pop();
        }
        // Sort ascending by priority so items.front() is the least important.
        std::sort(items.begin(), items.end()); // operator< is min-heap order
        // Drop the first (lowest priority) item.
        items.erase(items.begin());
        // Re-push remaining items.
        for (auto& item : items) {
            queue_.push(std::move(item));
        }
    }

    // ── Enqueue ───────────────────────────────────────────────────────────────
    SpeechRequest req;
    req.text        = text;
    req.priority    = priority;
    req.sequence    = ++sequence_counter_;
    req.enqueued_at = std::chrono::steady_clock::now();

    queue_.push(req);
    record_enqueue(text, priority);

    lk.unlock();
    cv_.notify_one();
}

// ─── interrupt ────────────────────────────────────────────────────────────────

void TtsEngine::interrupt()
{
    std::lock_guard<std::mutex> lk(mutex_);

    // Clear the queue.
    while (!queue_.empty()) queue_.pop();

    // Kill the currently running process.
    kill_current();
}

// ─── queue_depth ─────────────────────────────────────────────────────────────

size_t TtsEngine::queue_depth() const
{
    std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(mutex_));
    return queue_.size();
}

// ─── worker_loop ─────────────────────────────────────────────────────────────
//
// Main body of the background TTS thread.
// Waits on the condition variable until an item is in the queue, then pops
// and speaks the highest-priority item, then loops.

void TtsEngine::worker_loop()
{
    while (true) {
        SpeechRequest req;

        // ── Wait for work ─────────────────────────────────────────────────────
        {
            std::unique_lock<std::mutex> lk(mutex_);

            cv_.wait(lk, [this] {
                return !queue_.empty() || stop_flag_.load();
            });

            if (stop_flag_.load() && queue_.empty()) {
                break;
            }

            if (queue_.empty()) continue;

            req = queue_.top();
            queue_.pop();
        }

        // ── Speak ─────────────────────────────────────────────────────────────
        speak_sync(req);
    }

    speaking_.store(false);
}

// ─── speak_sync ───────────────────────────────────────────────────────────────
//
// Speaks one utterance synchronously by spawning espeak-ng as a child process.
//
// We use fork() + execvp() rather than popen() or system() so we have the PID
// available immediately — this lets kill_current() terminate the process
// precisely without having to parse popen's internal pipe state.
//
// Steps:
//   1. Record current_pid_ and current_priority_ (under the mutex) so
//      interrupt() / kill_current() can act on them.
//   2. fork() → child calls execvp("espeak-ng", args).
//   3. Parent releases the mutex and calls waitpid() (blocking).
//   4. After waitpid() returns, clear current_pid_ under the mutex.

void TtsEngine::speak_sync(const SpeechRequest& req)
{
    if (config_.verbose) {
        std::cout << "[TTS|" << priority_name(req.priority) << "] "
                  << req.text << "\n" << std::flush;
    }

    // Build argument list for execvp.
    // espeak-ng -s <speed> -p <pitch> -v <voice> -- "<text>"
    std::string speed_str = std::to_string(config_.speed_wpm);
    std::string pitch_str = std::to_string(config_.pitch);

    // We pass the text as a command-line argument (not stdin) to avoid
    // managing a write pipe. The '--' separator ensures the text is not
    // interpreted as a flag even if it starts with '-'.
    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(10);

    // We need non-const char* for execvp. Build a list of std::strings first.
    std::vector<std::string> args = {
        config_.executable,
        "-s", speed_str,
        "-p", pitch_str,
        "-v", config_.voice,
        "--",
        req.text
    };
    for (auto& a : args) {
        argv_ptrs.push_back(const_cast<char*>(a.c_str()));
    }
    argv_ptrs.push_back(nullptr);

    speaking_.store(true);

    pid_t pid = ::fork();

    if (pid < 0) {
        // fork() failed — log and continue.
        speaking_.store(false);
        return;
    }

    if (pid == 0) {
        // ── Child process ─────────────────────────────────────────────────────
        // Redirect stdout/stderr to /dev/null so espeak-ng's output doesn't
        // pollute the parent's terminal.
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
        ::execvp(argv_ptrs[0], argv_ptrs.data());
        // execvp only returns on error.
        ::_exit(1);
    }

    // ── Parent process ────────────────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(mutex_);
        current_pid_      = pid;
        current_priority_ = req.priority;
    }

    // Wait for the child to finish (or be killed by kill_current()).
    int status = 0;
    bool sent_term = false;
    bool sent_kill = false;
    const auto started_at = std::chrono::steady_clock::now();

    while (true) {
        const pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            break;
        }
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        const auto elapsed = std::chrono::steady_clock::now() - started_at;
        if (elapsed > kMaxUtteranceDuration) {
            if (!sent_term) {
                ::kill(pid, SIGTERM);
                sent_term = true;
            } else if (!sent_kill) {
                ::kill(pid, SIGKILL);
                sent_kill = true;
            }
        }

        std::this_thread::sleep_for(kWaitPollInterval);
    }

    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (current_pid_ == pid) {
            current_pid_ = -1;
        }
    }

    speaking_.store(false);
    ++utterances_spoken_;
}

// ─── kill_current ────────────────────────────────────────────────────────────
//
// Sends SIGTERM to the currently running espeak-ng process (if any).
// Must be called with mutex_ held.

void TtsEngine::kill_current()
{
    if (current_pid_ > 0) {
        ::kill(current_pid_, SIGTERM);
    }
}

// ─── build_command ────────────────────────────────────────────────────────────
//
// Builds a shell command string for espeak-ng.
// Used as a fallback for platforms where fork/exec is unavailable.
// (Not used in the primary path — kept for reference.)

std::string TtsEngine::build_command(const std::string& text) const
{
    std::ostringstream ss;
    ss << config_.executable
       << " -s " << config_.speed_wpm
       << " -p " << config_.pitch
       << " -v " << config_.voice
       << " -- " << shell_escape(text);
    return ss.str();
}

// ─── shell_escape ─────────────────────────────────────────────────────────────
//
// Wraps a string in single quotes for safe use as a shell argument.
// Single quotes inside the string are replaced with '\'' (end quote,
// literal single quote, re-open quote).

std::string TtsEngine::shell_escape(const std::string& s)
{
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }
    result += "'";
    return result;
}

// ─── is_duplicate ────────────────────────────────────────────────────────────
//
// Returns true if the same (text, priority) was enqueued within dedup_window.
// Must be called with mutex_ held.

bool TtsEngine::is_duplicate(const std::string& text,
                               SpeechPriority    priority) const
{
    const auto now = std::chrono::steady_clock::now();

    for (const auto& entry : dedup_history_) {
        if (entry.priority != priority) continue;
        if (entry.text     != text)     continue;

        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - entry.enqueued_at);
        if (age < config_.dedup_window) {
            return true;
        }
    }

    return false;
}

// ─── record_enqueue ──────────────────────────────────────────────────────────
//
// Records an enqueue event for dedup tracking.
// Must be called with mutex_ held.

void TtsEngine::record_enqueue(const std::string& text, SpeechPriority priority)
{
    dedup_history_.push_back({
        text,
        priority,
        std::chrono::steady_clock::now()
    });
}

// ─── prune_dedup_history ─────────────────────────────────────────────────────
//
// Removes entries from dedup_history_ that are older than dedup_window.
// Must be called with mutex_ held.
// Keeps the vector small to avoid linear scan overhead at 10 Hz.

void TtsEngine::prune_dedup_history()
{
    const auto now = std::chrono::steady_clock::now();

    dedup_history_.erase(
        std::remove_if(
            dedup_history_.begin(),
            dedup_history_.end(),
            [&](const DedupEntry& e) {
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - e.enqueued_at);
                return age >= config_.dedup_window;
            }
        ),
        dedup_history_.end()
    );
}

} // namespace audio
