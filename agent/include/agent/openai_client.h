#pragma once

// ─── openai_client.h ─────────────────────────────────────────────────────────
// Async HTTPS client for the OpenAI Chat Completions API (GPT-4o).
//
// Design
// ──────
// The client runs every request on a detached background thread so the
// pipeline thread is never blocked by network I/O. The caller registers a
// callback that is invoked exactly once per request, either with the model's
// response text or with an error string.
//
// libcurl is used for HTTPS — it is the only sane choice on a Raspberry Pi
// (available as `apt install libcurl4-openssl-dev`, no custom TLS setup needed).
//
// API key
// ───────
// The key is read from the environment variable OPENAI_API_KEY at construction.
// It is NEVER logged, stored in any file, or included in error messages.
// If the variable is not set, all requests immediately return an error via the
// callback without making any network call.
//
// Request format
// ──────────────
// POST https://api.openai.com/v1/chat/completions
// Headers:
//   Authorization: Bearer <OPENAI_API_KEY>
//   Content-Type:  application/json
// Body (JSON):
//   {
//     "model":      "<model_name>",        // default "gpt-4o"
//     "max_tokens": <max_tokens>,          // default 80
//     "temperature": <temperature>,        // default 0.4
//     "messages": [
//       { "role": "system",  "content": "<system_prompt>" },
//       { "role": "user",    "content": "<scene_json>"    }
//     ]
//   }
//
// Response parsing
// ────────────────
// The raw JSON response is parsed with a minimal hand-written extractor —
// no JSON library dependency. We only need one field:
//   choices[0].message.content
// which is extracted by scanning for the key pattern.
//
// Retry policy
// ────────────
// On HTTP 429 (rate-limit) or 5xx (server error), the client waits
// RETRY_DELAY_S seconds and retries up to MAX_RETRIES times.
// On HTTP 401/403 (auth error), no retry — the callback fires immediately
// with an error string.
//
// Timeout
// ───────
// connect_timeout_s : time to establish TCP connection (default 5 s)
// request_timeout_s : total time including response (default 8 s)
// If the request times out, the callback fires with a timeout error.
//
// Thread safety
// ─────────────
// request() is thread-safe — multiple threads may call it simultaneously.
// Each call spawns an independent thread and shares no mutable state with
// other in-flight requests. The curl handle is created fresh per request
// (curl_easy_init / curl_easy_cleanup per thread) to avoid sharing state.
//
// Memory
// ──────
// libcurl's global init (curl_global_init) is called once in the constructor
// and curl_global_cleanup is called in the destructor. The client must
// therefore outlive all in-flight requests — ensured by the AgentLoop which
// owns the client and joins all threads on destruction.
//
// Raspberry Pi note
// ─────────────────
// Install libcurl: sudo apt install libcurl4-openssl-dev
// Link with:       target_link_libraries(... CURL::libcurl)

#include <string>
#include <functional>
#include <atomic>
#include <cstdint>

namespace agent {

// ─── ResponseCallback ────────────────────────────────────────────────────────
//
// Called exactly once per request from the background thread.
//   success == true  : text contains the model's response string (trimmed).
//   success == false : text contains a human-readable error description.
//
// The callback MUST be thread-safe — it is invoked from a detached thread,
// not from the thread that called request().

using ResponseCallback = std::function<void(bool success, std::string text)>;

// ─── OpenAIConfig ─────────────────────────────────────────────────────────────

struct OpenAIConfig {
    // ── Model ─────────────────────────────────────────────────────────────────
    std::string model          = "gpt-4o";
    int         max_tokens     = 80;       // keep responses short + fast
    float       temperature    = 0.4f;     // low = focused, consistent responses

    // ── Prompts ───────────────────────────────────────────────────────────────
    // System prompt is baked in at construction and sent with every request.
    // It defines the assistant's role and response style.
    std::string system_prompt =
        "You are a navigation assistant embedded in smart glasses worn by a "
        "visually impaired person. You receive real-time LiDAR sensor data "
        "describing nearby obstacles, their distances, velocities, and "
        "predicted collision times. Give exactly one short, calm, actionable "
        "sentence of navigation advice. Never repeat the raw numbers — "
        "translate them into spatial guidance. If the situation is safe, "
        "say so briefly. Maximum 20 words.";

    // ── Endpoint ──────────────────────────────────────────────────────────────
    std::string api_url        = "https://api.openai.com/v1/chat/completions";

    // ── Timeouts ──────────────────────────────────────────────────────────────
    int connect_timeout_s = 5;
    int request_timeout_s = 8;

    // ── Retry policy ──────────────────────────────────────────────────────────
    int   max_retries     = 2;
    float retry_delay_s   = 1.5f;   // seconds to wait between retries

    // ── Environment variable name for the API key ─────────────────────────────
    // Change this if your deployment uses a different variable name.
    std::string api_key_env_var = "OPENAI_API_KEY";

    // ── Debug ─────────────────────────────────────────────────────────────────
    bool verbose = false;   // if true, logs request/response info to stderr
                            // NOTE: the API key is NEVER logged
};

// ─── OpenAIClient ─────────────────────────────────────────────────────────────

class OpenAIClient {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // Reads OPENAI_API_KEY from the environment at construction.
    // Calls curl_global_init(CURL_GLOBAL_DEFAULT).
    // Throws std::runtime_error if libcurl fails to initialise globally.
    explicit OpenAIClient(OpenAIConfig config = OpenAIConfig{});

    // Destructor: calls curl_global_cleanup().
    // The caller must ensure no in-flight requests remain before destruction
    // (AgentLoop handles this by tracking in-flight count).
    ~OpenAIClient();

    // Non-copyable, non-movable (owns global curl state + atomic counters).
    OpenAIClient(const OpenAIClient&)            = delete;
    OpenAIClient& operator=(const OpenAIClient&) = delete;
    OpenAIClient(OpenAIClient&&)                 = delete;
    OpenAIClient& operator=(OpenAIClient&&)      = delete;

    // ── Request ───────────────────────────────────────────────────────────────

    // Send a chat completion request asynchronously.
    //
    // user_content : the user-turn message (scene JSON from SceneBuilder).
    // callback     : called exactly once from a background thread with
    //                (success, response_text_or_error).
    //
    // Returns true if the request was dispatched (thread spawned).
    // Returns false if the API key is not set — callback is NOT invoked.
    //
    // Thread-safe: may be called from any thread simultaneously.
    bool request(const std::string& user_content,
                 ResponseCallback   callback);

    // ── Status ────────────────────────────────────────────────────────────────

    // Returns true if the API key was found in the environment.
    bool has_api_key() const { return has_api_key_; }

    // Number of requests currently in-flight (dispatched but callback not yet fired).
    int  in_flight_count() const { return in_flight_.load(); }

    // Cumulative counters since construction.
    uint64_t requests_sent()      const { return requests_sent_.load();      }
    uint64_t requests_succeeded() const { return requests_succeeded_.load(); }
    uint64_t requests_failed()    const { return requests_failed_.load();    }

    // ── Config access ─────────────────────────────────────────────────────────

    OpenAIConfig&       config()       { return config_; }
    const OpenAIConfig& config() const { return config_; }

private:
    // ── State ─────────────────────────────────────────────────────────────────

    OpenAIConfig         config_;
    std::string          api_key_;        // from env — never logged
    bool                 has_api_key_;

    std::atomic<int>      in_flight_       { 0 };
    std::atomic<uint64_t> requests_sent_   { 0 };
    std::atomic<uint64_t> requests_succeeded_ { 0 };
    std::atomic<uint64_t> requests_failed_    { 0 };

    // ── Per-request worker (runs on a detached thread) ────────────────────────

    // Performs the HTTP request synchronously (blocking), with retry logic.
    // Invokes callback when done. Decrements in_flight_ before returning.
    void do_request(std::string user_content, ResponseCallback callback);

    // ── libcurl helpers ───────────────────────────────────────────────────────

    // Build the full JSON request body from system_prompt + user_content.
    std::string build_request_body(const std::string& user_content) const;

    // Execute one HTTP POST with the given body. Returns the raw response body.
    // Populates http_code_out. Returns empty string on curl error.
    std::string do_post(const std::string& body, long& http_code_out) const;

    // Extract choices[0].message.content from the raw OpenAI JSON response.
    // Returns empty string if parsing fails.
    static std::string extract_content(const std::string& json_body);

    // Minimal JSON string escaping for building the request body.
    // Escapes backslash, double-quote, and control characters.
    static std::string json_escape(const std::string& s);

    // libcurl write callback — appends received data to a std::string.
    static size_t curl_write_cb(char* ptr, size_t size,
                                size_t nmemb, void* userdata);
};

} // namespace agent