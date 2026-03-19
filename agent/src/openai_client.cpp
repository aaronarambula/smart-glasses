// ─── openai_client.cpp ───────────────────────────────────────────────────────
// Full implementation of the OpenAI Chat Completions HTTPS client.
// See include/agent/openai_client.h for the full design notes.
//
// Dependencies:
//   libcurl   — sudo apt install libcurl4-openssl-dev
//
// The API key is read from the environment variable OPENAI_API_KEY at
// construction time. It is never logged, written to disk, or included in
// any error message string.

#include "agent/openai_client.h"

#include <curl/curl.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace agent {

// ─── Construction ─────────────────────────────────────────────────────────────

OpenAIClient::OpenAIClient(OpenAIConfig config)
    : config_(std::move(config))
    , has_api_key_(false)
{
    // ── Read API key from environment ─────────────────────────────────────────
    // We do this once at construction so the key is never re-read at request
    // time (avoids TOCTOU and reduces per-request overhead).
    const char* key = std::getenv(config_.api_key_env_var.c_str());
    if (key && key[0] != '\0') {
        api_key_     = std::string(key);
        has_api_key_ = true;
    }

    // ── Global libcurl init ───────────────────────────────────────────────────
    // curl_global_init must be called before any other curl function.
    // It is not thread-safe, so we call it here in the constructor which
    // is assumed to run on the main thread before any worker threads start.
    CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (rc != CURLE_OK) {
        throw std::runtime_error(
            std::string("OpenAIClient: curl_global_init failed: ")
            + curl_easy_strerror(rc));
    }
}

// ─── Destructor ───────────────────────────────────────────────────────────────

OpenAIClient::~OpenAIClient()
{
    curl_global_cleanup();
}

// ─── request ─────────────────────────────────────────────────────────────────
//
// Spawns a background thread that performs the HTTP request synchronously.
// Returns true immediately if the thread was spawned.
// Returns false (without invoking callback) if the API key is not set.

bool OpenAIClient::request(const std::string& user_content,
                             ResponseCallback   callback)
{
    if (!has_api_key_) {
        // Don't invoke the callback — the caller should check has_api_key()
        // before calling request() and disable the agent subsystem if false.
        return false;
    }

    ++in_flight_;
    ++requests_sent_;

    // Capture by value so the thread owns everything it needs.
    std::thread([this,
                 user_content,
                 cb = std::move(callback)]() mutable
    {
        do_request(std::move(user_content), std::move(cb));
    }).detach();

    return true;
}

// ─── do_request ──────────────────────────────────────────────────────────────
//
// Runs on a detached background thread.
// Performs the HTTP POST with retry logic, then invokes the callback.
// Decrements in_flight_ before returning.

void OpenAIClient::do_request(std::string user_content, ResponseCallback callback)
{
    const std::string body = build_request_body(user_content);

    std::string response_text;
    bool        success      = false;
    std::string error_msg;

    for (int attempt = 0; attempt <= config_.max_retries; ++attempt) {

        // Wait before retry (not before first attempt).
        if (attempt > 0) {
            std::this_thread::sleep_for(
                std::chrono::duration<float>(config_.retry_delay_s));
        }

        long http_code = 0;
        std::string raw = do_post(body, http_code);

        if (config_.verbose) {
            std::cerr << "[agent] HTTP " << http_code
                      << " (attempt " << attempt + 1 << "/"
                      << config_.max_retries + 1 << ")\n";
        }

        // ── Auth error — do not retry ─────────────────────────────────────────
        if (http_code == 401 || http_code == 403) {
            error_msg = "OpenAI auth error (HTTP "
                      + std::to_string(http_code)
                      + ") — check OPENAI_API_KEY";
            break;
        }

        // ── Rate limit or server error — retry ────────────────────────────────
        if (http_code == 429 || (http_code >= 500 && http_code < 600)) {
            error_msg = "OpenAI server error (HTTP "
                      + std::to_string(http_code) + ")";
            continue;
        }

        // ── curl / network error (http_code == 0) — retry ─────────────────────
        if (http_code == 0) {
            error_msg = "Network error — no HTTP response received";
            continue;
        }

        // ── Success (2xx) ─────────────────────────────────────────────────────
        if (http_code >= 200 && http_code < 300) {
            std::string content = extract_content(raw);
            if (content.empty()) {
                error_msg = "Failed to parse OpenAI response";
                // Don't retry parse errors — the response won't change.
                break;
            }
            response_text = std::move(content);
            success       = true;
            break;
        }

        // ── Other HTTP error — do not retry ───────────────────────────────────
        error_msg = "OpenAI HTTP error: " + std::to_string(http_code);
        break;
    }

    // ── Update counters ───────────────────────────────────────────────────────
    if (success) {
        ++requests_succeeded_;
    } else {
        ++requests_failed_;
    }

    // ── Invoke callback ───────────────────────────────────────────────────────
    if (callback) {
        callback(success, success ? response_text : error_msg);
    }

    --in_flight_;
}

// ─── build_request_body ───────────────────────────────────────────────────────
//
// Builds the JSON request body for the Chat Completions API.
//
// {
//   "model": "gpt-4o",
//   "max_tokens": 80,
//   "temperature": 0.4,
//   "messages": [
//     { "role": "system", "content": "..." },
//     { "role": "user",   "content": "..." }
//   ]
// }
//
// We build this by hand to avoid a JSON library dependency.
// json_escape() handles all special characters in the content strings.

std::string OpenAIClient::build_request_body(const std::string& user_content) const
{
    std::ostringstream ss;
    ss << "{"
       << "\"model\":"       << "\"" << json_escape(config_.model)         << "\","
       << "\"max_tokens\":"  << config_.max_tokens                         << ","
       << "\"temperature\":" << config_.temperature                        << ","
       << "\"messages\":["
           << "{"
               << "\"role\":\"system\","
               << "\"content\":\"" << json_escape(config_.system_prompt)   << "\""
           << "},"
           << "{"
               << "\"role\":\"user\","
               << "\"content\":\"" << json_escape(user_content)            << "\""
           << "}"
       << "]"
       << "}";
    return ss.str();
}

// ─── curl_write_cb ────────────────────────────────────────────────────────────
//
// libcurl write callback — appends received bytes to a std::string.
// `userdata` is a pointer to the std::string response buffer.

size_t OpenAIClient::curl_write_cb(char*  ptr,
                                    size_t size,
                                    size_t nmemb,
                                    void*  userdata)
{
    const size_t total = size * nmemb;
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, total);
    return total;
}

// ─── do_post ─────────────────────────────────────────────────────────────────
//
// Executes one synchronous HTTPS POST using a fresh curl handle.
//
// A fresh handle per request is important:
//   - No shared state between concurrent requests on different threads.
//   - Connection reuse is handled automatically by libcurl's connection cache
//     (per-handle, so each thread gets its own cache — acceptable for our
//     low request rate of 0.2 Hz).
//
// Returns the raw response body. Sets http_code_out to the HTTP status code,
// or 0 on a curl-level error (e.g. DNS failure, timeout, TLS error).

std::string OpenAIClient::do_post(const std::string& body,
                                   long&              http_code_out) const
{
    http_code_out = 0;
    std::string response_buf;

    // ── Init curl handle ──────────────────────────────────────────────────────
    CURL* curl = curl_easy_init();
    if (!curl) {
        return {};
    }

    // ── Build Authorization header ────────────────────────────────────────────
    // "Bearer <key>" — assembled into a single string, never logged.
    const std::string auth_header = "Authorization: Bearer " + api_key_;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");

    // ── Set curl options ──────────────────────────────────────────────────────
    curl_easy_setopt(curl, CURLOPT_URL,            config_.api_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,     body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body.size()));

    // Response write callback.
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,       &response_buf);

    // Timeouts.
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                     static_cast<long>(config_.connect_timeout_s));
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,
                     static_cast<long>(config_.request_timeout_s));

    // TLS verification — always enabled (never skip for production).
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    // Follow redirects (OpenAI may redirect HTTP → HTTPS).
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS,       3L);

    // User-agent string — identifies the glasses project.
    curl_easy_setopt(curl, CURLOPT_USERAGENT,
                     "smart-glasses-agent/1.0 libcurl");

    // ── Execute ───────────────────────────────────────────────────────────────
    CURLcode rc = curl_easy_perform(curl);

    if (rc == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code_out);
    } else if (config_.verbose) {
        std::cerr << "[agent] curl error: " << curl_easy_strerror(rc) << "\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return response_buf;
}

// ─── extract_content ─────────────────────────────────────────────────────────
//
// Minimal parser that extracts choices[0].message.content from the OpenAI
// Chat Completions JSON response.
//
// The full response structure is:
// {
//   "id": "chatcmpl-...",
//   "choices": [
//     {
//       "index": 0,
//       "message": {
//         "role": "assistant",
//         "content": "Turn right — clear path ahead."
//       },
//       "finish_reason": "stop"
//     }
//   ],
//   ...
// }
//
// Strategy: find the first occurrence of "\"content\":" after the first
// "\"choices\":" marker, then extract the string value.
// This is brittle against unusual whitespace but the OpenAI API response
// format is stable and predictable.

std::string OpenAIClient::extract_content(const std::string& json_body)
{
    // Find "choices" array.
    const size_t choices_pos = json_body.find("\"choices\"");
    if (choices_pos == std::string::npos) return {};

    // Find "content" key after "choices".
    const size_t content_pos = json_body.find("\"content\"", choices_pos);
    if (content_pos == std::string::npos) return {};

    // Find the colon after "content".
    size_t colon_pos = json_body.find(':', content_pos);
    if (colon_pos == std::string::npos) return {};

    // Skip whitespace after colon.
    size_t val_start = colon_pos + 1;
    while (val_start < json_body.size() &&
           (json_body[val_start] == ' '  ||
            json_body[val_start] == '\n' ||
            json_body[val_start] == '\r' ||
            json_body[val_start] == '\t')) {
        ++val_start;
    }

    if (val_start >= json_body.size()) return {};

    // Handle JSON null.
    if (json_body.compare(val_start, 4, "null") == 0) return {};

    // Expect a JSON string value starting with '"'.
    if (json_body[val_start] != '"') return {};

    // Extract the string value, handling escape sequences.
    std::string result;
    result.reserve(128);

    size_t i = val_start + 1;   // skip opening '"'
    while (i < json_body.size()) {
        char c = json_body[i];

        if (c == '"') {
            // Closing quote — done.
            break;
        }

        if (c == '\\' && i + 1 < json_body.size()) {
            char esc = json_body[i + 1];
            switch (esc) {
                case '"':  result += '"';  i += 2; continue;
                case '\\': result += '\\'; i += 2; continue;
                case '/':  result += '/';  i += 2; continue;
                case 'n':  result += '\n'; i += 2; continue;
                case 'r':  result += '\r'; i += 2; continue;
                case 't':  result += '\t'; i += 2; continue;
                case 'u':  {
                    // \uXXXX — decode 4-hex-digit codepoint.
                    if (i + 5 < json_body.size()) {
                        // For simplicity: only handle ASCII range (U+0000–U+007F).
                        // GPT responses are predominantly ASCII.
                        unsigned int cp = 0;
                        bool ok = true;
                        for (int h = 0; h < 4; ++h) {
                            char hc = json_body[i + 2 + h];
                            if      (hc >= '0' && hc <= '9') cp = cp * 16 + (hc - '0');
                            else if (hc >= 'a' && hc <= 'f') cp = cp * 16 + (hc - 'a' + 10);
                            else if (hc >= 'A' && hc <= 'F') cp = cp * 16 + (hc - 'A' + 10);
                            else { ok = false; break; }
                        }
                        if (ok && cp < 0x80) {
                            result += static_cast<char>(cp);
                        } else if (ok) {
                            // Non-ASCII: emit replacement character '?'
                            result += '?';
                        }
                        i += 6;
                        continue;
                    }
                    // Malformed — skip.
                    i += 2;
                    continue;
                }
                default:
                    result += esc;
                    i += 2;
                    continue;
            }
        }

        result += c;
        ++i;
    }

    // Trim leading/trailing whitespace from the extracted content.
    size_t first = result.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return {};
    size_t last  = result.find_last_not_of(" \t\n\r");
    return result.substr(first, last - first + 1);
}

// ─── json_escape ─────────────────────────────────────────────────────────────
//
// Escapes a string for embedding as a JSON string value.
// Handles: backslash, double-quote, and ASCII control characters (< 0x20).
// The output does NOT include surrounding double-quotes.

std::string OpenAIClient::json_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 16);

    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof(esc), "\\u%04x",
                                  static_cast<unsigned int>(c));
                    out += esc;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }

    return out;
}

} // namespace agent