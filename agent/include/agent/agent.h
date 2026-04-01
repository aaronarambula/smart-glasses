#pragma once

// ─── agent.h ─────────────────────────────────────────────────────────────────
// Umbrella header for the agent module.
//
// Include this single file to pull in:
//   - SceneBuilder   : converts FullPrediction → compact JSON for GPT
//   - OpenAIClient   : async HTTPS client for GPT-4o Chat Completions
//   - AgentLoop      : background thread that fires queries + delivers advice
//
// Dependency graph (no cycles):
//
//   prediction/prediction.h
//   perception/perception.h
//   audio/audio.h
//         │
//         ▼
//   scene_builder.h    (SceneBuilder, SceneBuilderConfig)
//         │
//         ▼
//   openai_client.h    (OpenAIClient, OpenAIConfig, ResponseCallback)
//         │
//         ▼
//   agent_loop.h       (AgentLoop, AgentConfig, AgentStats)
//         │
//         ▼
//   agent.h            ← this file
//
// Typical usage in the app module:
//
//   #include "agent/agent.h"
//   #include "audio/audio.h"
//   #include "prediction/prediction.h"
//   #include "perception/perception.h"
//
//   // Construct
//   agent::OpenAIClient  client;                  // reads OPENAI_API_KEY from env
//   agent::SceneBuilder  scene;
//   agent::AgentLoop     loop(client, scene, audio_system);
//
//   loop.start();
//
//   // Called at 10 Hz from the pipeline callback thread:
//   void on_prediction(const prediction::FullPrediction& pred,
//                      const perception::PerceptionResult& perc) {
//       loop.push_prediction(pred, &perc);
//       loop.set_training_info(predictor.training_steps(),
//                              predictor.last_loss());
//   }
//
//   loop.stop();
//
// API key setup (Raspberry Pi):
//   export OPENAI_API_KEY="sk-..."
//   # Or add to /etc/environment for persistence across reboots.
//   # NEVER hardcode the key in source code or CMake files.

#include "scene_builder.h"
#include "openai_client.h"
#include "agent_loop.h"

namespace agent {

// ─── AgentSystem ─────────────────────────────────────────────────────────────
//
// Convenience owner that constructs and wires OpenAIClient + SceneBuilder +
// AgentLoop together into a single object.
//
// This is the single object the app module instantiates for the entire agent
// subsystem. It owns all three components and manages their lifetimes safely.
//
// Lifecycle:
//   1. Construct AgentSystem (configs are optional — sensible defaults apply).
//   2. Call start() — launches the AgentLoop background thread.
//   3. Call push_prediction() once per frame from the pipeline thread.
//   4. Call set_training_info() from the pipeline thread after each MLP step.
//   5. Call stop() before destroying (or let the destructor handle it).
//
// Thread safety:
//   push_prediction()    — lock-free snapshot swap, safe at 10 Hz
//   set_training_info()  — atomic writes, safe from any thread
//   start() / stop()     — call once from the owning thread
//   All other methods    — call from the owning thread only
//
// Disabled mode:
//   If OPENAI_API_KEY is not set, the system operates in disabled mode:
//   push_prediction() is a no-op, no network requests are made, and the
//   AudioSystem never receives agent advice. Everything else still works.

class AgentSystem {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // audio          : AudioSystem that delivers GPT advice to the TTS queue.
    //                  Must outlive this AgentSystem.
    // openai_config  : model, timeouts, system prompt, etc.
    // scene_config   : max_objects, float_precision, etc.
    // agent_config   : query interval, risk gate, change detection, etc.
    AgentSystem(audio::AudioSystem&    audio,
                OpenAIConfig           openai_config = OpenAIConfig{},
                SceneBuilderConfig     scene_config  = SceneBuilderConfig{},
                AgentConfig            agent_config  = AgentConfig{})
        : client_(std::move(openai_config))
        , scene_(std::move(scene_config))
        , loop_(client_, scene_, audio, std::move(agent_config))
    {}

    // Destructor: calls stop() if still running.
    ~AgentSystem() {
        stop();
    }

    // Non-copyable, non-movable (owns an OpenAIClient + AgentLoop with threads).
    AgentSystem(const AgentSystem&)            = delete;
    AgentSystem& operator=(const AgentSystem&) = delete;
    AgentSystem(AgentSystem&&)                 = delete;
    AgentSystem& operator=(AgentSystem&&)      = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Starts the AgentLoop background thread.
    void start() {
        loop_.start();
    }

    // Stops the AgentLoop background thread. Blocks until it exits.
    void stop() {
        loop_.stop();
    }

    // ── Hot-path interface (called at 10 Hz) ──────────────────────────────────

    // Push the latest prediction + perception snapshot to the agent loop.
    // Lock-free on the pipeline thread. The agent thread reads the snapshot
    // asynchronously at its own query cadence.
    void push_prediction(const prediction::FullPrediction& pred,
                         const perception::PerceptionResult* perc_ptr = nullptr)
    {
        if (!loop_.is_running()) return;
        loop_.push_prediction(pred, perc_ptr);
    }

    // Update training diagnostics forwarded into the scene JSON context.
    // Called after each RiskPredictor training step.
    void set_training_info(int steps, float loss) {
        loop_.set_training_info(steps, loss);
    }

    // ── Status ────────────────────────────────────────────────────────────────

    bool     is_running()  const { return loop_.is_running();  }
    bool     is_enabled()  const { return loop_.is_enabled();  }

    // Returns true if the OPENAI_API_KEY environment variable was found.
    bool     has_api_key() const { return client_.has_api_key(); }

    // Cumulative query/response statistics.
    const AgentStats& stats() const { return loop_.stats(); }

    // API request counters from the underlying HTTP client.
    uint64_t requests_sent()      const { return client_.requests_sent();      }
    uint64_t requests_succeeded() const { return client_.requests_succeeded(); }
    uint64_t requests_failed()    const { return client_.requests_failed();    }

    // ── Component access (for tuning / diagnostics) ───────────────────────────

    OpenAIClient&        client()       { return client_; }
    const OpenAIClient&  client() const { return client_; }

    SceneBuilder&        scene()        { return scene_;  }
    const SceneBuilder&  scene()  const { return scene_;  }

    AgentLoop&           loop()         { return loop_;   }
    const AgentLoop&     loop()   const { return loop_;   }

    // ── Quick config shortcuts ────────────────────────────────────────────────

    // Adjust GPT query interval at runtime (e.g. slow down to save API cost).
    void set_query_interval(float seconds) {
        loop_.config().query_interval_s = seconds;
    }

    // Enable or disable verbose logging for the entire agent subsystem.
    void set_verbose(bool v) {
        client_.config().verbose = v;
        loop_.config().verbose   = v;
    }

private:
    OpenAIClient  client_;
    SceneBuilder  scene_;
    AgentLoop     loop_;
};

} // namespace agent
