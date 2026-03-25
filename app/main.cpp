// ─── main.cpp ────────────────────────────────────────────────────────────────
// Smart Glasses — top-level entry point.
// Supports real hardware (LD06, RPLidar A1) and the synthetic simulator:
//   ./smart_glasses --sensor sim --scene sidewalk
//   ./smart_glasses --sensor sim --scene crossing --no-agent --verbose
//
// Wires every module into a single 10 Hz real-time pipeline:
//
//   LD06 / RPLidar A1  (sensors/)
//         │  ScanFrame @ 10 Hz
//         ▼
//   PerceptionPipeline (perception/)
//         │  PerceptionResult  [OccupancyMap + Clusters + TrackedObjects]
//         ▼
//   PredictionPipeline (prediction/)
//         │  FullPrediction  [TTCFrame + PredictionResult + RiskLevel]
//         ├──▶  AudioSystem.process()   (audio/)   → espeak-ng TTS
//         └──▶  AgentSystem.push()      (agent/)   → GPT-4o advice @ 0.2 Hz
//
// Build (from smart-glasses/):
//   mkdir -p build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release
//   make -j4
//   ./smart_glasses [options]
//
// Usage:
//   ./smart_glasses [--sensor ld06|rplidar] [--port /dev/ttyAMA0]
//                   [--no-agent] [--no-train] [--verbose] [--map]
//                   [--checkpoint path/to/aaronnet_risk.bin]
//                   [--eps-mm 150] [--danger-mm 500]
//
// Environment:
//   OPENAI_API_KEY=sk-...   Required for agent mode. If unset, agent is
//                           silently disabled; all other subsystems run normally.
//
// Hardware:
//   Raspberry Pi 4 / Zero 2W
//   LD06 LiDAR on /dev/ttyAMA0  (230400 baud, GPIO UART)
//   USB speaker or 3.5mm jack   (espeak-ng → ALSA)
//
// ─── Thread layout ───────────────────────────────────────────────────────────
//
//   Main thread      — pipeline loop: sensor poll → perception → prediction
//                      → audio → agent push.  Everything on this thread is
//                      sequential and runs in < 5 ms per frame.
//   TtsEngine thread — blocks on priority queue, speaks via espeak-ng fork/exec.
//   AgentLoop thread — sleeps 200 ms per tick, fires GPT query when gates pass.
//   OpenAI threads   — one detached thread per in-flight HTTP request (max 1).
//
// ─── Signal handling ──────────────────────────────────────────────────────────
//   SIGINT / SIGTERM — sets g_shutdown flag; the main loop exits cleanly,
//                      all modules are stopped and joined before process exits.

#include "sensors/sensors.h"
#include "perception/perception.h"
#include "prediction/prediction.h"
#include "audio/audio.h"
#include "agent/agent.h"

// sim/ is an optional module — only included when USE_SIM is defined at
// compile time (i.e. when sim_lib is linked). Real hardware builds are
// completely unaffected: this block compiles to nothing without -DUSE_SIM.
#ifdef USE_SIM
#  include "sim/sim.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cmath>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <memory>

// ─── Global shutdown flag ─────────────────────────────────────────────────────

static std::atomic<bool> g_shutdown{ false };

static void signal_handler(int /*sig*/)
{
    g_shutdown.store(true);
}

// ─── CLI argument parser ──────────────────────────────────────────────────────

struct AppConfig {
    // Sensor
    sensors::LidarModel sensor_model = sensors::LidarModel::LD06;
    std::string         port         = "/dev/ttyAMA0";

    // Simulation (only used when sensor_model == Sim)
    std::string         sim_scene    = "sidewalk";   // scene name without "sim://"
    bool                sim_walk     = true;         // user walks forward
    uint32_t            sim_seed     = 0;            // 0 = random, >0 = reproducible
    float               sim_hz       = 10.0f;        // sim scan rate

    // Ultrasonic fallback
    int                 ultra_trigger_pin = 23;
    int                 ultra_echo_pin    = 24;
    float               ultra_hz          = 10.0f;
    float               ultra_mock_mm     = 0.0f;    // 0 = real GPIO sensor

    // Perception
    float eps_mm  = 150.0f;
    int   min_pts = 4;

    // Prediction / training
    std::string checkpoint   = "aaronnet_risk.bin";
    bool        online_train = true;

    // Audio
    int   tts_speed_wpm = 150;
    int   tts_pitch     = 55;
    bool  tts_verbose   = false;

    // Alert thresholds
    float danger_mm  =  500.0f;
    float warning_mm = 1000.0f;
    float caution_mm = 2000.0f;

    // Agent
    bool  agent_enabled      = true;
    float agent_interval_s   = 5.0f;
    bool  agent_verbose      = false;

    // General
    bool  verbose            = false;
    bool  show_map           = false;   // print ASCII occupancy map every N frames
    int   map_every_n_frames = 50;
    int   stats_every_n_frames = 100;  // print pipeline stats every N frames
};

// Prints usage string and exits.
[[noreturn]] static void print_usage_and_exit(const char* prog)
{
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "\n"
        "Sensor:\n"
        "  --sensor ld06|rplidar|ultrasonic|sim sensor model (default: ld06)\n"
        "  --port   PATH             Serial device (default: /dev/ttyAMA0)\n"
        "                            (ignored when --sensor sim)\n"
        "\n"
        "Simulation (only with --sensor sim):\n"
        "  --scene  NAME             Sim scene: sidewalk|crossing|hallway|\n"
        "                                       parking_lot|cyclist_overtake|crowd\n"
        "                            (default: sidewalk)\n"
        "  --sim-seed INT            RNG seed (0=random, >0=reproducible)\n"
        "  --sim-no-walk             Keep simulated user stationary\n"
        "  --sim-hz  FLOAT           Sim scan rate Hz (default: 10.0)\n"
        "\n"
        "Ultrasonic fallback (only with --sensor ultrasonic):\n"
        "  --ultra-trigger INT       GPIO trigger pin (default: 23)\n"
        "  --ultra-echo INT          GPIO echo pin    (default: 24)\n"
        "  --ultra-hz FLOAT          Poll rate Hz     (default: 10.0)\n"
        "  --ultra-mock-mm FLOAT     Desktop/mock fixed distance in mm\n"
        "\n"
        "Perception:\n"
        "  --eps-mm FLOAT            DBSCAN neighbourhood radius mm (default: 150)\n"
        "  --min-pts INT             DBSCAN minimum cluster size    (default: 4)\n"
        "\n"
        "Prediction:\n"
        "  --checkpoint PATH         aaronnet weight file           (default: aaronnet_risk.bin)\n"
        "  --no-train                Disable online MLP fine-tuning\n"
        "  --danger-mm  FLOAT        DANGER threshold mm  (default: 500)\n"
        "  --warning-mm FLOAT        WARNING threshold mm (default: 1000)\n"
        "  --caution-mm FLOAT        CAUTION threshold mm (default: 2000)\n"
        "\n"
        "Audio:\n"
        "  --speed  INT              espeak-ng words/min (default: 150)\n"
        "  --pitch  INT              espeak-ng pitch 0-99 (default: 55)\n"
        "\n"
        "Agent:\n"
        "  --no-agent                Disable GPT-4o agent (no API calls)\n"
        "  --agent-interval FLOAT    GPT query interval seconds (default: 5.0)\n"
        "  --agent-verbose           Log every GPT query + response\n"
        "\n"
        "General:\n"
        "  --verbose                 Verbose pipeline logging\n"
        "  --map                     Print ASCII occupancy map every 50 frames\n"
        "  --help                    Show this message\n"
        "\n"
        "Environment:\n"
        "  OPENAI_API_KEY            Required for agent mode\n"
        "\n"
        "Examples:\n"
        "  " << prog << " --sensor ld06 --port /dev/ttyAMA0 --verbose\n"
        "  " << prog << " --sensor rplidar --port /dev/ttyUSB0 --no-agent\n"
        "  " << prog << " --sensor sim --scene crossing --verbose\n"
        "  " << prog << " --sensor sim --scene crowd --sim-seed 42 --no-agent\n";
    std::exit(0);
}

static AppConfig parse_args(int argc, char* argv[])
{
    AppConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: " << flag << " requires an argument\n";
                std::exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h") {
            print_usage_and_exit(argv[0]);
        }
        else if (arg == "--sensor") {
            std::string s = next("--sensor");
            if (s == "ld06" || s == "LD06") {
                cfg.sensor_model = sensors::LidarModel::LD06;
                if (cfg.port == "/dev/ttyAMA0") cfg.port = "/dev/ttyAMA0";
            } else if (s == "rplidar" || s == "RPLidar" || s == "a1") {
                cfg.sensor_model = sensors::LidarModel::RPLidarA1;
                if (cfg.port == "/dev/ttyAMA0") cfg.port = "/dev/ttyUSB0";
            } else if (s == "ultrasonic" || s == "ultra" || s == "Ultrasonic") {
                cfg.sensor_model = sensors::LidarModel::Ultrasonic;
                cfg.port = sensors::default_port(cfg.sensor_model);
            } else if (s == "sim" || s == "Sim" || s == "SIM") {
#ifndef USE_SIM
                std::cerr << "error: --sensor sim requires the sim_lib to be "
                             "linked.\n"
                             "Rebuild with: cmake .. -DUSE_SIM=ON\n";
                std::exit(1);
#else
                cfg.sensor_model = sensors::LidarModel::Sim;
                cfg.port = "sim://sidewalk";   // default; overridden by --scene
#endif
            } else {
                std::cerr << "error: unknown sensor '" << s
                          << "' (use ld06, rplidar, ultrasonic, or sim)\n";
                std::exit(1);
            }
        }
        else if (arg == "--scene") {
            cfg.sim_scene = next("--scene");
            // If the user set --scene before --sensor sim, we still update port.
            cfg.port = "sim://" + cfg.sim_scene;
        }
        else if (arg == "--sim-seed")    { cfg.sim_seed = static_cast<uint32_t>(std::stoul(next("--sim-seed"))); }
        else if (arg == "--sim-no-walk") { cfg.sim_walk = false; }
        else if (arg == "--sim-hz")      { cfg.sim_hz   = std::stof(next("--sim-hz")); }
        else if (arg == "--ultra-trigger"){ cfg.ultra_trigger_pin = std::stoi(next("--ultra-trigger")); }
        else if (arg == "--ultra-echo")   { cfg.ultra_echo_pin    = std::stoi(next("--ultra-echo")); }
        else if (arg == "--ultra-hz")     { cfg.ultra_hz          = std::stof(next("--ultra-hz")); }
        else if (arg == "--ultra-mock-mm"){ cfg.ultra_mock_mm     = std::stof(next("--ultra-mock-mm")); }
        else if (arg == "--port")            { cfg.port              = next("--port");            }
        else if (arg == "--eps-mm")          { cfg.eps_mm            = std::stof(next("--eps-mm"));     }
        else if (arg == "--min-pts")         { cfg.min_pts           = std::stoi(next("--min-pts"));    }
        else if (arg == "--checkpoint")      { cfg.checkpoint        = next("--checkpoint");            }
        else if (arg == "--no-train")        { cfg.online_train      = false;                           }
        else if (arg == "--danger-mm")       { cfg.danger_mm         = std::stof(next("--danger-mm"));  }
        else if (arg == "--warning-mm")      { cfg.warning_mm        = std::stof(next("--warning-mm")); }
        else if (arg == "--caution-mm")      { cfg.caution_mm        = std::stof(next("--caution-mm")); }
        else if (arg == "--speed")           { cfg.tts_speed_wpm     = std::stoi(next("--speed"));      }
        else if (arg == "--pitch")           { cfg.tts_pitch         = std::stoi(next("--pitch"));      }
        else if (arg == "--no-agent")        { cfg.agent_enabled     = false;                           }
        else if (arg == "--agent-interval")  { cfg.agent_interval_s  = std::stof(next("--agent-interval")); }
        else if (arg == "--agent-verbose")   { cfg.agent_verbose     = true;                            }
        else if (arg == "--verbose")         { cfg.verbose           = true;  cfg.tts_verbose = true;   }
        else if (arg == "--map")             { cfg.show_map          = true;                            }
        else {
            std::cerr << "error: unknown argument '" << arg << "'\n"
                      << "Run with --help for usage.\n";
            std::exit(1);
        }
    }

    return cfg;
}

// ─── Pipeline statistics ──────────────────────────────────────────────────────

struct PipelineStats {
    uint64_t frames_total      = 0;
    uint64_t frames_danger     = 0;
    uint64_t frames_warning    = 0;
    uint64_t frames_caution    = 0;
    uint64_t frames_clear      = 0;
    double   total_pipeline_ms = 0.0;
    double   min_pipeline_ms   = 1e9;
    double   max_pipeline_ms   = 0.0;
    uint64_t training_steps    = 0;
    float    last_loss         = 0.0f;
    float    smoothed_loss     = 0.0f;

    void record(prediction::RiskLevel risk, double pipeline_ms) {
        ++frames_total;
        total_pipeline_ms += pipeline_ms;
        if (pipeline_ms < min_pipeline_ms) min_pipeline_ms = pipeline_ms;
        if (pipeline_ms > max_pipeline_ms) max_pipeline_ms = pipeline_ms;

        switch (risk) {
            case prediction::RiskLevel::DANGER:  ++frames_danger;  break;
            case prediction::RiskLevel::WARNING: ++frames_warning; break;
            case prediction::RiskLevel::CAUTION: ++frames_caution; break;
            default:                             ++frames_clear;   break;
        }
    }

    double avg_pipeline_ms() const {
        return frames_total > 0 ? total_pipeline_ms / frames_total : 0.0;
    }

    void print(const agent::AgentSystem* agent_sys) const {
        std::cout << "\n─── Pipeline Stats (" << frames_total << " frames) ───────────────\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Frame time  avg=" << avg_pipeline_ms()
                  << " ms  min=" << min_pipeline_ms
                  << " ms  max=" << max_pipeline_ms << " ms\n";
        std::cout << "  Risk dist   CLEAR="   << frames_clear
                  << "  CAUTION=" << frames_caution
                  << "  WARNING=" << frames_warning
                  << "  DANGER="  << frames_danger  << "\n";
        if (training_steps > 0) {
            std::cout << std::setprecision(4);
            std::cout << "  aaronnet    steps=" << training_steps
                      << "  loss=" << last_loss
                      << "  ema_loss=" << smoothed_loss << "\n";
        }
        if (agent_sys) {
            const auto& s = agent_sys->stats();
            std::cout << "  Agent       sent=" << s.queries_sent
                      << "  recv=" << s.responses_received
                      << "  err="  << s.api_errors
                      << "  skip=" << s.queries_skipped << "\n";
            if (!s.last_advice.empty()) {
                std::cout << "  Last GPT    \"" << s.last_advice << "\"\n";
            }
        }
        std::cout << "──────────────────────────────────────────────\n\n"
                  << std::flush;
    }
};

// ─── Per-frame verbose log ────────────────────────────────────────────────────

static void log_frame(const prediction::FullPrediction& pred,
                      const perception::PerceptionResult& perc,
                      double pipeline_ms)
{
    // One compact line per frame.
    // [frame 1042 | WARN | conf=0.87 | TTC=3.1s | 1.2m ahead | trk=3 | 4.2ms]
    std::cout << pred.log_str()
              << " | trk=" << perc.confirmed_count()
              << " | clst=" << perc.clusters.size();

    char buf[32];
    std::snprintf(buf, sizeof(buf), " | %.1fms\n", pipeline_ms);
    std::cout << buf << std::flush;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // ── Parse CLI ─────────────────────────────────────────────────────────────
    AppConfig cfg = parse_args(argc, argv);

    // ── Install signal handlers ───────────────────────────────────────────────
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── Banner ────────────────────────────────────────────────────────────────
    std::cout <<
        "╔══════════════════════════════════════════════════════════╗\n"
        "║           Smart Glasses  —  AI Obstacle Detection        ║\n"
        "║   aaronnet autograd  ·  Kalman tracker  ·  GPT-4o agent  ║\n"
        "╚══════════════════════════════════════════════════════════╝\n\n";

    // When sim is selected, assemble the full scene URI now (handles the case
    // where --scene was passed before --sensor sim, or vice versa).
#ifdef USE_SIM
    if (cfg.sensor_model == sensors::LidarModel::Sim) {
        cfg.port = "sim://" + cfg.sim_scene;
    }
#endif
    if (cfg.sensor_model == sensors::LidarModel::Ultrasonic) {
        std::ostringstream uri;
        if (cfg.ultra_mock_mm > 0.0f) {
            uri << "ultrasonic://mock?mm=" << cfg.ultra_mock_mm
                << "&hz=" << cfg.ultra_hz;
        } else {
            uri << "ultrasonic://" << cfg.ultra_trigger_pin
                << "," << cfg.ultra_echo_pin
                << "?hz=" << cfg.ultra_hz;
        }
        cfg.port = uri.str();

        // Keep the ultrasonic fallback from inheriting a stale LiDAR-trained
        // checkpoint unless the user explicitly asked for one.
        if (cfg.checkpoint == "aaronnet_risk.bin") {
            cfg.checkpoint = "aaronnet_ultrasonic.bin";
        }

        // Forward-only ultrasonic is a placeholder path. Default to frozen
        // heuristic behavior unless the user explicitly asked for training.
        if (cfg.online_train) {
            cfg.online_train = false;
        }
    }

    std::cout << "Sensor      : " << sensors::model_name(cfg.sensor_model)
              << " on " << cfg.port << "\n";
    std::cout << "Checkpoint  : " << cfg.checkpoint << "\n";
    std::cout << "Training    : " << (cfg.online_train ? "online (Adam)" : "frozen") << "\n";
    std::cout << "Agent       : " << (cfg.agent_enabled ? "enabled (GPT-4o)" : "disabled") << "\n";
    std::cout << "Thresholds  : DANGER=" << cfg.danger_mm / 1000.0f
              << "m  WARNING=" << cfg.warning_mm / 1000.0f
              << "m  CAUTION=" << cfg.caution_mm / 1000.0f << "m\n\n";

    // ══════════════════════════════════════════════════════════════════════════
    // 1. SENSORS — range sensor driver
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "[1/6] Opening sensor...\n";
    std::unique_ptr<sensors::LidarBase> lidar;
    try {
#ifdef USE_SIM
        if (cfg.sensor_model == sensors::LidarModel::Sim) {
            // Build the SimConfig from CLI args and construct directly.
            // This avoids going through the extern-C bridge and gives full
            // access to SimConfig fields (seed, walk, hz).
            sim::SimConfig sim_cfg;
            sim_cfg.rng_seed    = cfg.sim_seed;
            sim_cfg.user_walks  = cfg.sim_walk;
            sim_cfg.scan_hz     = cfg.sim_hz;
            lidar = sim::make_sim_lidar(cfg.port, sim_cfg);
        } else
#endif
        if (cfg.sensor_model == sensors::LidarModel::Ultrasonic) {
            lidar = std::make_unique<sensors::UltrasonicFallback>(
                cfg.ultra_trigger_pin,
                cfg.ultra_echo_pin,
                cfg.ultra_hz,
                cfg.ultra_mock_mm);
        } else
        {
            lidar = sensors::make_lidar(cfg.sensor_model, cfg.port);
        }
    } catch (const std::exception& e) {
        std::cerr << "FATAL: Cannot create sensor driver: " << e.what() << "\n";
        return 1;
    }

    if (!lidar->open()) {
        std::cerr << "FATAL: Cannot open sensor on " << cfg.port
                  << ": " << lidar->error_message() << "\n";
        if (cfg.sensor_model != sensors::LidarModel::Sim &&
            cfg.sensor_model != sensors::LidarModel::Ultrasonic) {
            std::cerr << "  → Check cable connection and port permissions.\n"
                      << "  → Run: sudo chmod 666 " << cfg.port << "\n";
        }
        return 1;
    }

    if (!lidar->start()) {
        std::cerr << "FATAL: Cannot start sensor: " << lidar->error_message() << "\n";
        lidar->close();
        return 1;
    }

    std::cout << "  ✓ " << lidar->model_name() << " scanning\n";
#ifdef USE_SIM
    if (cfg.sensor_model == sensors::LidarModel::Sim) {
        std::cout << "  ✓ Scene      : " << cfg.sim_scene << "\n";
        std::cout << "  ✓ Seed       : "
                  << (cfg.sim_seed == 0 ? "random" : std::to_string(cfg.sim_seed)) << "\n";
        std::cout << "  ✓ User walk  : " << (cfg.sim_walk ? "yes" : "no") << "\n";
        std::cout << "  ✓ Scan rate  : " << cfg.sim_hz << " Hz\n";
    }
#endif
    if (cfg.sensor_model == sensors::LidarModel::Ultrasonic) {
        if (cfg.ultra_mock_mm > 0.0f) {
            std::cout << "  ✓ Mode       : mock\n";
            std::cout << "  ✓ Distance   : " << cfg.ultra_mock_mm << " mm\n";
        } else {
            std::cout << "  ✓ Trigger pin: GPIO" << cfg.ultra_trigger_pin << "\n";
            std::cout << "  ✓ Echo pin   : GPIO" << cfg.ultra_echo_pin << "\n";
        }
        std::cout << "  ✓ Poll rate  : " << cfg.ultra_hz << " Hz\n";
        std::cout << "  ✓ Output     : narrow forward synthetic cluster\n";
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. PERCEPTION — occupancy map + DBSCAN clusterer + Kalman tracker
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "[2/6] Initialising perception pipeline...\n";
    perception::PerceptionPipeline perception(cfg.eps_mm, cfg.min_pts);
    std::cout << "  ✓ OccupancyMap(10m×10m, 25mm/cell)  "
              << "DBSCAN(eps=" << cfg.eps_mm << "mm)  "
              << "KalmanTracker\n";

    // ══════════════════════════════════════════════════════════════════════════
    // 3. PREDICTION — TTC engine + aaronnet MLP risk predictor
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "[3/6] Loading aaronnet risk predictor...\n";
    prediction::PredictionPipeline pred_pipeline(cfg.checkpoint, cfg.online_train);

    // Override pseudo-labeller thresholds from CLI.
    auto& labeller = pred_pipeline.risk_predictor().labeller();
    labeller.danger_dist_mm  = cfg.danger_mm;
    labeller.warning_dist_mm = cfg.warning_mm;
    labeller.caution_dist_mm = cfg.caution_mm;

    {
        const int steps = pred_pipeline.risk_predictor().training_steps();
        if (steps > 0) {
            std::cout << "  ✓ Loaded checkpoint (" << steps
                      << " prior training steps, "
                      << "ema_loss=" << pred_pipeline.risk_predictor().smoothed_loss()
                      << ")\n";
        } else {
            std::cout << "  ✓ Starting fresh (He-init weights)\n";
        }
    }

    std::cout << "  ✓ MLP: Linear(24→64)→ReLU→Linear(64→32)→ReLU→Linear(32→4)\n";

    // ══════════════════════════════════════════════════════════════════════════
    // 4. AUDIO — TTS engine + alert policy
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "[4/6] Starting audio system...\n";

    audio::TtsConfig tts_cfg;
    tts_cfg.speed_wpm = cfg.tts_speed_wpm;
    tts_cfg.pitch     = cfg.tts_pitch;
    tts_cfg.verbose   = cfg.tts_verbose;

    audio::AlertThresholds alert_thresh;
    // Cooldowns are intentionally left at their defaults (DANGER=1.5s, etc.)
    // so the policy is immediately usable. The user can tune via recompile
    // or future runtime config.

    audio::AudioSystem audio(tts_cfg, alert_thresh);
    audio.start();

    // Startup chime — lets the user know the glasses are active.
    audio.speak("Smart glasses active. Scanning for obstacles.",
                audio::SpeechPriority::CAUTION);

    std::cout << "  ✓ TTS engine running (espeak-ng, "
              << cfg.tts_speed_wpm << " wpm)\n";

    // ══════════════════════════════════════════════════════════════════════════
    // 5. AGENT — OpenAI GPT-4o integration
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "[5/6] Initialising agent subsystem...\n";

    // AgentSystem is built even when agent_enabled=false — it just never
    // makes API calls because the loop is not started.
    agent::AgentConfig agent_cfg;
    agent_cfg.query_interval_s       = cfg.agent_interval_s;
    agent_cfg.verbose                = cfg.agent_verbose;
    agent_cfg.min_risk_to_query      = prediction::RiskLevel::CAUTION;

    agent::OpenAIConfig openai_cfg;
    openai_cfg.verbose               = cfg.agent_verbose;
    // System prompt is the default embedded in OpenAIConfig — crafted for
    // visually impaired navigation guidance in ≤20 words.

    agent::AgentSystem agent_sys(audio,
                                  std::move(openai_cfg),
                                  agent::SceneBuilderConfig{},
                                  std::move(agent_cfg));

    if (cfg.agent_enabled) {
        if (agent_sys.has_api_key()) {
            agent_sys.start();
            std::cout << "  ✓ GPT-4o agent running (interval="
                      << cfg.agent_interval_s << "s)\n";
        } else {
            std::cout << "  ⚠ OPENAI_API_KEY not set — agent disabled.\n"
                      << "    Set it with: export OPENAI_API_KEY=\"sk-...\"\n";
        }
    } else {
        std::cout << "  ✓ Agent disabled by --no-agent flag\n";
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 6. WARM-UP — wait for first valid scan frame
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "[6/6] Waiting for first LiDAR scan frame...\n";
    {
        int wait_ms = 0;
        while (!g_shutdown.load()) {
            auto frame = lidar->get_latest_frame();
            if (!frame.empty()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            wait_ms += 50;
            if (wait_ms > 5000) {
                std::cerr << "FATAL: No scan frame received after 5 seconds.\n"
                          << "  → Check LiDAR power and serial connection.\n"
                          << "  → Check sensor error: " << lidar->error_message() << "\n";
                lidar->stop();
                lidar->close();
                audio.stop();
                return 1;
            }
        }
    }

    if (g_shutdown.load()) {
        lidar->stop();
        lidar->close();
        audio.stop();
        return 0;
    }

    std::cout << "  ✓ First frame received\n\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  Running — press Ctrl+C to stop\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n"
              << std::flush;

    // ══════════════════════════════════════════════════════════════════════════
    // MAIN PIPELINE LOOP — runs until SIGINT / SIGTERM
    // ══════════════════════════════════════════════════════════════════════════
    //
    // The LD06 pushes frames at ~10 Hz. We poll get_latest_frame() at up to
    // 20 Hz and process each new frame exactly once (tracked by frame_id).
    //
    // Per-frame work on this thread (all sequential):
    //   perception.process()        ~3 ms
    //   pred_pipeline.process()     ~0.6 ms  (inference) + ~0.5 ms (training, every 5th)
    //   audio.process()             ~0.05 ms (enqueue, no blocking)
    //   agent_sys.push_prediction() ~0.01 ms (atomic pointer swap)
    //   ─────────────────────────────────────
    //   Total per frame             ~4-5 ms  (well under 100 ms budget at 10 Hz)

    PipelineStats stats;
    uint64_t last_frame_id     = UINT64_MAX;
    auto     last_new_frame_at = std::chrono::steady_clock::now();

    // dt timing: measured wall-clock time between consecutive frames.
    auto last_frame_time = std::chrono::steady_clock::now();
    bool first_frame     = true;

    while (!g_shutdown.load()) {

        // ── Poll for a new frame ───────────────────────────────────────────────
        auto frame = lidar->get_latest_frame();

        if (frame.empty() || frame.frame_id == last_frame_id) {
            // No new frame yet — yield briefly then poll again.
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            const auto stall_s = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - last_new_frame_at).count();
            if (stall_s > 2.0) {
                std::cerr << "FATAL: Sensor stopped producing new frames for "
                          << std::fixed << std::setprecision(1)
                          << stall_s << " seconds.\n";
                if (!lidar->error_message().empty()) {
                    std::cerr << "  → Sensor error: " << lidar->error_message() << "\n";
                }
                break;
            }

            // Surface sensor errors (non-fatal — driver keeps retrying).
            if (!lidar->error_message().empty() && cfg.verbose) {
                std::cerr << "[sensor] " << lidar->error_message() << "\n";
            }
            continue;
        }

        last_frame_id = frame.frame_id;
        last_new_frame_at = std::chrono::steady_clock::now();

        // ── Measure dt ────────────────────────────────────────────────────────
        const auto frame_start_time = std::chrono::steady_clock::now();
        float dt_s = 0.0f;

        if (!first_frame) {
            dt_s = std::chrono::duration_cast<std::chrono::duration<float>>(
                frame_start_time - last_frame_time).count();
            // Clamp dt to [0.01, 0.5] s — handles startup jitter and pauses.
            dt_s = std::max(0.01f, std::min(dt_s, 0.5f));
        }
        first_frame     = false;
        last_frame_time = frame_start_time;

        // ─────────────────────────────────────────────────────────────────────
        // STEP 1 — PERCEPTION
        // OccupancyMap update + DBSCAN clustering + Kalman tracking
        // ─────────────────────────────────────────────────────────────────────
        perception::PerceptionResult perc = perception.process(frame, dt_s);

        // Local occupancy density: fraction of 8×8 cell neighbourhood
        // around the origin that is occupied. Passed to the MLP as feature[23].
        const float local_density =
            perception.map().local_density(/*radius_mm=*/1500.0f);

        // ─────────────────────────────────────────────────────────────────────
        // STEP 2 — PREDICTION
        // TTC (quadratic solver + CPA) + aaronnet MLP inference + training
        // ─────────────────────────────────────────────────────────────────────
        auto pred_out = pred_pipeline.process(perc, local_density);

        // Assemble into FullPrediction for downstream consumers.
        prediction::FullPrediction full_pred;
        full_pred.ttc        = std::move(pred_out.ttc);
        full_pred.prediction = std::move(pred_out.prediction);

        // Update stats with MLP training diagnostics.
        const auto& rp            = pred_pipeline.risk_predictor();
        stats.training_steps      = static_cast<uint64_t>(rp.training_steps());
        stats.last_loss           = rp.last_loss();
        stats.smoothed_loss       = rp.smoothed_loss();

        // ─────────────────────────────────────────────────────────────────────
        // STEP 3 — AUDIO
        // Rate-limited alert policy → TTS priority queue → espeak-ng
        // ─────────────────────────────────────────────────────────────────────
        audio.process(full_pred);

        // ─────────────────────────────────────────────────────────────────────
        // STEP 4 — AGENT
        // Atomic snapshot push → GPT query fires on agent thread when gated
        // ─────────────────────────────────────────────────────────────────────
        if (cfg.agent_enabled && agent_sys.is_running()) {
            agent_sys.push_prediction(full_pred, &perc);

            // Forward MLP training diagnostics into the scene JSON so GPT
            // can acknowledge the model's learning state.
            if (full_pred.prediction.trained_this_frame) {
                agent_sys.set_training_info(
                    rp.training_steps(),
                    rp.last_loss());
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // STEP 5 — TELEMETRY
        // Timing measurement, verbose log, stats, map dump
        // ─────────────────────────────────────────────────────────────────────
        const auto frame_end_time = std::chrono::steady_clock::now();
        const double pipeline_ms  = std::chrono::duration_cast<
            std::chrono::duration<double>>(
                frame_end_time - frame_start_time).count() * 1000.0;

        stats.record(full_pred.risk_level(), pipeline_ms);

        // Verbose per-frame log.
        if (cfg.verbose) {
            log_frame(full_pred, perc, pipeline_ms);
        }

        // Periodic stats printout.
        if (stats.frames_total % static_cast<uint64_t>(cfg.stats_every_n_frames) == 0) {
            const agent::AgentSystem* agent_ptr =
                (cfg.agent_enabled && agent_sys.is_running())
                ? &agent_sys : nullptr;
            stats.print(agent_ptr);
        }

        // Optional ASCII occupancy map dump.
        if (cfg.show_map &&
            stats.frames_total % static_cast<uint64_t>(cfg.map_every_n_frames) == 0)
        {
            std::cout << perception.map().debug_ascii(8) << std::flush;
        }

    } // end main pipeline loop

    // ══════════════════════════════════════════════════════════════════════════
    // GRACEFUL SHUTDOWN
    // Stop all subsystems in reverse dependency order:
    //   agent → audio → sensor
    // (perception and prediction are stack-allocated — no explicit stop needed)
    // ══════════════════════════════════════════════════════════════════════════

    std::cout << "\n[shutdown] Stopping agent...\n";
    agent_sys.stop();

    std::cout << "[shutdown] Stopping audio...\n";
    audio.stop();

    std::cout << "[shutdown] Stopping sensor...\n";
    lidar->stop();
    lidar->close();

    // Final stats printout.
    std::cout << "\n[shutdown] Final statistics:\n";
    stats.print(nullptr);

    // Save the aaronnet checkpoint one last time so no training steps are lost.
    if (cfg.online_train && stats.training_steps > 0) {
        try {
            pred_pipeline.risk_predictor().save_weights(cfg.checkpoint);
            std::cout << "[shutdown] aaronnet weights saved → "
                      << cfg.checkpoint << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[shutdown] Warning: failed to save weights: "
                      << e.what() << "\n";
        }
    }

    std::cout << "[shutdown] Clean exit.\n";
    return 0;
}
