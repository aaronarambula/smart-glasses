# Smart Glasses — Copilot Instructions

This is a **C++17 LiDAR-based AI navigation system** for Raspberry Pi that detects obstacles in real time and predicts collisions. The codebase is modular, memory-careful, and heavily commented with domain-specific context (e.g., Kalman filters, TTC geometry, autograd backprop).

## Build & Test

### Quick Build
```bash
cd smart-glasses
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### With Simulator (for development/testing without hardware)
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_SIM=ON
cmake --build . --parallel
./app/smart_glasses --sensor sim --scene crowd
```

### Run Tests
The project does **not** use a traditional test framework. Testing is via:
- **Simulator scenes**: `./app/smart_glasses --sensor sim --scene [crowd|sidewalk|crossing]`
- **Visual debugging**: Use `--verbose --map` flags to print grid and track state
- **Hardware integration**: Real LD06/RPLidar sensors on `/dev/ttyAMA0` or `/dev/ttyUSB0`

### Lint & Format
No automated linting. Code follows these patterns:
- **Headers**: Heavy `// ───` comment dividers (visual grouping)
- **Namespaces**: `autograd::`, `perception::`, `prediction::`, `audio::`, `agent::`
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes

## Architecture Overview

The system is a **real-time pipeline loop** on the main thread (~4–5 ms per frame at 10 Hz):

```
LD06/RPLidar @ 10 Hz
       ↓ ScanFrame (360° point cloud)
  PERCEPTION (sensors/ → perception/)
       ↓ OccupancyMap + DBSCAN clusters + Kalman tracked objects
  PREDICTION (prediction/)
       ↓ TTC engine + aaronnet MLP risk classifier
    AUDIO & AGENT (audio/ + agent/ in parallel)
       ├→ TtsEngine: espeak-ng spoken alerts (priority queue)
       └→ AgentLoop: GPT-4o advice via libcurl HTTPS
```

### Dependency Order (each module imports only from above)
```
autograd/     ← aaronnet C++ autograd engine (no dependencies)
  ↓
sensors/      ← LD06 + RPLidar drivers (no project deps)
  ↓
perception/   ← OccupancyMap, DBSCAN, Kalman tracker
  ↓
prediction/   ← TTC engine, aaronnet MLP risk predictor
  ↓
audio/        ← TTS engine, alert policy (uses espeak-ng)
agent/        ← SceneBuilder, OpenAI client (uses libcurl)
  ↓
app/          ← main.cpp wires everything together
```

## Key Modules & Conventions

### `autograd/` — Hand-written C++ Autograd Engine
- **Core**: `Tensor` class with automatic differentiation (forward & backward passes)
- **Features**: Reverse-mode autodiff, Adam optimizer, weight persistence (binary format)
- **Usage**: All operations return `TensorPtr = std::shared_ptr<Tensor>` to maintain the computation graph
- **Pattern**: `requires_grad=true` on weights; use `NoGradGuard` for inference to skip backward
- **Files**: `tensor.h`, `ops.h`, `layers.h`, `optimizer.h`

### `sensors/` — LiDAR & Serial Drivers
- **Interfaces**: Abstract `LidarBase` base class; implementations for LD06 and RPLidar A1
- **Output**: `ScanFrame` struct (360 points, cartesian {x,y} in mm, 1D array)
- **Serial**: `SerialPort` class wraps UART setup; LD06 uses 230400 baud
- **Key**: Frame is **synchronous per iteration**; main thread blocks on `read_frame()` (< 10 ms)

### `perception/` — Obstacle Detection & Tracking
- **OccupancyMap**: Log-odds 2D grid (10m × 10m, 20mm cells), inverse sensor model
- **Clusterer**: DBSCAN (eps=150mm) on grid cells → `Cluster` objects with centroid & bounds
- **Tracker**: Multi-object Kalman filter (4-state: {px, py, vx, vy}) + Hungarian algorithm assignment
  - **States**: TENTATIVE (age < 3 frames) → CONFIRMED → LOST (1–5 frames unmatched) → DEAD
  - **Key metric**: Velocity (`vx, vy`) is fed to TTC engine and risk predictor
  - **Thread-safe**: Not internally locked; main thread is sole user

### `prediction/` — Time-to-Collision & Risk Classification
- **TTCEngine**: Solves `|p + v·t| = R` (R=300mm collision radius) → exact collision time in ms
  - Returns 7-point trajectory projection, CPA (closest point of approach), TTC
- **RiskPredictor**: aaronnet MLP `24→64→32→4` classifies scene into risk levels
  - **Inputs**: 8 sector distances + 8 sector TTCs + 4 density stats + 4 global stats
  - **Outputs**: 4 logits → softmax → `RiskLevel { CLEAR=0, CAUTION=1, WARNING=2, DANGER=3 }`
  - **Training**: Online Adam on pseudo-labels (rules-based from TTCFrame) every 5 frames
  - **Inference**: Wrapped in `NoGradGuard` to skip backprop; called every frame

### `audio/` — Text-to-Speech & Alert Scheduling
- **TtsEngine**: Priority queue worker thread; spawns `espeak-ng` via fork/exec per utterance
  - **Priority**: DANGER > WARNING > CAUTION > CLEAR (preempts lower in ~20 ms)
  - **Rate limiting**: Per-level cooldowns + escalation bypass + TTC urgency override
  - **Dedup**: Same utterance within 2 seconds is silenced
- **AlertPolicy**: Three-tier rate limiter; converts `RiskLevel` → utterance
  - **Distance lookup**: Hardcoded table converts mm → spoken words ("one point two metres")
  - **TTC rounding**: Always **rounds up** (safety: never undercount urgency)
- **Key**: All speech is natural language, not codes. Designed for visually impaired users

### `agent/` — GPT-4o Navigation Agent
- **SceneBuilder**: Serialises `FullPrediction` to compact JSON (~<400 tokens)
  - Objects in metres/m·s⁻¹; only occupied sectors; training diagnostics included
- **OpenAIClient**: Detached thread per request; libcurl HTTPS POST
  - API key from `OPENAI_API_KEY` env var (checked at init; agent silently disabled if missing)
  - Retry on 429/5xx; hand-written JSON response parser
- **AgentLoop**: Background thread, 200 ms tick; four gates:
  1. **Risk gate**: Must be ≥ CAUTION to fire
  2. **Cooldown gate**: 3 seconds between requests
  3. **In-flight gate**: Only one HTTP request at a time
  4. **Change gate**: Scene must have changed meaningfully
  - DANGER frames wake thread immediately via `condvar`
  - Staleness check drops advice if scene clears before response arrives
- **System prompt** (baked): *"You are a navigation assistant embedded in smart glasses worn by a visually impaired person. Give exactly one short, calm, actionable sentence of navigation advice. Maximum 20 words."*

### `app/` — Entry Point & Main Loop
- **main.cpp**: Orchestrates all modules; runs on main thread (< 5 ms per iteration)
  - CLI flags: `--sensor [ld06|rplidar|sim]`, `--port`, `--no-agent`, `--verbose`, `--map`, etc.
  - Signal handling: SIGINT/SIGTERM sets `g_shutdown`; clean teardown of all threads
  - Thread layout: Main (pipeline) + TtsEngine + AgentLoop + detached OpenAI threads

### `sim/` — Synthetic Testing Environment
- **Optional module** (`#ifdef USE_SIM`); compiled only with `-DUSE_SIM`
- **Scenes**: Crowd, sidewalk, crossing (procedurally generated point clouds)
- **Deterministic**: Useful for reproducible testing without hardware

## Important Patterns & Gotchas

### Memory Management
- **Tensors**: All `Tensor` objects are heap-allocated via `TensorPtr = std::shared_ptr<Tensor>`. The backward graph holds references; tensors must outlive `backward()`.
- **Tracks**: `Tracker` uses `std::unordered_map<uint32_t, Track>` (track ID → state). No external references; safe to modify in-place on main thread.
- **No dynamic allocation in hot loop**: Pre-allocate vectors/grids; reuse buffers across frames.

### Thread Safety
- **Main thread**: Sensor I/O, perception, prediction, audio enqueue, agent push
- **TtsEngine thread**: Pulls from priority queue; uses `fork()/execvp()` to spawn espeak-ng
- **AgentLoop thread**: Conditional variable wakes on DANGER; sleeps 200 ms between ticks
- **OpenAI threads**: Detached; one per in-flight request; max 1 (in-flight gate)
- **Lock-free**: Agent push is atomic swap on main thread; no mutexes on hot path

### Numerical Invariants
- **Distances**: Always **mm** internally; converted to metres for audio/JSON
- **Angles**: 0–360° (0 = forward); sectors are 45° wide (8 sectors)
- **Velocity**: Always **mm/s** in Kalman state; converted to m/s for JSON
- **Time**: Always **seconds** in most APIs; **ms** in TTC output (to-collision time in milliseconds)
- **Risk levels**: Enum { CLEAR=0, CAUTION=1, WARNING=2, DANGER=3 } (used as both index and category)

### Kalman & Tracking
- **State**: `x = [px, py, vx, vy]^T` with constant-velocity model
- **Process noise Q**: Tuned for ~500 mm/s² pedestrian acceleration
- **Measurement noise R**: ~40 mm std dev (LD06 centroid accuracy)
- **Hungarian assignment**: O(N³) optimal matching; gating threshold 800 mm
- **Min confirmation**: 3 frames (TENTATIVE → CONFIRMED)
- **Max missing**: 5 frames (LOST → DEAD)

### TTC & Risk
- **TTC**: Exact time in ms when object enters 300 mm collision radius
- **Pseudo-labels**: Rules-based (distance + TTC thresholds) bootstrap MLP training from frame 1
- **No ground truth**: Labels are synthetic (deterministic rules), not human-annotated
- **Online training**: Adam step every 5 frames; checkpoint persisted to disk

### Command-Line Interface
- Real hardware: `./smart_glasses --sensor ld06 --port /dev/ttyAMA0`
- Simulator: `./smart_glasses --sensor sim --scene crowd --verbose --no-agent`
- Training override: `--no-train` to freeze the MLP
- Checkpoint: `--checkpoint path/to/aaronnet_risk.bin` to load pre-trained weights
- Environment: `OPENAI_API_KEY=sk-...` required for agent; agent silently disabled if missing

## Development Workflows

### Adding a New Sensor Driver
1. Inherit from `LidarBase` in `sensors/lidar_base.h`
2. Implement `read_frame()` → `ScanFrame`
3. Register in `sensors/sensors.h`
4. Add CLI flag in `app/main.cpp`

### Tuning the Kalman Tracker
- **Q matrix** (process noise): Edit `PROCESS_NOISE_*` in `perception/tracker.h`
- **R matrix** (measurement noise): Edit `MEASUREMENT_NOISE_*`
- **Gating threshold**: `GATE_MM` constant
- **Confirmation/deletion**: `MIN_HITS`, `MAX_LOST_FRAMES` constants
- Recompile and test on `--sensor sim --scene crowd --verbose`

### Retraining the Risk MLP
- MLP trains online (pseudo-label supervised) every 5 frames
- To reset: Delete `aaronnet_risk.bin`; system will retrain from scratch
- To freeze: Use `--no-train` flag
- To export: See `prediction/risk_predictor.h` save_checkpoint()

### Adding GPT Prompts or Changing Risk Thresholds
- **System prompt**: Hardcoded in `agent/agent_loop.h`, search for `"You are a navigation assistant"`
- **Risk thresholds for pseudo-labels**: `prediction/risk_predictor.h`, `PseudoLabeller::label_frame()`
- **Alert texts**: `audio/audio.h`, AlertPolicy::text_for_level()

## Debugging Tips

### Logs & Verbose Output
- `--verbose`: Prints per-frame perception, prediction, and audio state
- `--map`: Prints ASCII occupancy grid every frame (slow, ~2 frames/s)
- Environment: `OPENAI_API_KEY=` (unset) silently disables agent without errors

### Runtime Metrics (printed on shutdown)
- Frame timing (sensor, perception, prediction, audio, agent latencies)
- Track count and lifecycle statistics
- MLP inference speed and training loss
- TTS utterance counts and dequeue time

### Simulator Testing
- Always test with `-DUSE_SIM=ON` before hardware deployment
- Use `--scene crowd` (many obstacles, high risk) to stress-test tracking
- Use `--verbose` with simulator for detailed per-frame diagnostics

### Hardware
- Ensure LD06 is on `/dev/ttyAMA0` (GPIO UART) or set `--port` explicitly
- Check `OPENAI_API_KEY` is exported: `echo $OPENAI_API_KEY`
- Verify espeak-ng is installed: `which espeak-ng`
- Test audio separately: `echo "hello" | espeak-ng`
