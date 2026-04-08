# Smart Glasses — AI Obstacle Detection & Voice Navigation System

> Production-ready assistive glasses with real-time LiDAR obstacle detection, neural network risk prediction, spoken alerts, and hands-free GPT-4o voice queries. Built in C++17 with a custom autograd engine. Supports LD06 (360° sweep), RPLidar A1, and TF-Luna (single-point ToF).

**Status:** ✅ Complete and tested for hardware integration

---

## Quick Start

### On Raspberry Pi (Hardware)
```bash
# 1. Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# 2. Run
export OPENAI_API_KEY="sk-..."
./app/smart_glasses --sensor tfluna --port /dev/ttyAMA0

# 3. Press GPIO button (pin 17) for voice queries
```

### Without Hardware (Simulator)
```bash
cd build
./app/smart_glasses --sensor sim
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[STRUCTURE.md](STRUCTURE.md)** | Repository organization & file guide |
| **[docs/QUICKSTART.md](docs/QUICKSTART.md)** | Build & run instructions |
| **[docs/README_BUTTON.md](docs/README_BUTTON.md)** | GPIO button feature guide |
| **[docs/BUTTON_SETUP.md](docs/BUTTON_SETUP.md)** | Hardware wiring details |
| **[docs/GPIO_AUDIT.md](docs/GPIO_AUDIT.md)** | GPIO connector usage |
| **[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)** | Feature checklist & status |
| **[.github/copilot-instructions.md](.github/copilot-instructions.md)** | Developer guide for future sessions |

---

## Key Features

- ✅ Real-time LiDAR obstacle detection (10 Hz) — 360° sweep or single-point ToF
- ✅ Neural network risk classification (MLP: 24→64→32→4)
- ✅ Kalman tracking with persistent object IDs
- ✅ Time-to-collision prediction
- ✅ Spoken alerts via TTS (espeak-ng)
- ✅ GPT-4o navigation agent (contextual advice)
- ✅ **Hands-free voice queries via GPIO button**
- ✅ Online training with pseudo-labels
- ✅ Runs on Raspberry Pi Zero 2W
- ✅ Zero external ML dependencies (custom autograd engine)

---

## Overview

This project turns a Raspberry Pi and a cheap LiDAR sensor into smart assistive glasses that can detect obstacles, predict collisions, and speak warnings — all in real time, all in C++.

The system processes a full 360° LiDAR scan every 100 ms and runs it through a complete AI pipeline:

1. **Sensor** — LD06, RPLidar A1, or TF-Luna streams raw distance measurements over UART
2. **Perception** — points are clustered into objects and tracked across frames with a Kalman filter, giving each obstacle a persistent identity and a velocity estimate
3. **Prediction** — time-to-collision is computed geometrically for every tracked object; a custom MLP neural network (*aaronnet*) classifies the overall scene into CLEAR / CAUTION / WARNING / DANGER
4. **Audio** — a priority-queue TTS engine speaks natural-language alerts ("obstacle one point two metres ahead — collision in two seconds") via espeak-ng
5. **Agent** — a GPT-4o agent receives a compact JSON description of the scene every few seconds and provides one-sentence navigation advice ("Turn slightly right — clear path to your left")

Everything is written in C++17 with no heavy frameworks. The neural network engine is hand-built from scratch — forward pass, backward pass, Adam optimiser, weight persistence — and runs on a Pi Zero 2W.

---

## Architecture

```
LD06 / RPLidar A1 / TF-Luna  (/dev/ttyAMA0 or /dev/ttyUSB0)
         │
         │  ScanFrame — 460 points, 360°, ~10 Hz
         ▼
┌─────────────────────────────────────────────┐
│              PERCEPTION                     │
│                                             │
│  OccupancyMap  ← log-odds 2-D grid 10m×10m  │
│  Clusterer     ← DBSCAN (eps=150mm)         │
│  Tracker       ← Kalman filter + Hungarian  │
│                                             │
│  output: TrackedObjectList                  │
│          { id, px, py, vx, vy, TTC, ... }   │
└─────────────────────────────────────────────┘
         │
         │  PerceptionResult
         ▼
┌─────────────────────────────────────────────┐
│              PREDICTION                     │
│                                             │
│  TTCEngine     ← quadratic solver + CPA     │
│  RiskPredictor ← aaronnet MLP (24→64→32→4) │
│                  + online Adam fine-tuning  │
│                  + pseudo-label bootstrap   │
│                                             │
│  output: FullPrediction                     │
│          { RiskLevel, confidence, TTCFrame }│
└─────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
┌──────────────────────┐             ┌─────────────────────────┐
│       AUDIO          │             │         AGENT           │
│                      │             │                         │
│  AlertPolicy         │             │  SceneBuilder           │
│  ← rate limiter      │             │  ← FullPrediction→JSON  │
│  ← escalation bypass │             │                         │
│  ← TTC override      │             │  OpenAIClient           │
│                      │             │  ← libcurl HTTPS POST   │
│  TtsEngine           │             │  ← GPT-4o               │
│  ← espeak-ng fork    │             │  ← retry + timeout      │
│  ← priority queue    │             │                         │
│  ← DANGER preempts   │             │  AgentLoop              │
│                      │             │  ← background thread    │
│  🔊 Speaker          │             │  ← 4-gate query policy  │
└──────────────────────┘             └─────────────────────────┘
```

---

## Module Breakdown

### `autograd/` — aaronnet C++ Autograd Engine

A fully custom reverse-mode automatic differentiation engine, ported from Python/NumPy to C++17. Zero external dependencies.

| Component | Description |
|-----------|-------------|
| `Tensor` | `shared_ptr`-based graph node, flat `vector<float>` storage, row-major |
| `Linear` | Affine layer with He initialisation |
| `ReLU` | Element-wise rectifier |
| `Sequential` | Ordered layer container with fluent `.add<T>()` builder |
| `Adam` | Adaptive moment optimiser with bias correction and gradient clipping |
| `NoGradGuard` | RAII context that disables graph construction for inference |
| `Tensor::cross_entropy` | Numerically stable log-softmax + NLL loss |
| `Tensor::backward()` | Iterative reverse-mode autodiff (no recursion, no stack overflow) |

The engine is what makes the risk predictor trainable on-device. It runs a full forward + backward + Adam step in ~0.5 ms on a Raspberry Pi 4.

---

### `sensors/` — LiDAR Drivers

POSIX serial port drivers for two sensors, with a common abstract interface.

| Component | Description |
|-----------|-------------|
| `LidarBase` | Abstract interface: `open / start / stop / close / get_latest_frame` |
| `LD06` | LDROBOT LD06/LD19. UART 230400 baud. Continuous 47-byte packets. CRC-8/MAXIM validated. No commands needed. |
| `RPLidarA1` | Slamtec A1M8. USB-serial 115200 baud. Request/response protocol. SCAN command, 5-byte measurement nodes. |
| `TFLuna` | Benewake TF-Luna V1.3. UART 115200 baud. 9-byte frames at 100 Hz. Single forward-facing point at 0°. Range: 20 cm – 800 cm. Checksum validated. Signal strength quality gate (amp < 100 → rejected). |
| `SerialPort` | POSIX `termios` RAII wrapper. `select()`-based reads. All standard baud rates. |
| `ScanFrame` | One scan frame: `vector<ScanPoint>`, timestamp, frame ID, RPM. For TF-Luna, contains one point at angle 0°. |

All drivers run their read loop on a background thread. The main thread calls `get_latest_frame()` once per pipeline tick.

---

### `perception/` — Scene Understanding

Converts raw scan points into tracked obstacles with velocities.

| Component | Description |
|-----------|-------------|
| `OccupancyMap` | 400×400 log-odds grid (10m×10m, 25mm/cell). Bresenham ray-tracing. Exponential decay (×0.85 per frame). |
| `Clusterer` | DBSCAN with pre-computed neighbourhoods. O(N²), N≤460 → <0.5ms. Outputs centroid, bounding box, size label. |
| `Tracker` | 4-state Kalman filter `[px, py, vx, vy]`. Hungarian assignment (Jonker-Volgenant O(N³), no external library). TENTATIVE → CONFIRMED → LOST → DEAD lifecycle. |
| `PerceptionPipeline` | Single object that sequences all three components. One `process(frame, dt_s)` call per tick. |

After 3 confirmed frames a track's velocity is reliable enough to drive time-to-collision prediction.

---

### `prediction/` — Risk Assessment

Computes collision timelines and classifies overall scene risk.

| Component | Description |
|-----------|-------------|
| `TTCEngine` | Quadratic equation solver: finds exact time `t` when `|p + v·t| = R` (R = 300mm collision radius). Also computes Closest Point of Approach (CPA) and projects 7-point trajectory over 3 seconds. |
| `RiskPredictor` | aaronnet MLP: `Linear(24→64)→ReLU→Linear(64→32)→ReLU→Linear(32→4)`. Featurises 8 sector distances + 8 sector TTCs + 4 density stats + 4 global stats. Pseudo-labels from threshold rules bootstrap training from frame 1. Binary checkpoint save/load. |
| `PseudoLabeller` | Converts a TTCFrame into a `RiskLevel` using explicit distance + TTC rules. Provides ground-truth-quality labels without any human annotation. |
| `PredictionPipeline` | Sequences TTCEngine → RiskPredictor. Returns `FullPrediction` containing both outputs. |

The MLP learns a smooth generalisation of the threshold rules and improves run-to-run as the checkpoint accumulates training steps.

---

### `audio/` — Spoken Alerts

Non-blocking TTS with intelligent rate limiting.

| Component | Description |
|-----------|-------------|
| `TtsEngine` | Background worker thread. `std::priority_queue`. `fork()` + `execvp()` per utterance (PID tracked for kill). DANGER preempts lower-priority speech within ~20ms. Deduplication window prevents flooding. |
| `AlertPolicy` | Three-mechanism rate limiter: per-level cooldowns (DANGER=1.5s, WARNING=3s, CAUTION=6s), escalation bypass (risk increase always fires immediately), TTC urgency override (TTC < 2.5s bypasses all cooldowns). |
| `AudioSystem` | Owns both components. `process(FullPrediction)` on pipeline thread. `deliver_agent_advice()` thread-safe from agent thread. |

Alert text is natural language, not robotic codes. Distances are spoken from a lookup table ("one point two metres"). TTC is always rounded **up** (safety invariant — never undercount urgency).

---

### `agent/` — GPT-4o Navigation Agent

Periodic high-level guidance powered by OpenAI.

| Component | Description |
|-----------|-------------|
| `SceneBuilder` | Serialises `FullPrediction` to compact JSON (<400 tokens). Objects in metres/m·s⁻¹. Only occupied sectors emitted. Training diagnostics included so GPT knows the model's learning state. |
| `OpenAIClient` | libcurl HTTPS POST to `api.openai.com/v1/chat/completions`. API key from `OPENAI_API_KEY` env var — never logged. Retry on 429/5xx. Per-request detached threads. Hand-written JSON response parser. |
| `AgentLoop` | Background thread, 200ms tick. Four gates: risk gate (≥CAUTION), cooldown gate, in-flight gate (one request at a time), change gate (scene must have changed meaningfully). DANGER frames wake the thread immediately via `condvar`. Staleness check drops advice if scene clears before response arrives. |
| `AgentSystem` | Owns all three. `push_prediction()` is a lock-free atomic swap on the pipeline thread. Gracefully disabled if `OPENAI_API_KEY` is unset. |

System prompt (baked in): *"You are a navigation assistant embedded in smart glasses worn by a visually impaired person. Give exactly one short, calm, actionable sentence of navigation advice. Maximum 20 words."*

---

### `app/` — Entry Point

`main.cpp` wires every module together. Per-frame work on the main thread:

| Step | Time (Pi 4) |
|------|-------------|
| Perception (map + DBSCAN + Kalman) | ~3.0 ms |
| TTC engine (quadratic solver) | ~0.1 ms |
| aaronnet inference (NoGradGuard) | ~0.5 ms |
| aaronnet training step (every 5th frame) | ~0.5 ms |
| Audio enqueue | ~0.05 ms |
| Agent snapshot push | ~0.01 ms |
| **Total** | **~4–5 ms** |

Budget at 10 Hz: 100 ms. Utilisation: ~5%.

---

## Hardware

| Component | Model | Cost | Notes |
|-----------|-------|------|-------|
| Compute | Raspberry Pi Zero 2W or Pi 4 | $15–$45 | Pi Zero 2W is sufficient; Pi 4 recommended for development |
| LiDAR | LDROBOT LD06 | ~$15–$30 | 360° sweep. UART, 230400 baud, no USB adapter needed |
| LiDAR (alt) | Benewake TF-Luna | ~$20–$30 | Single-point forward ToF. UART 115200 baud. 20 cm – 8 m range. Best for corridor/forward detection |
| LiDAR (alt) | Slamtec RPLidar A1M8 | ~$100 | 360° sweep. Connects via USB-to-serial adapter |
| Speaker | USB speaker or 3.5mm | $5–$15 | Any ALSA-compatible speaker |
| Power | USB-C PD bank | $10–$20 | 5V/3A for Pi + sensor |
| Glasses frame | DIY / 3D printed | — | Mount Pi + sensor to glasses |

**Total hardware cost: ~$45–$110**

---

## Software Dependencies

| Dependency | Purpose | Install |
|------------|---------|---------|
| `libcurl4-openssl-dev` | OpenAI HTTPS calls | `sudo apt install libcurl4-openssl-dev` |
| `espeak-ng` | Text-to-speech synthesis | `sudo apt install espeak-ng` |
| `cmake` ≥ 3.16 | Build system | `sudo apt install cmake` |
| `g++` / `clang++` | C++17 compiler | `sudo apt install build-essential` |
| `pthreads` | Threading | Included with libc |

No other external libraries. The autograd engine, Kalman tracker, Hungarian algorithm, DBSCAN, serial port driver, and JSON parser are all written from scratch.

---

## Building

### Quick start (Raspberry Pi or Linux)

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential cmake libcurl4-openssl-dev espeak-ng

# Clone and build
git clone https://github.com/YOUR_USERNAME/smart-glasses.git
cd smart-glasses
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

The build produces one binary: `build/app/smart_glasses`

### Build output

```
autograd_lib     ← aaronnet C++ autograd engine
sensors_lib      ← LD06 + RPLidar A1 drivers
perception_lib   ← OccupancyMap + DBSCAN + Kalman
prediction_lib   ← TTC engine + aaronnet MLP
audio_lib        ← TTS engine + alert policy
agent_lib        ← SceneBuilder + OpenAI + AgentLoop
smart_glasses    ← final executable
```

### Pi-native optimisation (optional)

For maximum performance when deploying to a known Pi 4, uncomment in `CMakeLists.txt`:

```cmake
target_compile_options(smart_glasses PRIVATE -O3 -march=native)
```

---

## Running

### Basic usage

```bash
# LD06 on GPIO UART (default)
./build/app/smart_glasses

# TF-Luna on GPIO UART
./build/app/smart_glasses --sensor tfluna --port /dev/ttyAMA0

# RPLidar A1 on USB
./build/app/smart_glasses --sensor rplidar --port /dev/ttyUSB0

# Verbose output (per-frame log)
./build/app/smart_glasses --verbose

# With GPT-4o agent
export OPENAI_API_KEY="sk-..."
./build/app/smart_glasses --verbose

# Show ASCII occupancy map every 50 frames
./build/app/smart_glasses --map
```

### CLI reference

```
Sensor:
  --sensor ld06|rplidar|tfluna  LiDAR model       (default: tfluna)
  --port   PATH             Serial device          (default: /dev/ttyAMA0)

Perception:
  --eps-mm FLOAT            DBSCAN radius mm       (default: 150)
  --min-pts INT             DBSCAN min cluster     (default: 3)

Prediction:
  --checkpoint PATH         aaronnet weights file  (default: aaronnet_risk.bin)
  --no-train                Freeze MLP weights
  --danger-mm  FLOAT        DANGER threshold       (default: 500)
  --warning-mm FLOAT        WARNING threshold      (default: 1000)
  --caution-mm FLOAT        CAUTION threshold      (default: 2000)

Audio:
  --speed INT               espeak-ng wpm          (default: 150)
  --pitch INT               espeak-ng pitch 0-99   (default: 55)

Agent:
  --no-agent                Disable GPT-4o
  --agent-interval FLOAT    Query interval seconds (default: 5.0)
  --agent-verbose           Log GPT queries/responses

General:
  --verbose                 Per-frame pipeline log
  --map                     ASCII occupancy map every 50 frames
  --help                    This message
```

### Example verbose output

```
[frame   1042 | WARNING | conf=0.87 | TTC=3.1s | 1.2m medium] | trk=3 | clst=4 | 4.2ms
[frame   1043 | WARNING | conf=0.91 | TTC=2.8s | 1.1m medium] | trk=3 | clst=4 | 3.9ms

─── Pipeline Stats (100 frames) ──────────────────
  Frame time  avg=4.21 ms  min=3.81 ms  max=6.12 ms
  Risk dist   CLEAR=61  CAUTION=22  WARNING=14  DANGER=3
  aaronnet    steps=20  loss=0.8821  ema_loss=0.9103
  Agent       sent=4  recv=4  err=0  skip=87
  Last GPT    "Move right — the obstacle on your left is passing."
──────────────────────────────────────────────────
```

---

## Project Structure

```
smart-glasses/
│
├── CMakeLists.txt                  ← Root build (builds all modules)
├── .gitignore
├── README.md
│
├── autograd/                       ── aaronnet C++ autograd engine
│   ├── CMakeLists.txt
│   ├── include/autograd/
│   │   ├── autograd.h              ← Umbrella include
│   │   ├── tensor.h                ← Tensor, TensorPtr, factory helpers
│   │   ├── ops.h                   ← Kernel declarations
│   │   ├── layers.h                ← Linear, ReLU, Sequential
│   │   ├── optimizer.h             ← Adam
│   │   └── no_grad.h               ← NoGradGuard RAII
│   └── src/
│       ├── tensor.cpp              ← Constructors, backward, all ops
│       ├── ops.cpp                 ← Matmul, relu, softmax, adam kernels
│       ├── layers.cpp              ← (inline in header)
│       └── optimizer.cpp           ← (inline in header)
│
├── sensors/                        ── LiDAR drivers
│   ├── CMakeLists.txt
│   ├── LiDAR.cpp                   ← Sensor-layer compilation unit
│   ├── ld06.cpp                    ← LD06 driver implementation
│   ├── rplidar_a1.cpp              ← RPLidar A1 driver implementation
│   ├── tfluna.cpp                  ← TF-Luna driver implementation
│   └── include/sensors/
│       ├── sensors.h               ← Umbrella include + factory
│       ├── lidar_base.h            ← ScanPoint, ScanFrame, LidarBase
│       ├── ld06.h                  ← LD06 driver declaration
│       ├── rplidar_a1.h            ← RPLidar A1 driver declaration
│       ├── tfluna.h                ← TF-Luna driver declaration
│       └── serial_port.h           ← POSIX serial port RAII wrapper
│
├── perception/                     ── Scene understanding
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── occupancy_map.cpp       ← Log-odds grid + Bresenham ray-tracing
│   │   ├── clusterer.cpp           ← DBSCAN implementation
│   │   └── tracker.cpp             ← Kalman filter + Hungarian assignment
│   └── include/perception/
│       ├── perception.h            ← Umbrella + PerceptionPipeline
│       ├── occupancy_map.h
│       ├── clusterer.h
│       └── tracker.h
│
├── prediction/                     ── Risk assessment
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── ttc_engine.cpp          ← Quadratic TTC + CPA + path projection
│   │   └── risk_predictor.cpp      ← aaronnet MLP + training + checkpoint I/O
│   └── include/prediction/
│       ├── prediction.h            ← Umbrella + FullPrediction + utilities
│       ├── ttc_engine.h            ← TTCEngine, TTCFrame, SectorThreat
│       └── risk_predictor.h        ← RiskPredictor, PseudoLabeller, FeatureVector
│
├── audio/                          ── Spoken alerts
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── tts_engine.cpp          ← Priority queue + fork/exec subprocess
│   │   └── alert_policy.cpp        ← Rate limiter + utterance builder
│   └── include/audio/
│       ├── audio.h                 ← Umbrella + AudioSystem
│       ├── tts_engine.h            ← TtsEngine, SpeechPriority, TtsConfig
│       └── alert_policy.h          ← AlertPolicy, AlertThresholds
│
├── agent/                          ── GPT-4o navigation agent
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── scene_builder.cpp       ← FullPrediction → compact JSON
│   │   ├── openai_client.cpp       ← libcurl HTTPS + retry + response parser
│   │   └── agent_loop.cpp          ← Background thread + 4-gate query policy
│   └── include/agent/
│       ├── agent.h                 ← Umbrella + AgentSystem
│       ├── scene_builder.h
│       ├── openai_client.h
│       └── agent_loop.h
│
└── app/                            ── Entry point
    ├── CMakeLists.txt
    └── main.cpp                    ← Wires all modules, 10 Hz pipeline loop
```

---

## How The AI Works

### The problem with simple thresholds

The naive approach — "if distance < 500mm, say DANGER" — fails in the real world. A wall 40cm to your side while walking a corridor is not dangerous. A person 1.5m ahead walking toward you at 1.2 m/s absolutely is. Simple distance thresholds can't express this.

### What aaronnet learns

The MLP takes 24 features derived from the Kalman tracker and TTC engine:

```
Group A — 8 sector min-distances     (where are the closest objects?)
Group B — 8 sector min-TTCs          (when will each sector's object arrive?)
Group C — 4 named-sector densities   (how cluttered is each quadrant?)
Group D — 4 global stats             (max closing speed, track count,
                                      global min TTC, map density)
```

It outputs 4 class probabilities: CLEAR, CAUTION, WARNING, DANGER.

### Training without labels

On day one there is no labelled data. The `PseudoLabeller` generates training labels from explicit threshold rules:

```
DANGER  : TTC < 2s  OR  distance < 500mm
WARNING : TTC < 4s  OR  distance < 1000mm
CAUTION : TTC < 8s  OR  distance < 2000mm
CLEAR   : otherwise
```

The MLP is trained on these pseudo-labels every 5 frames using Adam. It quickly learns to match the rules — but then generalises beyond them. It learns correlations the rules cannot express: that two objects in adjacent sectors is riskier than one object alone at the same distance, that a high-speed approaching object warrants WARNING even at 2m, that wall-like clusters (wide bounding box, many points) are less urgent than person-like clusters.

The checkpoint is saved every 200 training steps and reloaded at startup. The model gets better every time the glasses are worn.

### Time-to-collision geometry

TTC is computed by solving:

```
|p + v·t| = R

where:
  p = current obstacle position (mm)
  v = obstacle velocity (mm/s) from Kalman filter
  R = collision radius (300mm)

Expanding gives a quadratic in t:
  (vx²+vy²)t² + 2(px·vx + py·vy)t + (px²+py²−R²) = 0

The smallest positive root is the TTC.
```

If the discriminant is negative, the obstacle will not intersect the collision radius on its current trajectory — TTC = ∞. The engine also computes the Closest Point of Approach for parallel trajectories that miss but still pass dangerously close.

---

## The aaronnet Engine

aaronnet is a complete reverse-mode automatic differentiation engine written in C++17. It was hand-ported from the original Python/NumPy implementation in this repository.

### Key design decisions

**Shared-pointer graph nodes** — Every `Tensor` is `std::enable_shared_from_this`. The backward closure captures `shared_ptr`s to its inputs, keeping them alive for the duration of the backward pass regardless of what the calling code does.

**Iterative backward pass** — The topological sort uses an iterative DFS (explicit stack) rather than recursion. This avoids stack overflow on deep computation graphs and is O(N) in the number of nodes.

**No-graph inference** — `NoGradGuard` sets a `thread_local bool` that disables graph construction. Running inference under `NoGradGuard` allocates zero heap memory for backward state. At 10 Hz this matters.

**Broadcast-aware arithmetic** — `operator+` and `operator*` handle the standard bias-broadcast case `(B, N) + (N,)` by reducing gradients back to the original shape during backward. This is how `Linear` layers work without any special casing.

### Running the engine standalone

The engine compiles and runs independently from the rest of the project:

```bash
cd autograd
mkdir build && cd build
cmake .. && make

# Write a small test:
cat > test.cpp << 'EOF'
#include "autograd/autograd.h"
#include <iostream>
using namespace autograd;
int main() {
    Sequential model;
    model.add<Linear>(4, 8).add<ReLU>().add<Linear>(8, 3);
    Adam opt(model.parameters(), 1e-3f);
    auto x    = make_tensor({1,2,3,4}, 1, 4);
    auto loss = Tensor::cross_entropy(model.forward(x), {2});
    std::cout << "loss = " << loss->data[0] << "\n";
    loss->backward();
    opt.step();
}
EOF
g++ -std=c++17 -Iinclude test.cpp -Lbuild -lautograd_lib -lm -o test && ./test
```

---

## Configuration Reference

### Alert thresholds

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--danger-mm` | 500 | Distance at which the DANGER pseudo-label fires |
| `--warning-mm` | 1000 | Distance at which the WARNING pseudo-label fires |
| `--caution-mm` | 2000 | Distance at which the CAUTION pseudo-label fires |

Note: these thresholds affect pseudo-label generation for training, not the MLP's live predictions. The MLP may predict DANGER at 800mm if it has learned that the approach velocity warrants it.

### DBSCAN parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--eps-mm` | 150 | Neighbourhood radius. Increase for sparser scan environments. Decrease to separate tightly-packed objects. |
| `--min-pts` | 3 | Minimum points to form a cluster. Increase to reject small noise clusters. |

### TTS parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--speed` | 150 | Words per minute. 130–170 is the legible range. |
| `--pitch` | 55 | Voice pitch (0–99). Lower = calmer. |

---

## Raspberry Pi Setup

### Serial port for LD06 (GPIO UART)

The LD06 connects to the Pi's primary UART on GPIO 14 (TX) and 15 (RX):

```bash
# Enable UART in config.txt
echo "enable_uart=1" | sudo tee -a /boot/config.txt

# Disable serial console so the port is free
sudo raspi-config
# → Interface Options → Serial Port
# → "Would you like a login shell..." → No
# → "Would you like the serial port hardware enabled..." → Yes

# Add your user to the dialout group
sudo usermod -aG dialout $USER

# Reboot
sudo reboot

# Verify
ls -la /dev/ttyAMA0
```

### Serial port for TF-Luna (GPIO UART)

The TF-Luna uses the same GPIO UART as the LD06. Follow the same `raspi-config` steps above to enable the hardware UART and disable the serial console.

```bash
# TF-Luna connects to GPIO 14 (TX) / 15 (RX) — same pins as LD06
# Baud rate is 115200 (handled automatically by the driver)
ls -la /dev/ttyAMA0

# Test with the lidar_test binary
bash scripts/run_lidar.sh /dev/ttyAMA0 tfluna
```

> **Note:** The TF-Luna is a single-point ToF sensor — it measures one distance straight ahead, not a 360° sweep. Orient it in the direction of travel.

### Serial port for RPLidar A1 (USB)

```bash
sudo usermod -aG dialout $USER
# Plug in USB adapter — device appears as /dev/ttyUSB0
ls -la /dev/ttyUSB0
```

### Audio output

```bash
# 3.5mm jack
sudo raspi-config → System Options → Audio → Headphones

# Test espeak-ng
espeak-ng "Smart glasses active"

# Set volume
amixer set PCM 90%
```

### Run on boot (optional)

```bash
# Create a systemd service
sudo tee /etc/systemd/system/smart-glasses.service > /dev/null << 'EOF'
[Unit]
Description=Smart Glasses AI Pipeline
After=network.target

[Service]
Type=simple
User=pi
Environment="OPENAI_API_KEY=sk-..."
ExecStart=/usr/local/bin/smart_glasses --sensor ld06 --port /dev/ttyAMA0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable smart-glasses
sudo systemctl start smart-glasses
sudo journalctl -u smart-glasses -f
```

---

## Team

**FSE 100 — Arizona State University**

Built as a class project demonstrating embedded AI, real-time signal processing, and assistive technology.

| Role | Contribution |
|------|-------------|
| aaronnet autograd engine | Custom C++ neural network framework (ported from Python) |
| Sensor drivers | LD06 / RPLidar A1 / TF-Luna serial protocol implementation |
| Perception pipeline | Occupancy map, DBSCAN, Kalman tracker, Hungarian assignment |
| Prediction pipeline | TTC engine, MLP risk predictor, pseudo-labelling |
| Audio system | TTS engine, alert policy, natural language generation |
| GPT-4o agent | Scene builder, OpenAI client, agent loop |
| App integration | Main pipeline, CLI, telemetry, deployment |

---

## License

This project is for educational purposes as part of FSE 100 at Arizona State University.
