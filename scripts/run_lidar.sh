#!/bin/bash
# ─── run_lidar.sh ─────────────────────────────────────────────────────────────
# Pull latest, build lidar_test, then run it.
#
# Usage:
#   bash scripts/run_lidar.sh [port] [model]
#
#   port   – serial device              (default: /dev/ttyAMA0)
#   model  – rplidar | ld06 | tfluna   (default: tfluna)
#
# Examples:
#   bash scripts/run_lidar.sh
#   bash scripts/run_lidar.sh /dev/ttyAMA0 tfluna
#   bash scripts/run_lidar.sh /dev/ttyUSB0 rplidar
#   bash scripts/run_lidar.sh /dev/ttyAMA0 ld06

set -e

PORT="${1:-/dev/ttyAMA0}"
MODEL="${2:-tfluna}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_DIR/build"
BINARY="$BUILD_DIR/sensors/lidar_test"

# ── 1. Pull latest ─────────────────────────────────────────────────────────────
echo "==> git pull"
cd "$REPO_DIR"
git pull

# ── 2. Configure (only if build dir doesn't exist) ────────────────────────────
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "==> cmake configure"
    cmake -B "$BUILD_DIR" -S "$REPO_DIR" -DCMAKE_BUILD_TYPE=Release
fi

# ── 3. Build lidar_test ───────────────────────────────────────────────────────
echo "==> cmake build lidar_test"
cmake --build "$BUILD_DIR" --target lidar_test --parallel

# ── 4. Run ────────────────────────────────────────────────────────────────────
echo "==> starting lidar_test  port=$PORT  model=$MODEL"
echo ""
exec "$BINARY" "$PORT" "$MODEL"
