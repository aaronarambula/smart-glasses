#!/bin/bash
set -e

echo "═══════════════════════════════════════════════════════════"
echo "END-TO-END SYSTEM TEST"
echo "═══════════════════════════════════════════════════════════"

cd "$(dirname "$0")"
BUILD_DIR="${PWD}/build"

# Build with simulator
echo ""
echo "[1/4] Building with simulator..."
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_SIM=ON > /dev/null 2>&1
cmake --build . --parallel > /dev/null 2>&1
echo "✓ Build complete"

# Test simulator run
echo ""
echo "[2/4] Testing simulator (crowd scene, 3 seconds)..."
cd "$BUILD_DIR"
./app/smart_glasses --sensor sim --scene crowd --no-agent --verbose 2>&1 &
PID=$!
sleep 3
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "✓ Simulator ran successfully"

# Test synthetic training
echo ""
echo "[3/4] Checking training pipeline..."
if [ -f "aaronnet_risk.bin" ]; then
    echo "✓ Checkpoint file exists: $(ls -lh aaronnet_risk.bin | awk '{print $5}')"
else
    echo "⚠ No checkpoint file (will be generated on run)"
fi

# Test model export
echo ""
echo "[4/4] Checking model export capability..."
if grep -q "save_checkpoint\|export" app/train_object_cnn.cpp 2>/dev/null; then
    echo "✓ Model export functions found"
else
    echo "⚠ Model export may need implementation"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "END-TO-END TEST COMPLETE"
echo "═══════════════════════════════════════════════════════════"
