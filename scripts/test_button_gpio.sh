#!/bin/bash
# ─── test_button_gpio.sh ──────────────────────────────────────────────────────
# Simulates GPIO button presses for testing ButtonAgent on a Raspberry Pi.
#
# Usage:
#   # First terminal: run the smart glasses app
#   ./build/app/smart_glasses --sensor sim
#
#   # Second terminal: simulate 2-second button hold
#   bash test_button_gpio.sh simulate 17 2.0
#
# Real hardware usage (on Raspberry Pi):
#   # Export GPIO 17 and set to input
#   echo 17 | sudo tee /sys/class/gpio/export
#   echo in | sudo tee /sys/class/gpio/gpio17/direction
#
#   # To simulate a button press, toggle the value
#   echo 1 > /sys/class/gpio/gpio17/value  # button pressed
#   echo 0 > /sys/class/gpio/gpio17/value  # button released

set -e

GPIO_PIN="${2:-17}"
HOLD_TIME="${3:-2.0}"
GPIO_PATH="/sys/class/gpio/gpio${GPIO_PIN}/value"

case "${1:-help}" in
    simulate)
        echo "[test_button_gpio] Simulating ${HOLD_TIME}s button hold on GPIO ${GPIO_PIN}"
        
        # Check if GPIO path exists
        if [[ ! -f "$GPIO_PATH" ]]; then
            echo "⚠ GPIO ${GPIO_PIN} not accessible at ${GPIO_PATH}"
            echo "  On Raspberry Pi, export GPIO first:"
            echo "    echo ${GPIO_PIN} | sudo tee /sys/class/gpio/export"
            echo "    echo in | sudo tee /sys/class/gpio/gpio${GPIO_PIN}/direction"
            echo ""
            echo "  On other systems, ButtonAgent gracefully falls back to testing mode."
            exit 1
        fi
        
        # Simulate button press (hold for specified time)
        echo "  [1] Button press (GPIO ${GPIO_PIN} → 1)"
        echo 1 > "$GPIO_PATH"
        
        sleep "$HOLD_TIME"
        
        echo "  [2] Button release (GPIO ${GPIO_PIN} → 0)"
        echo 0 > "$GPIO_PATH"
        
        echo "  ✓ GPIO simulation complete"
        ;;
    
    export)
        echo "[test_button_gpio] Exporting GPIO ${GPIO_PIN}"
        if [[ ! -d "/sys/class/gpio/gpio${GPIO_PIN}" ]]; then
            echo ${GPIO_PIN} | sudo tee /sys/class/gpio/export > /dev/null
            sleep 0.5
        fi
        echo in | sudo tee /sys/class/gpio/gpio${GPIO_PIN}/direction > /dev/null
        echo "  ✓ GPIO ${GPIO_PIN} ready at ${GPIO_PATH}"
        ;;
    
    unexport)
        echo "[test_button_gpio] Unexporting GPIO ${GPIO_PIN}"
        if [[ -d "/sys/class/gpio/gpio${GPIO_PIN}" ]]; then
            echo ${GPIO_PIN} | sudo tee /sys/class/gpio/unexport > /dev/null
        fi
        echo "  ✓ GPIO ${GPIO_PIN} unexported"
        ;;
    
    *)
        echo "Smart Glasses Button GPIO Simulator"
        echo ""
        echo "Usage:"
        echo "  bash test_button_gpio.sh simulate [PIN] [HOLD_TIME]"
        echo "  bash test_button_gpio.sh export [PIN]"
        echo "  bash test_button_gpio.sh unexport [PIN]"
        echo ""
        echo "Examples:"
        echo "  # Simulate 2-second button hold on default pin (17)"
        echo "  bash test_button_gpio.sh simulate"
        echo ""
        echo "  # Simulate 3-second hold on GPIO 26"
        echo "  bash test_button_gpio.sh simulate 26 3.0"
        echo ""
        echo "  # Export GPIO 17 for testing"
        echo "  bash test_button_gpio.sh export 17"
        echo ""
        echo "How to test:"
        echo "  1. Start the app in one terminal:"
        echo "       ./build/app/smart_glasses --sensor sim"
        echo ""
        echo "  2. In another terminal, simulate button hold (press and release):"
        echo "       bash test_button_gpio.sh simulate 17 2.0"
        echo ""
        echo "  3. App will:"
        echo "       - Detect 2-second hold"
        echo "       - Prompt for query on stdin"
        echo "       - Send query to GPT-4o"
        echo "       - Speak response via TTS"
        ;;
esac
