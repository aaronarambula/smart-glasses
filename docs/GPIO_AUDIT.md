# Raspberry Pi GPIO Connection Audit

## Quick Summary

| Component | GPIO Pin | Type | Status | Location |
|-----------|----------|------|--------|----------|
| Button (voice queries) | 17 | Input | **ACTIVE** | app/main.cpp:762 |
| Haptics motor | (optional) | Output | Disabled | audio/haptics_engine.h |
| LiDAR (LD06) | N/A (UART) | Serial | **ACTIVE** | /dev/ttyAMA0 |

---

## Current GPIO Usage

### 1. **GPIO 17** - Button Input (ACTIVE)
```
Purpose:  2-second hold detection for voice queries
Type:     INPUT (sysfs)
Status:   ACTIVELY USED
Pin:      11 on GPIO header
Config:   app/main.cpp line 762
   
Code:
  agent::ButtonAgent button_agent(17, &agent_sys, &audio);
  button_agent.start();
```

### 2. **GPIO (Configurable)** - Haptics/Vibration (OPTIONAL, DISABLED)
```
Purpose:  Vibration feedback for alerts
Type:     OUTPUT (libgpiod)
Status:   DISABLED (default gpio_pin = -1)
Config:   audio/include/audio/haptics_engine.h

To enable:
  audio::HapticsConfig haptics_cfg;
  haptics_cfg.enabled = true;
  haptics_cfg.gpio_pin = 12;  // Pick any safe GPIO
```

---

## Sensor Connections (Don't Use GPIO)

### LD06 LiDAR (DEFAULT)
- **Connection**: `/dev/ttyAMA0` (UART, not GPIO)
- **Baud**: 230400
- **Status**: ACTIVE
- **Note**: Uses Pi's hardware UART internally (pins 14/15)

### RPLidar A1 (ALTERNATIVE)
- **Connection**: Serial port (USB or UART)
- **Status**: CONFIGURABLE
- **Note**: No GPIO used

---

## Available GPIO Pins for New Devices

### Safe to Use ✓
- GPIO 4, 5, 6, 7, 8, 9, 10, 11
- GPIO 12, 13, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27

### Avoid ✗
- **GPIO 0, 1** - I2C EEPROM (don't touch!)
- **GPIO 2, 3** - I2C on some Pi models
- **GPIO 14, 15** - UART on some Pi models
- **GPIO 17** - Currently used for button

---

## Raspberry Pi GPIO Header Layout

```
Looking at the 40-pin GPIO header from above:

        ┌─────────────────────────────────┐
        │ 1   3.3V    ●●●    2   5V      │
        │ 3   GPIO2   ●●●    4   5V      │
        │ 5   GPIO3   ●●●    6   GND     │
        │ 7   GPIO4   ●●●    8   GPIO14  │
        │ 9   GND     ●●●   10   GPIO15  │
        │11   GPIO17  ●●●   12   GPIO18  │ ← BUTTON HERE
        │13   GPIO27  ●●●   14   GND     │
        │15   GPIO22  ●●●   16   GPIO23  │
        │17   3.3V    ●●●   18   GPIO24  │
        │19   GPIO10  ●●●   20   GND     │
        │21   GPIO9   ●●●   22   GPIO25  │
        │23   GPIO11  ●●●   24   GPIO8   │
        │25   GND     ●●●   26   GPIO7   │
        │27   GPIO0   ●●●   28   GPIO1   │
        │29   GPIO5   ●●●   30   GND     │
        │31   GPIO6   ●●●   32   GPIO12  │
        │33   GPIO13  ●●●   34   GND     │
        │35   GPIO19  ●●●   36   GPIO16  │
        │37   GPIO26  ●●●   38   GPIO20  │
        │39   GND     ●●●   40   GPIO21  │
        └─────────────────────────────────┘
```

---

## How to Check on Your Raspberry Pi

### See all exported GPIO pins:
```bash
ls /sys/class/gpio/
# Output example:
#   gpiochip0  gpio17

# gpio17 means GPIO 17 is exported
```

### Check if GPIO 17 is exported:
```bash
ls /sys/class/gpio/gpio17/
# If exists: GPIO 17 is exported and ready
# If "No such file": needs to be exported first
```

### Read button value:
```bash
cat /sys/class/gpio/gpio17/value
# Output: 0 or 1
# 1 = button not pressed (pull-up)
# 0 = button pressed (pulled to GND)
```

### Monitor button in real-time:
```bash
watch -n 0.1 'cat /sys/class/gpio/gpio17/value'
# Press button while watching - value should toggle 1 → 0 → 1
```

### Export a new GPIO pin:
```bash
echo 27 | sudo tee /sys/class/gpio/export
echo in | sudo tee /sys/class/gpio/gpio27/direction
```

### See all GPIO pins currently in use:
```bash
ls -d /sys/class/gpio/gpio* 2>/dev/null
```

### Use GPIO utility (if installed):
```bash
gpio readall
# Shows complete GPIO status if you have WiringPi installed
```

### Check system information:
```bash
cat /proc/cpuinfo | grep Model
# Shows which Pi model you have

cat /proc/device-tree/model
# Also shows Pi model
```

---

## Wiring Your Button (GPIO 17)

### Physical Connections

```
         Raspberry Pi GPIO Header
         
         ┌──────────────────────┐
         │ PIN 9  (GND)    ●●●  │
         │ PIN 11 (GPIO17) ●    │ ← Wire to button here
         │ PIN 17 (3.3V)   ●●●  │ (with pull-up resistor)
         └──────────────────────┘
```

### Circuit Diagram

```
3.3V (Pin 17) ─────[10kΩ resistor]─────┐
                                        │
                              ┌─[Button]─┐
                              │          │
                          GPIO 17      GND (Pin 9)
                          (Pin 11)     (Pin 6, 14, 20, etc.)
```

### Step-by-Step Wiring

1. **Resistor connection**:
   - One end: 3.3V (Pin 1 or Pin 17)
   - Other end: GPIO 17 (Pin 11)

2. **Button connection**:
   - One side: GPIO 17 (Pin 11)
   - Other side: GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)

3. **Result**:
   - Normally: GPIO 17 reads 1 (pulled high by resistor)
   - When pressed: GPIO 17 reads 0 (pulled to GND)

---

## Adding Another Device (Example: LED on GPIO 27)

### 1. Export GPIO 27

```bash
echo 27 | sudo tee /sys/class/gpio/export
echo out | sudo tee /sys/class/gpio/gpio27/direction
```

### 2. Control from command line

```bash
echo 1 > /sys/class/gpio/gpio27/value   # LED ON
echo 0 > /sys/class/gpio/gpio27/value   # LED OFF
```

### 3. Or control from C++ code

```cpp
#include <fstream>

void turn_led_on() {
    std::ofstream gpio_file("/sys/class/gpio/gpio27/value");
    if (gpio_file.is_open()) {
        gpio_file << "1";
        gpio_file.close();
    }
}

void turn_led_off() {
    std::ofstream gpio_file("/sys/class/gpio/gpio27/value");
    if (gpio_file.is_open()) {
        gpio_file << "0";
        gpio_file.close();
    }
}
```

---

## Enabling Optional Haptics Motor (Vibration Feedback)

The system has a built-in haptics engine but it's disabled by default. Here's how to enable it:

### Option 1: Enable in Code

In your main initialization or audio system setup:

```cpp
#include "audio/haptics_engine.h"

// Create haptics config
audio::HapticsConfig haptics_cfg;
haptics_cfg.enabled = true;
haptics_cfg.gpio_pin = 12;           // GPIO 12 (Pin 32)
haptics_cfg.active_low = true;       // Active LOW (common for motors)
haptics_cfg.pulse_count = 2;         // Number of pulses
haptics_cfg.pulse_on_ms = 180;       // Pulse on time (ms)
haptics_cfg.pulse_off_ms = 120;      // Pulse off time (ms)
haptics_cfg.verbose = false;

// Create and start engine
audio::HapticsEngine haptics(haptics_cfg);
haptics.start();

// Trigger a pulse
haptics.pulse_caution();
```

### Option 2: Wiring a Vibration Motor

If using GPIO 12 (Pin 32):

```
         GPIO 12 (Pin 32)
              │
              ├──[N-Channel Mosfet Gate]
              │
        Drain ─ Connected to Motor
        Source ─ Connected to GND
        
   5V ──[Motor]──[Mosfet Drain]
        GND ──[Mosfet Source]
```

---

## Checking Current GPIO Status on Your Pi

### Quick status check:

```bash
# See which GPIO pins are exported
ls -d /sys/class/gpio/gpio* 2>/dev/null | xargs -I {} basename {}

# Read the button value
cat /sys/class/gpio/gpio17/value

# See the button direction
cat /sys/class/gpio/gpio17/direction
```

### Full system check:

```bash
#!/bin/bash
echo "=== GPIO Status ==="
echo "Exported GPIO pins:"
ls -d /sys/class/gpio/gpio* 2>/dev/null | xargs -I {} basename {} | sort -V

echo ""
echo "GPIO 17 Value:"
cat /sys/class/gpio/gpio17/value

echo ""
echo "GPIO 17 Direction:"
cat /sys/class/gpio/gpio17/direction

echo ""
echo "Total GPIO pins on this Pi:"
cat /sys/class/gpio/gpiochip0/ngpio
```

---

## Troubleshooting GPIO Issues

### GPIO 17 shows "Permission denied"

```bash
# Solution: Make it readable without sudo
sudo chmod 644 /sys/class/gpio/gpio17/value

# Or set permissions for all GPIO:
sudo chmod -R 644 /sys/class/gpio/*/value
```

### GPIO 17 doesn't exist (not exported)

```bash
# Export it:
echo 17 | sudo tee /sys/class/gpio/export
echo in | sudo tee /sys/class/gpio/gpio17/direction

# Verify:
ls /sys/class/gpio/gpio17/
```

### Button always reads 0 (always pressed)

Check wiring:
- Is the pull-up resistor connected to 3.3V (not 5V)?
- Is the button wired correctly (one side to GPIO, one to GND)?
- Is the resistor value correct (10kΩ)?

### Button always reads 1 (never pressed)

Check wiring:
- Is the button actually connected to GND?
- Is the wire not broken?
- Try toggling the pin manually: `echo 0 > /sys/class/gpio/gpio17/value`

---

## Summary Table

| Item | Value | Notes |
|------|-------|-------|
| Button GPIO | 17 | Pin 11 on header |
| Button Type | Input | Reads 1 normally, 0 when pressed |
| LiDAR | /dev/ttyAMA0 | UART, not GPIO |
| Safe GPIO | 4-11, 12, 13, 16, 19-27 | Available for new devices |
| Used GPIO | 17 | Button only |
| Optional GPIO | Any safe pin | For future features (LED, haptics, etc.) |

---

## Quick Reference Commands

```bash
# Check button status
cat /sys/class/gpio/gpio17/value

# Monitor button changes
watch -n 0.1 'cat /sys/class/gpio/gpio17/value'

# Export new GPIO
echo 27 | sudo tee /sys/class/gpio/export

# Set GPIO as output
echo out | sudo tee /sys/class/gpio/gpio27/direction

# Control GPIO output
echo 1 > /sys/class/gpio/gpio27/value   # HIGH
echo 0 > /sys/class/gpio/gpio27/value   # LOW

# List all exported pins
ls /sys/class/gpio/gpio*/

# See button wiring test
gpio readall  # (if wiringPi installed)
```

---

## For More Information

- **Button usage**: See README_BUTTON.md
- **Hardware wiring**: See BUTTON_SETUP.md
- **LiDAR setup**: See QUICKSTART.md or README.md
- **System architecture**: See .github/copilot-instructions.md
