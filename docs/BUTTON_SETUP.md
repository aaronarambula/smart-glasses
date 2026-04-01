# GPIO Button Setup & Testing

## Feature Overview

The **ButtonAgent** module enables hands-free voice interaction via GPIO button:

1. **Press and hold GPIO pin 17 for 2 seconds** → System enters query mode
2. **Type your question** on stdin → Query is sent to GPT-4o
3. **Speaker plays response** via TTS (espeak-ng)

Perfect for visually impaired users who can't look at a screen.

## Hardware Setup

### Raspberry Pi (Real Hardware)

1. **Wire button to GPIO 17** (and GND):
   ```
   GPIO 17 ───[Button]───┬─── GND
                          └─── 10kΩ resistor to 3.3V (pull-up)
   ```

2. **Export GPIO for sysfs access**:
   ```bash
   echo 17 | sudo tee /sys/class/gpio/export
   echo in | sudo tee /sys/class/gpio/gpio17/direction
   ```

3. **Run the app** (button is active automatically):
   ```bash
   ./build/app/smart_glasses --sensor ld06 --port /dev/ttyAMA0
   ```

### Without Hardware (Testing)

ButtonAgent gracefully falls back to a testing mode if GPIO is not accessible (e.g., on macOS/Linux without a Pi).

## Testing on Raspberry Pi

### Scenario 1: Real Button
1. Start the app:
   ```bash
   ./build/app/smart_glasses --sensor ld06
   ```
2. Press and hold button on GPIO 17 for 2+ seconds
3. Type: `What obstacles are ahead?`
4. Press Enter
5. App queries GPT-4o and speaks response

### Scenario 2: GPIO Simulation (via sysfs)
In two terminals:

**Terminal 1** — Run app:
```bash
./build/app/smart_glasses --sensor sim --no-agent
```

**Terminal 2** — Simulate button press:
```bash
bash test_button_gpio.sh simulate 17 2.0
```

This writes to `/sys/class/gpio/gpio17/value` to simulate a button hold.

### Scenario 3: Test on Non-Pi System
The button is disabled gracefully on non-Pi systems. To fully test the button code path, you'd need a Raspberry Pi with actual GPIO.

## ButtonAgent Architecture

### Header: `agent/include/agent/button_agent.h`
- `ButtonAgent(gpio_pin, agent_sys, audio_sys)` - Constructor
- `start()` - Begin monitoring (spawns background thread)
- `stop()` - Stop monitoring
- `is_query_mode_active()` - Check if in query mode

### Implementation: `agent/button_agent.cpp`
- **GPIO access** via sysfs (`/sys/class/gpio/gpio17/value`)
- **Debouncing** with 50ms intervals
- **2-second hold detection** with hysteresis
- **stdin input** for queries (ready for speech-to-text upgrade)
- **TTS integration** via `audio_->deliver_agent_advice(response)`

### Integration: `app/main.cpp`
- Instantiated alongside AgentSystem
- Receives pointers to AgentSystem and AudioSystem
- Runs on dedicated background thread
- Clean shutdown on SIGINT/SIGTERM

## Query Flow

```
1. User holds button 2+ seconds
   └─> ButtonAgent detects in monitor_button() thread

2. System prints: "Query mode active. Type your question:"
   └─> Ready for stdin input

3. User types question + Enter
   └─> ButtonAgent reads stdin (blocks until newline)

4. ButtonAgent calls agent_.query_direct(question)
   └─> Synchronous wrapper around OpenAI API

5. OpenAIClient::query_direct()
   └─> Dispatches async HTTP request
   └─> Blocks on condition_variable until response

6. Response received → TTS engine speaks it
   └─> "That obstacle is 0.5 meters ahead"

7. Back to monitoring button for next press
```

## Configuration

### Change GPIO Pin
Edit main.cpp line ~762:
```cpp
agent::ButtonAgent button_agent(26, &agent_sys, &audio);  // GPIO 26 instead of 17
```

### Change Hold Time
Edit button_agent.cpp line ~120:
```cpp
constexpr int HOLD_THRESHOLD_MS = 3000;  // 3 seconds instead of 2
```

### Disable Button Feature
At build time:
- Remove `#include "agent/button_agent.h"` from main.cpp
- Remove ButtonAgent instantiation from main.cpp
- Recompile

## Troubleshooting

### Button not responding
1. Check GPIO pin number matches your wiring (default: 17)
2. Verify GPIO is exported: `ls -la /sys/class/gpio/gpio17/`
3. Check permissions: should be readable without sudo

### GPIO path doesn't exist
If `/sys/class/gpio/gpio17/` missing, export manually:
```bash
echo 17 | sudo tee /sys/class/gpio/export
```

### Query times out (>30 seconds)
- Likely OPENAI_API_KEY not set or invalid
- Check: `echo $OPENAI_API_KEY`
- Set if missing: `export OPENAI_API_KEY="sk-..."`

### Query returns error
- Check internet connectivity
- Verify API key is valid
- Check API rate limits

## Future Enhancements

1. **Speech-to-text (STT)** instead of stdin
   - Libraries: pocketsphinx, vosk, Google Cloud STT
   - Would replace stdin read in button_agent.cpp

2. **Multi-button layout**
   - Short press: quick shortcuts (turn left, describe area)
   - Long press: GPT query (current behavior)
   - Double-tap: emergency alert

3. **LED feedback** on button
   - Blink while listening
   - Solid when processing query
   - Requires GPIO output setup

4. **Haptic feedback** (already have vibration motors)
   - Buzz when query received
   - Pulse while processing
   - Could integrate with haptics_engine

## Files Modified

- `agent/include/agent/button_agent.h` - New header (71 lines)
- `agent/button_agent.cpp` - New implementation (165 lines)
- `agent/include/agent/openai_client.h` - Added `query_direct()` signature
- `agent/src/openai_client.cpp` - Implemented `query_direct()` (~50 lines)
- `agent/CMakeLists.txt` - Added button_agent.cpp to build
- `app/main.cpp` - Integrated ButtonAgent lifecycle
- `test_button_gpio.sh` - GPIO simulation script (new)
