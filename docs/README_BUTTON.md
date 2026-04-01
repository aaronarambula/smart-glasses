# Smart Glasses - GPIO Button Voice Query Feature

## Quick Start

### On Raspberry Pi with Real Hardware

1. **Wire the button to GPIO 17**:
   ```
   Raspberry Pi GPIO 17 ──[Pushbutton]──┬─── GND
                                         └─── 10kΩ resistor to 3.3V
   ```

2. **Build the system**:
   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --parallel
   ```

3. **Run with your LiDAR sensor**:
   ```bash
   export OPENAI_API_KEY="sk-..."  # Required for GPT queries
   ./app/smart_glasses --sensor ld06 --port /dev/ttyAMA0
   ```

4. **Use the button**:
   - **Press and hold button for 2 seconds**
   - **Type your question**: `What obstacles are ahead?`
   - **Press Enter**
   - **Hear the answer spoken via speaker**

---

## Feature Overview

The ButtonAgent enables completely hands-free operation:

```
┌─────────────────────────────────────────────────┐
│ User holds GPIO button (2+ seconds)            │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│ System enters Query Mode                        │
│ "Query mode active. Type your question:"       │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│ User types: "Is there anything in my path?"    │
│ Presses Enter                                  │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│ Query sent to GPT-4o (via OpenAI API)         │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│ Response: "There's a person 1.5 meters ahead" │
│ Spoken aloud via espeak-ng TTS                 │
└─────────────────────────────────────────────────┘
```

---

## Testing Without Hardware

### On macOS or Linux (without GPIO)

**Terminal 1** — Run the simulator:
```bash
./build/app/smart_glasses --sensor sim --no-agent
```

**Terminal 2** — Simulate button press:
```bash
bash test_button_gpio.sh simulate 17 2.0
```

This writes to the GPIO sysfs interface to trigger the button logic (if on a Pi).

---

## Architecture

### Components

```
┌──────────────────────────────────────────────────────┐
│ ButtonAgent (background thread)                     │
│                                                      │
│ • Monitors GPIO 17 via /sys/class/gpio/gpio17/...  │
│ • Detects 2-second button hold (with debouncing)   │
│ • Reads stdin for user query                       │
│ • Calls OpenAIClient::query_direct()               │
│ • Delivers response to AudioSystem (TTS)           │
└──────────────┬───────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  GPIO 17       │
       │  (sysfs)       │
       └────────────────┘
```

### Thread Safety

- **Button monitoring**: Runs on dedicated background thread
- **Query blocking**: Uses mutex + condition_variable in `OpenAIClient::query_direct()`
- **Query mode**: Atomic flag prevents race conditions
- **Graceful shutdown**: All threads properly joined on SIGINT

### Key Methods

**ButtonAgent class** (`agent/include/agent/button_agent.h`):
- `ButtonAgent(gpio_pin, agent_sys, audio_sys)` - Constructor
- `start()` - Begin monitoring (spawns thread)
- `stop()` - Stop monitoring (joins thread)
- `is_query_mode_active()` - Check current state
- `monitor_button()` - Main loop (runs in background)

**OpenAIClient class** (`agent/include/agent/openai_client.h`):
- `query_direct(user_question)` - **NEW** synchronous query method
  - Blocks until response arrives
  - Timeout: request_timeout_s + 5 seconds
  - Throws exception on failure

---

## Configuration

### Change GPIO Pin Number

Edit `app/main.cpp` line ~762:

```cpp
// Default: GPIO 17
agent::ButtonAgent button_agent(17, &agent_sys, &audio);

// Change to GPIO 26:
agent::ButtonAgent button_agent(26, &agent_sys, &audio);
```

Then rebuild:
```bash
cd build && cmake --build . --target smart_glasses
```

### Change Button Hold Time

Edit `agent/button_agent.cpp` line ~120:

```cpp
constexpr int HOLD_THRESHOLD_MS = 2000;  // 2 seconds (default)
constexpr int HOLD_THRESHOLD_MS = 3000;  // 3 seconds (if you prefer)
```

Rebuild to apply.

### Disable Button Feature Completely

1. Remove include from `app/main.cpp`:
   ```cpp
   // #include "agent/button_agent.h"  // REMOVE THIS LINE
   ```

2. Remove instantiation from `app/main.cpp` (~762):
   ```cpp
   // Remove these lines:
   // agent::ButtonAgent button_agent(17, &agent_sys, &audio);
   // button_agent.start();
   // std::cout << "  ✓ Button agent running...\n";
   ```

3. Remove cleanup from `app/main.cpp` (~975):
   ```cpp
   // Remove these lines:
   // button_agent.stop();
   ```

4. Rebuild:
   ```bash
   cd build && cmake --build . --target smart_glasses
   ```

---

## Hardware Setup Details

### Required Components

- Raspberry Pi (3B, 4, Zero 2W, or equivalent)
- Pushbutton switch
- 10kΩ resistor
- Jumper wires

### Wiring Diagram

```
Raspberry Pi GPIO Header
┌─────────────────────────────────┐
│ Pin 1 (3.3V)                   │
│ Pin 2 (5V)                     │
│ Pin 3 (GPIO 2 - SDA)           │
│ Pin 4 (5V)                     │
│ Pin 5 (GPIO 3 - SCL)           │
│ Pin 6 (GND)    ◄──── GND ──┐   │
│ ...                        │   │
│ Pin 11 (GPIO 17)     ◄─────┼─┐ │  ┌──[Pushbutton]──┐
│ ...                        │ │ │  │                 │
│ Pin 16 (GPIO 23)           │ │ │  │                 │
│ Pin 17 (3.3V)  ◄────[10kΩ]─┴─┴─┴──┼────────────────┘
│ ...                        │       │
└─────────────────────────────────────┘
```

**Or in words:**
1. Connect button between GPIO 17 and GND
2. Connect 10kΩ resistor from GPIO 17 to 3.3V (pull-up)
3. Button is now active-low (pressing it pulls GPIO 17 to 0V)

### Export GPIO for User Access (Optional)

If you want to run without sudo:

```bash
# Export GPIO 17 permanently (add to /etc/rc.local or systemd service)
echo 17 | sudo tee /sys/class/gpio/export
echo in | sudo tee /sys/class/gpio/gpio17/direction

# Set permissions so non-root can read
sudo chmod 644 /sys/class/gpio/gpio17/value
```

---

## Interaction Flow

### Step-by-Step User Workflow

1. **User presses and holds button**
   - ButtonAgent monitors GPIO 17 every 8ms
   - Debounces noise (50ms hysteresis)
   - Detects when held for 2+ seconds
   - Sets `query_mode_active_` atomic flag

2. **System prompts for input**
   ```
   [ButtonAgent] Query mode active. Type your question:
   ```

3. **User types question and presses Enter**
   - ButtonAgent reads from stdin (blocking)
   - Example: `What distance is the obstacle ahead?`

4. **ButtonAgent calls OpenAIClient::query_direct()**
   - Synchronous wrapper around async HTTP request
   - Blocks on condition_variable
   - Timeout: 35 seconds (request_timeout_s=30 + 5s buffer)

5. **OpenAI API processes and responds**
   - GPT-4o generates context-aware answer
   - Response includes obstacle distance, direction, etc.

6. **Response is spoken**
   - TtsEngine enqueues response to priority queue
   - espeak-ng fork/exec speaks the text
   - Audio plays through 3.5mm jack or USB speaker

7. **System returns to monitoring**
   - ButtonAgent continues watching GPIO 17
   - Ready for next button press

---

## Troubleshooting

### Button doesn't respond

**Check 1: GPIO pin number**
```bash
# Verify GPIO 17 is correct for your wiring
# Check which pin you connected button to
echo "GPIO pin: $(grep -o 'ButtonAgent(.*' app/main.cpp | cut -d',' -f1)"
```

**Check 2: GPIO permissions**
```bash
# Can you read the GPIO value?
cat /sys/class/gpio/gpio17/value
# Should print 0 or 1 (not "Permission denied")

# If permission denied, export with sudo:
echo 17 | sudo tee /sys/class/gpio/export
echo in | sudo tee /sys/class/gpio/gpio17/direction
```

**Check 3: Button wiring**
- Use multimeter to verify button makes contact with GPIO 17 when pressed
- Verify 10kΩ pull-up resistor is connected to 3.3V

### Query returns no response

**Check: API key**
```bash
echo $OPENAI_API_KEY
# Should be non-empty and start with "sk-"

# If empty:
export OPENAI_API_KEY="sk-..."
```

**Check: Internet**
```bash
curl https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" 2>&1 | head -5
# Should not timeout or refuse connection
```

**Check: Rate limiting**
- OpenAI API has rate limits (depends on your plan)
- Wait a few seconds between queries if getting rate limit errors

### Timeout errors (>30 seconds)

- Usually means API is slow or unreachable
- Check internet: `ping 8.8.8.8`
- Try again in a few seconds (may be temporary API issue)

---

## Performance Notes

- **Button detection latency**: ~100ms (8ms polling + 50ms debounce + reaction time)
- **Query latency**: 10-30 seconds (mostly network/API, not code)
- **TTS latency**: ~500ms for typical response
- **Total interaction time**: ~15-40 seconds per query

---

## Future Enhancements

### 1. Speech-to-Text (STT) Input

Replace stdin with microphone:

```cpp
// Instead of: read from stdin
// std::getline(std::cin, query);

// Do this (pseudocode):
// std::string query = stt_engine.listen(3.0);  // 3-second listening window
```

Candidate libraries:
- **Vosk** - Lightweight, offline
- **PocketSphinx** - Offline, lower latency
- **Google Cloud STT** - Online, higher accuracy

### 2. Multi-Button Layout

Support different button actions:

- **Short press (< 1 second)**: Quick shortcuts (turn left, describe obstacles)
- **Medium press (1-2 seconds)**: Start listening for voice query
- **Long press (> 2 seconds)**: Emergency alert

### 3. LED Feedback

Add visual indication:

```cpp
// Blink LED on GPIO 27 while listening
// Solid LED while processing
// Requires additional GPIO setup
```

### 4. Haptic Feedback

Integrate with existing vibration motors:

```cpp
// Buzz when query received
// Pulse pattern while processing
// Short burst when response ready
```

---

## Code Examples

### Example 1: Programmatic Query (Without Button)

If you want to query GPT-4o without the button:

```cpp
#include "agent/openai_client.h"

// In your code:
agent::OpenAIClient client(agent::OpenAIConfig{});

try {
    std::string response = client.query_direct("What's ahead?");
    std::cout << "Answer: " << response << "\n";
} catch (const std::exception& e) {
    std::cerr << "Query failed: " << e.what() << "\n";
}
```

### Example 2: Custom Button Integration

If you want to use a different GPIO pin:

```cpp
// In app/main.cpp, change line ~762 from:
agent::ButtonAgent button_agent(17, &agent_sys, &audio);

// To use GPIO 26:
agent::ButtonAgent button_agent(26, &agent_sys, &audio);
```

---

## Testing Checklist

- [ ] Button wiring is correct (GPIO 17 + GND + pull-up)
- [ ] GPIO 17 is exported and readable
- [ ] OPENAI_API_KEY environment variable is set
- [ ] Internet connection is working
- [ ] Build succeeds: `cmake --build . --target smart_glasses`
- [ ] App starts: `./app/smart_glasses --sensor sim`
- [ ] Button detected: See "Button agent running" in startup message
- [ ] Query works: Press button 2+ sec, type question, hear response
- [ ] TTS output is audible through speaker
- [ ] Shutdown is clean: No crashes on Ctrl+C

---

## Files

| File | Purpose |
|------|---------|
| `agent/include/agent/button_agent.h` | Button monitoring interface |
| `agent/button_agent.cpp` | Button monitoring implementation |
| `agent/include/agent/openai_client.h` | OpenAI client (with query_direct) |
| `agent/src/openai_client.cpp` | OpenAI implementation (with query_direct) |
| `app/main.cpp` | Integration point |
| `test_button_gpio.sh` | GPIO simulation script |
| `BUTTON_SETUP.md` | Detailed hardware setup guide |

---

## Support & Questions

For issues or questions:

1. Check BUTTON_SETUP.md for hardware-specific guidance
2. Review Troubleshooting section above
3. Examine log output for clues (--verbose flag)
4. Check OpenAI API status: https://status.openai.com
