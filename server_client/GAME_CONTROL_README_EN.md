# Game Control Integration Guide

> **Language Selection / 語言版本選擇**
> 
> - 🇺🇸 [English](GAME_CONTROL_README_EN.md) ← Current version
> - 🇹🇼 [繁體中文 (Traditional Chinese)](GAME_CONTROL_README.md)

## Overview

This project has integrated the functionality to transmit EEG prediction states to the game.

## Features

`eeg_server_ctnet.py` can now:
1. Receive EEG data from BIOPAC
2. Perform real-time classification using CTNet model
3. Automatically send control signals to the game based on prediction results

## Usage

### 1. Start the Game (Must Start First)

Start the game with socket input enabled in one terminal:

```bash
source .venv/bin/activate
cd balance_game
python main.py --socket-input --socket-port 4789
```

### 2. Start EEG Classification Server

Start the classification server in another terminal:

```bash
source .venv/bin/activate
cd server_client
python eeg_server_ctnet.py
```

By default, it will try to connect to `127.0.0.1:4789`. If the game is on a different host or port, you can use:

```bash
python eeg_server_ctnet.py --game-host 127.0.0.1 --game-port 4789
```

If you only want to perform classification without connecting to the game, you can use:

```bash
python eeg_server_ctnet.py --no-game
```

### 3. Test Connection

Before starting the game and server, you can use the test script to verify the connection:

```bash
python test_game_control.py
```

This will enter interactive mode where you can manually send control commands for testing.

## Control Logic

The default control mapping is as follows:

- **Relaxed** → Lean left (`lean: -0.5`)
- **Focused** → Lean right (`lean: 0.5`)
- **Blink** → Trigger jump (`jump: true` then `jump: false`)

### Custom Control Logic

You can modify the control logic in the `GameController.update_control()` method in `eeg_server_ctnet.py`:

```python
def update_control(self, label, prob=None):
    if label == 'Relaxed':
        # Custom control for relaxed state
        self.send_command({"lean": -0.5})
    elif label == 'Focused':
        # Custom control for focused state
        self.send_command({"lean": 0.5})
    # ... other states
```

## Game Control Command Format

JSON command format received by the game:

```json
{"lean": -0.35}      // Lean value, range -1.0 (left) to 1.0 (right)
{"jump": true}       // Trigger jump
{"jump": false}      // Stop jump
{"reset": true}      // Reset control
```

You can include multiple commands in one message:

```json
{"lean": 0.1, "jump": true}
```

## Test Script Usage

`test_game_control.py` provides interactive testing:

```bash
python test_game_control.py
```

Available commands:
- `lean <value>` - Set lean value (-1.0 to 1.0)
- `jump` - Trigger jump
- `reset` - Reset control
- `test` - Test connection
- `demo` - Execute demo sequence
- `quit` - Exit

You can also test connection only:

```bash
python test_game_control.py --test-only
```

## Troubleshooting

### Connection Refused

If you see `Connection refused` error:

1. Confirm the game is started and using `--socket-input` parameter
2. Confirm the port number is correct (default 4789)
3. Use `test_game_control.py` to test the connection

### No Control Signals

If classification is normal but no control signals are sent:

1. Check if `GameController` successfully connected (check startup logs)
2. Confirm class names in `CLASS_NAMES` match the logic in `update_control()`
3. Check if prediction results are being generated (check log output)

### Modify Class Names

If your model's class names are different, modify in `eeg_server_ctnet.py`:

```python
CLASS_NAMES = ['Relaxed', 'Focused']  # Change to your class names
```

And update the corresponding logic in the `update_control()` method.

## Code Structure

- `eeg_server_ctnet.py` - Main server, contains:
  - `GameController` class: Manages socket connection with game
  - `OnlineCTNet` class: Online EEG classification
  - `main()` function: Main program logic

- `test_game_control.py` - Test tool for verifying connection and control functionality

## Notes

1. **Startup Order**: Must start the game first, then start the EEG server
2. **Port Conflicts**: Ensure ports used by game and server don't conflict
3. **Data Format**: Ensure BIOPAC data format matches `parse_lines_to_values()` expectations
4. **Model Path**: Confirm model directory `Loso_C_heads_2_depth_8_0` exists and contains model files

## Related Files

- [Project Main README](../README.md)
- [Balance Game README](../balance_game/README.md)
- [BrainLink Usage Guide](../brainlink/README_USAGE.md)

