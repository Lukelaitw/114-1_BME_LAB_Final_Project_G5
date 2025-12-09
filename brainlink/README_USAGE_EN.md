# BrainLink Raw Data Reading -> Classifier -> Game Control Usage Guide

> **Language Selection / 語言版本選擇**
> 
> - 🇺🇸 [English](README_USAGE_EN.md) ← Current version
> - 🇹🇼 [繁體中文 (Traditional Chinese)](README_USAGE.md)

## Overview

This script (`brainlink_raw.py`) implements a complete data pipeline:
1. **Read raw EEG data from BrainLink device**
2. **Real-time classification using CTNet classifier** (relaxed/focused)
3. **Blink detection**
4. **Send control signals to game**

## Prerequisites

### 1. Install Dependencies

```bash
# Ensure necessary Python packages are installed
pip install cushy-serial numpy torch
```

### 2. Verify Model Files Exist

Ensure model files exist at the following path:
```
server_client/Loso_C_heads_2_depth_8_0/
  ├── model_1.pth
  ├── model_2.pth
  └── ... (other model files)
```

### 3. Connect BrainLink Device

- **Windows**: After Bluetooth pairing, find COM port in "Device Manager" (select "Output" port)
- **Mac**: After pairing, usually `/dev/cu.BrainLink_Pro` or `/dev/cu.BrainLink_Lite`

### 4. Start Game (if controlling game)

The game needs to be started in socket mode:
```bash
cd balance_game
python main.py --socket-input --socket-port 4789
```

## Basic Usage

### Windows System

```bash
# Execute from project root directory
python brainlink/brainlink_raw.py --serial-port COM3
```

### Mac System

```bash
# Execute from project root directory
python brainlink/brainlink_raw.py --serial-port /dev/cu.BrainLink_Pro
```

## Command Line Arguments

### Required Arguments
None (has default values)

### Optional Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--serial-port` | BrainLink serial port path | `COM3` | `--serial-port COM3` (Windows)<br>`--serial-port /dev/cu.BrainLink_Pro` (Mac) |
| `--baud` | Serial port baud rate | `115200` | `--baud 57600` |
| `--game-host` | Game socket host address | `127.0.0.1` | `--game-host 192.168.1.100` |
| `--game-port` | Game socket port | `4789` | `--game-port 5000` |
| `--no-game` | Don't connect to game, only perform classification | `False` | `--no-game` |

## Usage Examples

### Example 1: Basic Usage (Windows)

```bash
python brainlink/brainlink_raw.py --serial-port COM3
```

### Example 2: Mac System, Custom Baud Rate

```bash
python brainlink/brainlink_raw.py \
    --serial-port /dev/cu.BrainLink_Pro \
    --baud 57600
```

### Example 3: Classification Only, No Game Connection

```bash
python brainlink/brainlink_raw.py \
    --serial-port COM3 \
    --no-game
```

### Example 4: Custom Game Address and Port

```bash
python brainlink/brainlink_raw.py \
    --serial-port COM3 \
    --game-host 127.0.0.1 \
    --game-port 5000
```

## How to Find the Correct Serial Port

### Windows System

1. **Method 1: Device Manager**
   - Open "Device Manager"
   - Expand "Ports (COM & LPT)"
   - Find BrainLink-related COM port (usually `COM3`, `COM4`, etc.)
   - **Important**: Select "Output" port (refer to BrainLinkParser README)

2. **Method 2: List All Ports Using Python**
   ```python
   import serial.tools.list_ports
   ports = list(serial.tools.list_ports.comports())
   for p in ports:
       print(f"{p.device}: {p.description}")
   ```

### Mac System

1. **List All Serial Ports**
   ```bash
   ls /dev/cu.*
   ```

2. **Common BrainLink Port Names**
   - `/dev/cu.BrainLink_Pro` (BrainLink Pro)
   - `/dev/cu.BrainLink_Lite` (BrainLink Lite)
   - `/dev/cu.BrainLink` (Generic)

## Output Description

### Initialization Phase

```
[Initialization] Loading CTNet classifier...
[Initialization] Classifier loaded successfully
[GameController] Connected to game 127.0.0.1:4789
[Serial] Connecting to COM3 (Baud rate: 115200)...
[Start] Starting to read BrainLink data...
[Tip] Status output every 1 second; blink detection will show prompt. Ctrl+C to exit.
```

### Runtime Phase

**Status Output (every second)**:
```
[Status] Attention =  45 , Meditation =  60  --> Relaxed
[Status] Attention =  70 , Meditation =  30  --> Focused (Blink detected)
```

**Classification Result Output**:
```
  → Predicted state = Relaxed, prob(Relaxed,Focused) = [0.85 0.15]
  → [Control] Relaxed
  → Detected state = Blink (raw spike), CTNet prob(Relaxed,Focused) = [0.75 0.25]
  → [Control] Blink + Relaxed
```

## Control Logic

### State Mapping

| Detected State | Game Action |
|---------------|-------------|
| **Relaxed** | Lean left (`lean: "Relaxed"`) |
| **Focused** | Lean right (`lean: "Focused"`) |
| **Blink** | Trigger jump (`jump: true` → `jump: false`) + Send current CTNet classification result as lean |

### Classification Parameters

- **Window Size**: 1000 samples (approximately 2 seconds, assuming 500 Hz sampling rate)
- **Stride**: 300 samples (approximately 0.6 seconds)
- **Blink Detection**: Number of samples with amplitude exceeding 150.0 >= 10

## Troubleshooting

### Issue 1: Cannot Connect to Serial Port

```
[Serial] Connecting to COM3 (Baud rate: 115200)...
Error: [Errno 2] No such file or directory: 'COM3'
```

**Solution**:
- Confirm BrainLink device is paired and powered on
- Check if serial port name is correct
- Windows: Check COM port number in "Device Manager"
- Mac: Use `ls /dev/cu.*` to list available ports

### Issue 2: Cannot Connect to Game

```
[GameController] Warning: Cannot connect to game 127.0.0.1:4789
  Please confirm game is started and using --socket-input parameter
```

**Solution**:
- Confirm game is started
- Confirm game is using `--socket-input` parameter
- Check if port number matches (default 4789)
- Use `--no-game` parameter to skip game connection and only perform classification

### Issue 3: Classifier Loading Failed

```
[Initialization] Loading CTNet classifier...
Error: Model file not found
```

**Solution**:
- Confirm `server_client/Loso_C_heads_2_depth_8_0/` directory exists
- Confirm directory contains `model_*.pth` files
- Check file permissions

### Issue 4: No Classification Results Output

```
[Status] Data not yet updated
```

**Possible Causes**:
- Data hasn't accumulated enough (needs at least 1000 samples)
- BrainLink device not transmitting data correctly
- Check if serial port connection is normal

**Solution**:
- Wait a few seconds for data to accumulate
- Check if BrainLink device is working properly
- Confirm `onRaw` callback is being called

## Advanced Configuration

### Adjust Blink Detection Sensitivity

Edit parameters in `brainlink_raw.py`:

```python
# Blink detection parameters
BLINK_AMP_THRESHOLD = 150.0          # Increase = harder to detect, decrease = easier to detect
BLINK_MIN_SAMPLES = int(0.02 * FS)   # At least 20 ms above threshold to count as blink
```

### Adjust Classification Window and Stride

```python
WIN_SAMPLES = 1000         # Window size (number of samples)
STRIDE_SAMPLES = 300       # Stride (number of samples)
```

## Stopping the Program

Press `Ctrl+C` to stop the program. The program will automatically:
- Close game connection
- Close serial port connection
- Clean up resources

## Notes

1. **Sampling Rate**: Currently assumes BrainLink sampling rate is 500 Hz. If different, adjust `FS` parameter
2. **Model Path**: Ensure execution from project root directory, otherwise model path may not resolve correctly
3. **Game Connection**: If game is not started, program will continue running but won't send control signals
4. **Data Buffer**: Raw data accumulates in memory. Pay attention to memory usage during long runs

## Related Files

- `server_client/eeg_server_ctnet.py` - Reference implementation (receives data from Biopac socket)
- `server_client/GAME_CONTROL_README.md` - Game control protocol documentation
- `BrainLinkParser-Python/README.md` - BrainLinkParser usage guide
