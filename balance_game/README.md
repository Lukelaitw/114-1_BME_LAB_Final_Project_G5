# Balance Game Usage Guide

> **Language Selection / 語言版本選擇**
> 
> - 🇺🇸 [English](README.md) ← Current version
> - 🇹🇼 [繁體中文 (Traditional Chinese)](README_chinese.md)

## First Time: From the Project Root

```
source .venv/bin/activate
python -m pip install -r requirements.txt
```
# Afterwards entering the virtual environment
```
source .venv/bin/activate
```
# compiling the app
```
python -m compileall balance_game                                                
```
# run the app with 2 terminal
```
source .venv/bin/activate
python main.py --socket-input --socket-port 4789
```

```
source .venv/bin/activate
python tools/brainlink_serial_bridge.py \
    --serial-port /dev/cu.BrainLink_Lite \
    --profile assets/blink_energy_profile.json \
    --game-port 4789 \
    --verbose \
    --debug-sensors

```



## Blink-to-jump via BrainLink

The game can react to BrainLink / NeuroSky blink events using the built-in ThinkGear socket service.

1. Pair and start the BrainLink headset with the official ThinkGear Connector (or compatible service).  
   Ensure it is streaming JSON packets on `127.0.0.1:13854`.
2. Launch the game with blink support:

   ```bash
   python main.py --brainlink
   ```

Optional overrides:

- `--blink-threshold <value>` – change the blink strength needed to trigger a jump (default 55).
- `--brainlink-host <host>` / `--brainlink-port <port>` – connect to a non-default ThinkGear socket.

You can still lean/jump via keyboard; successful blinks act like tapping the jump key.

## External control via JSON socket

If your ML model or AutoHotKey script already interprets BrainLink data, you can stream the
resulting control signals straight into the game.

1. Start the game with the socket listener enabled (defaults to `127.0.0.1:4789`):

   ```bash
   python main.py --socket-input
   ```

   Use `--socket-host` / `--socket-port` to change the bind address.

2. From your pipeline, open a TCP connection to that address and send newline-delimited JSON
   messages such as:

   ```json
   {"lean": -0.35}
   {"jump": true}
   {"jump": false}
   ```

   - `lean` accepts values between `-1.0` (hard left) and `1.0` (hard right).
   - `jump` acts like pressing and releasing the jump key; short pulses are enough.
   - Include both fields in one message if you prefer: `{"lean": 0.1, "jump": true}`.
   - Optional `{"reset": true}` returns control to the keyboard baseline.

The socket layer stacks with the keyboard and blink input, so you can fall back to manual control at any time.

## Blink energy training + BrainLink bridge

1. **Derive an energy profile (one-time setup)**

   ```bash
   python tools/train_blink_energy.py \
       --datasets ~/Downloads/BME_Lab_BCI_training/bci_dataset_114-1 \
                 ~/Downloads/BME_Lab_BCI_training/bci_dataset_113-2 \
       --output assets/blink_energy_profile.json
   ```

   This reads each subject's `S*/3.txt` (containing 20-second open/20-second closed eye cycles), calculates the energy distribution of open and closed eyes, and outputs suggested energy thresholds. Results are written to `assets/blink_energy_profile.json`, which subsequent bridge programs and real-time detection will automatically read.

2. **Start the game's socket listener**

   ```bash
   python main.py --socket-input
   ```

3. **Execute BrainLink → Model → Game bridge script**

   ```bash
   python tools/brainlink_socket_bridge.py \
       --thinkgear-host 127.0.0.1 --thinkgear-port 13854 \
       --game-port 4789 \
       --profile assets/blink_energy_profile.json \
       --model-module your_ml_module
   ```

   - `--profile` points to the energy settings generated in the previous step, which drives `EnergyBlinkDetector` to read raw EEG (requires ThinkGear Connector to be started first).
   - `--model-module` is an optional Python module that needs to provide `predict(packet: dict) -> dict`, where you can load your focus/relaxation model and output `{"lean": …, "jump": …}`. If not specified, it defaults to using meditation value for lean, and blink is determined by energy detection.
   - If your model also needs to output JSON, you can directly return a dictionary from `predict`.

4. The bridge script converts each blink (brief energy drop) into a `{"jump": true}` JSON command sent to the game's socket. You can also use `packet["rawEeg"]` in your custom module to process features yourself.

## Direct Connection to BrainLink Using BrainLinkParser (Without ThinkGear Connector)

1. **Install requirements (one-time)**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Find BrainLink's serial port**: Use `ls /dev/cu.*` to find BrainLink's serial port (e.g., `/dev/cu.BrainLink_Lite`).

3. **Start game socket**:
   ```bash
   python main.py --socket-input
   ```

4. **Use `tools/brainlink_serial_bridge.py` to directly parse BrainLink's serial data and send to game**:
   ```bash
   python tools/brainlink_serial_bridge.py \
       --serial-port /dev/cu.BrainLink_Lite \
       --profile assets/blink_energy_profile.json \
       --game-port 4789 \
       --verbose \
       --model-module your_ml_module   # Optional if not available
   ```

   - Raw EEG data goes through `EnergyBlinkDetector` for energy spike detection → triggers jump.
   - `--model-module` can define `predict(packet: dict) -> dict`, returning fields like `{"lean": …}`; if not specified, defaults to using attention value for lean.
   - Without profile, it falls back to using `blinkStrength >= threshold` to detect blinks.

> If the bridge program shows `Connection refused`, it means you haven't started `python main.py --socket-input` yet; please start the game socket first, then start the bridge.

## Keyboard Controls

When the game is running, you can use the following keyboard controls:

- `A` / `←`: Lean left
- `D` / `→`: Lean right
- `Space` / `↑`: Jump

## Troubleshooting

### Connection Refused

If you see `Connection refused` error:

1. Confirm the game is started and using `--socket-input` parameter
2. Confirm the port number is correct (default 4789)
3. Check firewall settings

### Cannot Detect Blinks

1. Confirm BrainLink device is properly connected
2. Check if `blink_energy_profile.json` exists
3. Adjust `--blink-threshold` parameter

## Related Files

- [Project Main README](../README.md)
- [Game Control Integration Guide](../server_client/GAME_CONTROL_README.md)
- [BrainLink Usage Guide](../brainlink/README_USAGE.md)
