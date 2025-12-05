# 114-1 BME LAB Final Project - Brain-Computer Interface Game Control System

> **Language Selection / 語言版本選擇**
> 
> - 🇺🇸 [English](Readme.md) ← Current version
> - 🇹🇼 [繁體中文 (Traditional Chinese)](Readme_chinese.md)
> 
> You can also click the 📝 icon next to the README title to view history, or use GitHub's branch/tag feature to switch between versions.

## Project Overview

This project implements a game control system based on EEG (electroencephalography) signals, using the CTNet (Convolution-Transformer Network) model for real-time classification of brain signals and converting classification results into game control commands. The system consists of three main modules:

1. **Balance Game** (`balance_game/`): A tightrope balance game supporting multiple input methods
2. **EEG Classifier** (`Classifier/`): Uses CTNet model for EEG signal classification (relaxed/focused)
3. **Real-time Server** (`server_client/`): Receives BIOPAC EEG data, performs real-time classification, and controls the game

## Project Structure

```
114-1_BME_LAB_Final_Project_G5/
├── balance_game/              # Balance game module
│   ├── assets/                # Game resources (images, fonts, etc.)
│   ├── balance_game/          # Core game code
│   │   ├── game.py            # Main game logic
│   │   ├── input.py           # Input handling
│   │   ├── blink_detector.py  # Blink detection
│   │   └── brainlink.py       # BrainLink integration
│   ├── tools/                 # Utility scripts
│   │   ├── brainlink_serial_bridge.py
│   │   ├── brainlink_socket_bridge.py
│   │   └── train_blink_energy.py
│   ├── main.py                # Game entry point
│   └── README.md              # Game usage instructions
│
├── Classifier/                # EEG classifier module
│   ├── bci_dataset_113-2/     # BCI dataset (35 subjects)
│   ├── Loso_C_heads_2_depth_6_0/  # Trained models (6 layers)
│   ├── Loso_C_heads_2_depth_8_0/  # Trained models (8 layers)
│   ├── loso.py                # CTNet model architecture and training
│   ├── inference.py           # Inference script
│   ├── inference_example.py  # Inference examples
│   ├── utils.py              # Utility functions
│   ├── plot_figures/         # Result visualization scripts
│   └── README.md             # Classifier documentation
│
└── server_client/             # Real-time server module
    ├── eeg_server_ctnet.py   # EEG server main program
    ├── inference.py          # Inference module
    ├── loso.py              # CTNet model
    ├── utils.py             # Utility functions
    ├── test_game_control.py # Game control testing
    ├── Loso_C_heads_2_depth_8_0/  # Model files
    └── GAME_CONTROL_README.md    # Game control documentation
```

## Installation

### 1. Environment Setup

```bash
# Create virtual environment (first time use)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r balance_game/requirements.txt
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn einops
```

### 2. Compile Game Module

```bash
python -m compileall balance_game
```

## Usage

### Quick Start: Control Game with EEG

#### Step 1: Start the Game

Start the game with socket input enabled in one terminal:

```bash
source .venv/bin/activate
cd balance_game
python main.py --socket-input --socket-port 4789
```

#### Step 2: Start EEG Classification Server

Start the classification server in another terminal:

```bash
source .venv/bin/activate
cd server_client
python eeg_server_ctnet.py
```

The server will:
- Listen for BIOPAC EEG data (default port 50007)
- Perform real-time classification using CTNet model
- Convert classification results to game control commands and send to the game

#### Control Logic

- **Relaxed State** → Lean left (`lean: -0.5`)
- **Focused State** → Lean right (`lean: 0.5`)
- **Blink** → Trigger jump (`jump: true`)

### Other Usage Methods

#### Direct Control with BrainLink

```bash
# Terminal 1: Start game
python main.py --socket-input --socket-port 4789

# Terminal 2: Start BrainLink bridge
python tools/brainlink_serial_bridge.py \
    --serial-port /dev/cu.BrainLink_Lite \
    --profile assets/blink_energy_profile.json \
    --game-port 4789 \
    --verbose
```

#### Keyboard Control

```bash
python main.py
```

Controls:
- `A` / `←`: Lean left
- `D` / `→`: Lean right
- `Space` / `↑`: Jump

For detailed instructions, please refer to each module's README:
- [Game Usage Instructions](balance_game/README.md)
- [Game Control Integration Guide](server_client/GAME_CONTROL_README.md)

## Classifier Results

This project uses the CTNet (Convolution-Transformer Network) model for EEG signal classification, employing Leave-One-Subject-Out (LOSO) cross-validation on a dataset of 35 subjects.

### Model Architecture

![CTNet Architecture](Classifier/architecture.png)

### Classification Results

#### Best Results

![Best Classification Results](Classifier/bci_results_best.png)

#### Comparison of Results with Different Training Epochs

**Base Model (100 epochs)**

![Base Model Results](Classifier/bci_results_data_base_e100.png)

**Extended Model (1000 epochs)**

![Extended Model Results](Classifier/bci_results_data_e1000.png)

### Model Configuration

The project includes two main model configurations:

- **Loso_C_heads_2_depth_6_0**: 6-layer Transformer encoder with 2 attention heads
- **Loso_C_heads_2_depth_8_0**: 8-layer Transformer encoder with 2 attention heads (used for real-time inference)

### Training and Evaluation

For detailed training and evaluation methods, please refer to:
- [Classifier README](Classifier/README.md)
- `Classifier/loso.py`: Model training script
- `Classifier/inference.py`: Inference script

## Technical Details

### CTNet Model

This project is based on the following paper:

**Citation:**
```
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network 
for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). 
https://doi.org/10.1038/s41598-024-71118-7
```

### Data Processing

- **Sampling Rate**: 500 Hz (BIOPAC)
- **Window Size**: 1000 samples (approximately 2 seconds)
- **Stride**: 300 samples (approximately 0.6 seconds)
- **Channels**: 22-channel EEG

### Real-time Inference

- Uses sliding windows for continuous data processing
- Supports smoothing to reduce prediction fluctuations
- Blink detection based on amplitude analysis of raw EEG signals

## Troubleshooting

### Connection Issues

If you encounter `Connection refused` errors:

1. Ensure the game is started with the `--socket-input` parameter
2. Verify the port number is correct (default 4789)
3. Use `test_game_control.py` to test the connection

### Model Loading Issues

1. Ensure model files exist in `server_client/Loso_C_heads_2_depth_8_0/` directory
2. Check if model path configuration is correct

### Data Format Issues

1. Ensure BIOPAC data format matches expectations
2. Check if sampling rate is 500 Hz

## Development Team

114-1 BME LAB Final Project Group 5

## License

Please refer to the license files in each module.

