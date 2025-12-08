# CTNet EEG Classifier

## Project Overview

This module implements an EEG signal classification system based on CTNet (Convolution-Transformer Network) to distinguish between "focused" and "relaxed" brain states. It employs Leave-One-Subject-Out (LOSO) cross-validation for training and evaluation on a dataset of 35 subjects.

## Model Architecture

CTNet combines the advantages of Convolutional Neural Networks (CNN) and Transformers for processing EEG time-series data:

![CTNet Architecture](architecture.png)

### Architecture Features

1. **Patch Embedding CNN**: Converts raw EEG signals into patch embeddings suitable for Transformer processing
   - Temporal Convolution
   - Depth-wise Convolution
   - Spatial Convolution
   - Average Pooling

2. **Transformer Encoder**: Captures long-range temporal dependencies
   - Multi-Head Self-Attention mechanism
   - Feed-Forward Network
   - Layer Normalization

3. **Classification Head**: Final classification layer

### Model Configuration

This project includes two main model configurations:

- **Loso_C_heads_2_depth_6_0**: 6-layer Transformer encoder with 2 attention heads
- **Loso_C_heads_2_depth_8_0**: 8-layer Transformer encoder with 2 attention heads (recommended for real-time inference)

## Dataset

### Data Format

- **Dataset Path**: `bci_dataset_113-2/`
- **Number of Subjects**: 35 (S01-S35)
- **Data Structure**:
  ```
  bci_dataset_113-2/
  ├── S01/
  │   ├── 1.txt  # Class 1: Focused
  │   └── 2.txt  # Class 2: Relaxed
  ├── S02/
  │   ├── 1.txt
  │   └── 2.txt
  └── ...
  ```

### Data Specifications

- **Channels**: 1 channel (single-channel EEG)
- **Sampling Rate**: 500 Hz
- **Window Size**: 1000 samples (approximately 2 seconds)
- **Classes**: 2 classes (Focused vs Relaxed)

## Training

### Training Method

Uses **Leave-One-Subject-Out (LOSO)** cross-validation:
- Each training iteration uses data from 34 subjects as the training set
- The remaining 1 subject's data is used as the test set
- Repeats 35 times, with each subject serving as the test set in turn
- Results in 35 models (`model_1.pth` to `model_35.pth`)

### Training Parameters

Default configuration (see `loso.py`):

```python
EPOCHS = 1000              # Number of training epochs
HEADS = 2                  # Number of Transformer attention heads
EMB_DIM = 16               # Embedding dimension
DEPTH = 8                  # Transformer encoder depth
BATCH_SIZE = 512           # Batch size
LEARNING_RATE = 0.001      # Learning rate
N_AUG = 3                  # Data augmentation multiplier
N_SEG = 50                 # Segmentation times (S&R data augmentation)
VALIDATE_RATIO = 0.1       # Validation set ratio
DROPOUT_RATE = 0.25        # Dropout rate (LOSO mode)
```

### Data Augmentation

Uses **Segmentation and Reconstruction (S&R)** method:
- Segments and reconstructs original sequences
- Increases training data diversity
- Improves model generalization

### Running Training

```bash
cd Classifier
python loso.py
```

The training process will:
1. Automatically perform LOSO cross-validation
2. Save the best model for each subject
3. Generate result Excel files:
   - `result_metric.xlsx`: Evaluation metrics for each subject
   - `process_train.xlsx`: Training process records
   - `pred_true.xlsx`: Prediction results and true labels
4. Automatically plot result charts

## Inference

### Single Model Inference

Use the `CTNetInference` class for single model inference:

```python
from inference import CTNetInference

# Initialize inferencer
inferencer = CTNetInference(
    model_path="Loso_C_heads_2_depth_8_0/model_1.pth",
    dataset_type='C',
    heads=2, emb_size=16, depth=8,
    eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
    eeg1_pooling_size1=8, eeg1_pooling_size2=8,
    eeg1_dropout_rate=0.25, flatten_eeg1=240
)

# Inference from txt file
prediction, probability = inferencer.predict_from_txt("bci_dataset_113-2/S01/1.txt")
print(f"Predicted class: {prediction}, Probability: {probability}")
```

### Ensemble Inference (Recommended)

Use the `CTNetEnsembleInference` class for multi-model ensemble inference, which typically provides more stable and accurate results:

```python
from inference import CTNetEnsembleInference

# Initialize Ensemble inferencer (automatically loads all models)
inferencer = CTNetEnsembleInference(
    model_dir="Loso_C_heads_2_depth_8_0",
    dataset_type='C',
    heads=2, emb_size=16, depth=8,
    eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
    eeg1_pooling_size1=8, eeg1_pooling_size2=8,
    eeg1_dropout_rate=0.25, flatten_eeg1=240
)

# Inference from txt file (uses average prediction from all models)
prediction, probability = inferencer.predict_from_txt("bci_dataset_113-2/S01/1.txt")
print(f"Predicted class: {prediction}, Probability: {probability}")
```

### Real-time Inference

Use sliding windows for real-time inference on continuous data streams:

```python
import numpy as np
from inference import CTNetEnsembleInference

# Initialize inferencer
inferencer = CTNetEnsembleInference(
    model_dir="Loso_C_heads_2_depth_8_0",
    dataset_type='C',
    heads=2, emb_size=16, depth=8,
    eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
    eeg1_pooling_size1=8, eeg1_pooling_size2=8,
    eeg1_dropout_rate=0.25, flatten_eeg1=240
)

# Define data stream generator (simulates real-time reception)
def data_stream_generator(data, chunk_size=200):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

# Read data
data = np.loadtxt("bci_dataset_113-2/S01/1.txt", dtype=np.float32)

# Real-time inference
class_names = ['Relaxed', 'Focused']
for result in inferencer.predict_realtime(
    data_stream_generator(data, chunk_size=200),
    window_size=1000,      # Window size (2 seconds)
    stride=500,            # Sliding stride (1 second)
    smoothing_window=5     # Smoothing window
):
    pred = result['prediction']
    prob = result['probability']
    print(f"Prediction: {class_names[pred]}, Probability: {prob}")
```

### Command Line Inference

```bash
# Inference from txt file
python inference.py \
    --model_path Loso_C_heads_2_depth_8_0/model_1.pth \
    --txt_file bci_dataset_113-2/S01/1.txt \
    --dataset_type C \
    --heads 2 --emb_size 16 --depth 8

# Batch inference (from directory)
python inference.py \
    --model_path Loso_C_heads_2_depth_8_0/model_1.pth \
    --data_dir ./test_data/ \
    --dataset_type C
```

### Inference Examples

See `inference_example.py` for more usage examples:

```bash
python inference_example.py
```

## Results

### Classification Performance

#### Best Results

![Best Classification Results](bci_results_best.png)

#### Comparison of Results with Different Training Configurations

**Base Model (100 epochs)**

![Base Model Results](bci_results_data_base_e100.png)

**Extended Model (1000 epochs)**

![Extended Model Results](bci_results_data_e1000.png)

### Evaluation Metrics

Model evaluation uses the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Cohen's Kappa**

Results are saved in `result_metric.xlsx`, containing detailed metrics for each subject.

### Result Visualization

Use scripts in the `plot_figures/` directory to generate visualization charts:

- `confusion_matrix.py`: Confusion matrix
- `depth.py`: Comparison of different depth configurations
- `heads.py`: Comparison of different attention head numbers
- `length.py`: Comparison of different window lengths

## File Structure

```
Classifier/
├── bci_dataset_113-2/          # Dataset (35 subjects)
│   ├── S01/
│   │   ├── 1.txt               # Focused state data
│   │   └── 2.txt               # Relaxed state data
│   └── ...
├── Loso_C_heads_2_depth_6_0/    # 6-layer model results
│   ├── model_1.pth ~ model_35.pth
│   ├── result_metric.xlsx
│   ├── process_train.xlsx
│   └── pred_true.xlsx
├── Loso_C_heads_2_depth_8_0/   # 8-layer model results (recommended)
│   ├── model_1.pth ~ model_35.pth
│   ├── result_metric.xlsx
│   ├── process_train.xlsx
│   └── pred_true.xlsx
├── loso.py                     # Training script
├── inference.py                # Inference script
├── inference_example.py        # Inference examples
├── utils.py                    # Utility functions
├── plot_figures/              # Result visualization scripts
│   ├── confusion_matrix.py
│   ├── depth.py
│   ├── heads.py
│   └── length.py
├── architecture.png            # Model architecture diagram
├── bci_results_best.png        # Best results chart
├── bci_results_data_base_e100.png
└── bci_results_data_e1000.png
```

## Dependencies

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn einops openpyxl
```

## Usage Recommendations

1. **Training Phase**:
   - Use `loso.py` for LOSO cross-validation training
   - Adjust `batch_size` according to GPU memory
   - Adjust `N_AUG` and `N_SEG` to control data augmentation intensity

2. **Inference Phase**:
   - **Recommend using Ensemble inference** (`CTNetEnsembleInference`) for better stability
   - For real-time applications, use the `predict_realtime()` method
   - Adjust `smoothing_window` parameter to balance response speed and stability

3. **Model Selection**:
   - 8-layer model (`Loso_C_heads_2_depth_8_0`) generally performs better, recommended for real-time inference
   - 6-layer model (`Loso_C_heads_2_depth_6_0`) has lower computational cost, suitable for resource-constrained environments

## Citation

This project is based on the following paper:

```
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network 
for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). 
https://doi.org/10.1038/s41598-024-71118-7
```

## Notes

1. **CUDA Setup**: Ensure CUDA environment is properly configured; model training and inference require GPU support
2. **Data Format**: Ensure input data format matches expectations (single channel, 500 Hz sampling rate)
3. **Model Path**: Ensure model file paths are correct during inference
4. **Memory Management**: Ensemble inference loads multiple models; pay attention to GPU memory usage

## Troubleshooting

### Model Loading Failure

- Verify model file paths are correct
- Check if model parameter configuration matches training settings

### CUDA Out of Memory

- Reduce `batch_size`
- Use single model inference instead of Ensemble
- Reduce `smoothing_window` size

### Unstable Inference Results

- Increase `smoothing_window` parameter
- Use Ensemble inference instead of single model
- Check input data quality


