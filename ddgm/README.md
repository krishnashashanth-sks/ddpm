# DDGM - Denoising Diffusion Gamma Model

A U-Net based deep learning model that estimates Gamma distribution parameters from noisy images for image denoising using a custom Gamma Negative Log-Likelihood (NLL) loss.

## Overview

This project implements a specialized image denoising approach using a U-Net architecture that predicts Gamma distribution parameters (shape `k` and scale `θ`) for each pixel. The model leverages a custom Gamma NLL loss function to learn optimal parameter estimation from noisy observations.

## Project Structure

```
ddgm/
├── main.py           # Entry point for training and evaluation
├── model.py          # U-Net architecture for Gamma parameter estimation
├── losses.py         # Custom Gamma NLL loss function
├── dataset.py        # Data generation and preprocessing
└── README.md         # This file
```

## Key Components

### 1. Model Architecture (`model.py`)

**Function**: `build_unet_gamma_estimator()`

A U-Net encoder-decoder architecture with the following characteristics:

- **Input**: Single-channel grayscale images (default: 128×128×1)
- **Encoder**: Progressive downsampling with 3 levels (32, 64, 128 filters)
- **Bridge**: Central bottleneck layer (256 filters)
- **Decoder**: Progressive upsampling with skip connections
- **Output**: 2-channel output representing Gamma parameters:
  - Channel 0: Shape parameter `k` (via softplus activation)
  - Channel 1: Scale parameter `θ` (via softplus activation)

**Architecture Details**:
- Convolutional layers: 3×3 kernels with ReLU activation
- Pooling/Upsampling: 2×2 operations
- Skip connections: Concatenate encoder features with decoder layers
- Output activation: Softplus (ensures positive parameter values)

### 2. Loss Function (`losses.py`)

**Function**: `gamma_nll_loss()`

Implements the Gamma Negative Log-Likelihood loss:

```
Loss = (1 - k) * log(y_true) + (y_true / θ) + log(Γ(k)) + k * log(θ)
```

Where:
- `y_true`: Observed noisy pixel values
- `y_pred`: Model predictions [k, θ]
- `Γ(k)`: Gamma function of shape parameter

**Features**:
- Ensures numerical stability with epsilon clipping
- Prevents log(0) errors with safe maximum operations
- Broadcasts 1-channel predictions across spatial dimensions

### 3. Dataset (`dataset.py`)

Generates synthetic training data with the following specifications:

- **Batch Size**: 4
- **Num Samples**: 100
- **Image Size**: 128×128×1
- **Noise Model**: Gamma-distributed noise

**Data Generation Process**:
1. Creates clean baseline signal (constant 0.5)
2. Simulates Gamma-distributed noise with:
   - Shape parameter k = 2.0
   - Scale parameter θ = 0.2
3. Normalizes noisy images to [0, 1] range
4. Creates target labels (identical to input for NLL loss)

### 4. Training Script (`main.py`)

**Workflow**:

1. **Model Building**: Initializes U-Net with input shape (128, 128, 1)
2. **Compilation**: Uses Adam optimizer with custom Gamma NLL loss
3. **Training**: Fits model on synthetic data
4. **Prediction**: Generates Gamma parameter predictions
5. **Visualization**: Plots input image and predicted k, θ maps

**Configuration**:
- Input shape: 128×128×1
- Batch size: 4
- Epochs: 1 (configurable for longer training)
- Validation split: 20%

**Output**:
- Predicted k and θ parameter maps for test image
- Visualization showing:
  - Original noisy input
  - Predicted shape parameter (k) heatmap
  - Predicted scale parameter (θ) heatmap

## Installation

```bash
pip install tensorflow matplotlib numpy
```

## Usage

### Basic Training

```bash
python main.py
```

### Customization

**Modify dataset parameters** in `dataset.py`:
```python
batch_size = 8  # Change batch size
num_samples = 500  # Increase training samples
dummy_k_true = 3.0  # Adjust noise shape parameter
dummy_theta_true = 0.5  # Adjust noise scale parameter
```

**Adjust model architecture** in `model.py`:
```python
gamma_unet_model = build_unet_gamma_estimator(
    input_shape=(128, 128, 1),
    n_filters=64  # Increase model capacity
)
```

**Modify training** in `main.py`:
```python
history = gamma_unet_model.fit(
    X_train_noisy, Y_train_noisy_target,
    batch_size=batch_size,
    epochs=50,  # Increase training epochs
    validation_split=0.2
)
```

## Mathematical Background

### Gamma Distribution

The Gamma distribution with shape `k` and scale `θ` has PDF:

```
p(x; k, θ) = (1 / (Γ(k) * θ^k)) * x^(k-1) * exp(-x/θ)
```

### Negative Log-Likelihood

For a noisy observation `y`, the NLL is:

```
NLL = -log(p(y; k, θ)) = (1-k)*log(y) + y/θ + log(Γ(k)) + k*log(θ)
```

The model learns to minimize this loss, effectively learning optimal Gamma parameters that maximize the likelihood of observing the noisy pixels.

## Expected Output

When running `main.py`, you'll see:

1. **Model Summary**: U-Net architecture with layer details
2. **Training Logs**: Loss and validation loss per epoch
3. **Predictions**: Gamma parameters for the first sample
4. **Visualization**: 3-panel figure showing:
   - Noisy input image (grayscale)
   - Predicted k values (viridis colormap)
   - Predicted θ values (plasma colormap)

## Future Enhancements

- [ ] Real image datasets (CIFAR-10, MNIST, custom datasets)
- [ ] Multi-channel image support (RGB)
- [ ] Data augmentation techniques
- [ ] Model checkpointing and early stopping
- [ ] Quantitative evaluation metrics (PSNR, SSIM)
- [ ] Inference pipeline for denoising
- [ ] Batch normalization for improved convergence
- [ ] Learning rate scheduling

## Notes

- Current implementation uses synthetic Gamma-distributed data
- For production use, replace dummy data with real noisy images
- The model output represents pixel-wise Gamma parameters, not direct denoised pixels
- Consider using the predicted parameters for downstream denoising tasks

## References

- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Gamma Distribution: Standard probabilistic distribution commonly used in noise modeling
- Deep Learning for Image Denoising: Recent advances in CNN-based denoising methods
