# Score-based SDE

This directory contains an implementation of score-based generative modeling using stochastic differential equations (SDEs). The code is organized for training, evaluation/inference, and model definition.

## Contents

- `dataset.py` - Dataset loading and preprocessing utilities.
- `layers.py` - Neural network building blocks used by the model (e.g., convolutions, residual blocks, normalization, etc.).
- `model.py` - Score network (the neural network that predicts the score) and model utilities.
- `losses.py` - Loss functions used to train the score network (e.g., denoising score matching, weighted losses for time-dependent noise levels).
- `train.py` - Training script to train the score model on a dataset.
- `inference.py` - Sampling / inference utilities that run reverse SDE or probability flow ODE to generate samples from the learned score model.
- `main.py` - Top-level entrypoint that ties together dataset, model, training, and inference (may provide CLI options).

## Requirements

- Python 3.8+
- PyTorch
- torchvision (optional, for image datasets)
- numpy
- tqdm

Install with pip:

```bash
pip install torch torchvision numpy tqdm
```

(Adjust the torch installation command for your CUDA version — see https://pytorch.org/.)

## Quickstart

1. Prepare your dataset and configure dataset paths in `dataset.py` or pass them via CLI (if supported).

2. Train the model:

```bash
python score-based-sde/train.py
```

Typical arguments you may want to supply (check `train.py` for exact names):

- `--dataset` or `--data-dir` — path to the training data
- `--batch-size` — batch size
- `--epochs` — number of training epochs
- `--lr` — learning rate
- `--save-dir` — directory to save checkpoints

3. Sample / run inference:

```bash
python score-based-sde/inference.py
```

Typical inference options (check `inference.py`):

- `--checkpoint` — path to a trained model checkpoint
- `--num-samples` — number of samples to generate
- `--output-dir` — directory to save generated images

## Files and functions (overview)

- `dataset.py`
  - Exposes a Dataset / DataLoader for training and validation.
  - Handles transforms and normalization used by the model.

- `layers.py`
  - Contains building blocks such as residual blocks, attention, up/down sampling layers, and any time/score embedding utilities.

- `model.py`
  - Implements the score network architecture and helper functions for saving/loading state dicts.

- `losses.py`
  - Implements time-dependent denoising score matching losses or other objectives used to train the score network.

- `train.py`
  - Contains the training loop, checkpoint saving, logging, and evaluation hooks.

- `inference.py`
  - Implements sampling schedules (reverse SDE, predictor-corrector, or ODE solvers) and utilities to convert model outputs to images.

- `main.py`
  - Useful top-level script that may provide a unified CLI for training and sampling. Check this file for argument names and usage examples.

## Tips

- Check the docstrings and argument parsers in `train.py` and `inference.py` for the exact CLI flags.
- Use smaller models and datasets for quick experiments.
- Keep an eye on numerical stability when integrating SDEs — adjust step sizes or solvers as needed.

## Contact / References

- Score-based generative modeling: Song & Ermon (NIPS/ICML papers)
- For questions or bug reports, open an issue in the repository.
