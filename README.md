# ShiftGuard10

Submission for EE708 Project — 10-class robust image classification on 32×32 RGB images.

**Goal**: Maximize **Macro F1** across 10 classes under extreme class imbalance and distribution shift (train ≠ test).

## Overview

The repository features an "All-in-One Training + Inference" script (`notebook.py`, v3). It replaces the previous multi-phase pipeline with a simplified, highly effective single-script approach that trains from scratch and generates `submission.csv`.

## Architecture & Methods in Detail

To combat extreme class imbalances and testing-time distribution shifts without external data or pre-trained internet models, the challenge pipeline employs several customized methods:

### 1. Model Architecture
- **WideResNet-28-10 (WRN)**: A high-capacity Residual Network variant specifically constructed for 32x32 image inputs natively. 
- With a depth of 28 layers and a widening factor of 10, it features approximately **~36 Million parameters**. Its wider blocks and Dropout layers (p=0.3) provide significant representational capacity whilst regularizing against overfitting.

### 2. Imbalance Handling
- **Sqrt-Inverse Frequency Sampling**: A Weighted Random Sampler handles training loop stochasticity by heavily oversampling the minority (tail) classes. Unlike standard inverse frequency, weights are calculated as `1.0 / sqrt(class_counts)`, which balances uniformity without entirely neglecting the head classes.
- **Balanced Softmax Loss**: The network tackles the long-tailed class distribution through a specialized Balanced Softmax Loss. Logits are actively shifted by `log(class_prior)` during the forward pass, neutralizing the classifier's inherent bias towards majority class predictions. 

### 3. Advanced Data Augmentations
- **Train Transforms**: `AutoAugment` (CIFAR-10 Policy) combined with `Cutout` (16x16 pixel masking), Random Cropping, and Horizontal Flipping to ensure extensive spatial and color invariances.
- **MixUp & CutMix Regularization**: Employed stochastically (50% probability) within the training loop with `alpha=1.0` to interpolate images and labels directly. This forces the model to learn smoother decision boundaries and linearly robust features between classes.

### 4. Optimization & Convergence
- **Optimization Strategy**: Trained with SGD (Stochastic Gradient Descent) featuring Nesterov Momentum (0.9) and high weight decay (`5e-4`). 
- **Cosine Annealing with Warmup**: A tailored Lambda LR scheduler initiates with 5 warmup epochs to gracefully stabilize initial gradients, organically decaying over the total epochs using a cosine curve.
- **Stochastic Weight Averaging (SWA)**: After standard convergence (Epoch 360), SWA begins caching weight states alongside a flat `SWALR`. Final batch-norm statistics are updated natively, flattening out the local minima curves to drastically improve underlying test generalization.

### 5. Multi-Ensemble Inference
- **3-Seed Ensembling**: Models are fully trained from scratch across three distinct pseudo-random seeds (42, 137, 7). The Softmax probability distributions are then numerically averaged during inference.
- **Aggressive Test-Time Augmentation (TTA)**: During inference, for every single image, the network generates predictions for **1 clean view + 30 augmented views** per seed model. TTA incorporates dynamic Random Rotations (10 degrees), Crop, Flip, and Color Jittering to completely stabilize shifting prediction confidence scores.

---

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision numpy scikit-learn pillow
```

> For GPU training, install PyTorch with CUDA (example for CUDA 11.8):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Dataset

Ensure the competition data is available on your machine (e.g., from Kaggle). You can specify the path with `--data-root` when running the notebook. By default, it looks for common paths or the `shift-guard-10-robust-image-classification-challenge/` directory in the current working directory.

## Usage

The entire pipeline is driven by `notebook.py`. 

### Quick Debug / Smoke Test
Run a fast smoke test (2 epochs, 1 seed, 2 TTA views) to ensure everything works:
```bash
python notebook.py --debug
```

### Full Training & Inference
Run the complete pipeline (3 seeds × 450 epochs + 30 TTA views). This will generate `submission.csv` at the end:
```bash
python notebook.py --data-root /path/to/data
```

### Advanced Training Configuration
You can customize hyperparameters to adjust training duration or ensemble size:
```bash
# Single seed training (faster, equivalent to v2)
python notebook.py --seeds 42

# Custom epochs and TTA views
python notebook.py --epochs 300 --tta 20

# Run on a specific GPU
python notebook.py --gpu 1
```

### Inference Only
If you already have trained checkpoints (saved by default in the specified `--checkpoint-dir`), you can skip training and just generate predictions:
```bash
python notebook.py --inference-only
```

## Checkpoints
Checkpoints are saved automatically after each epoch and support resuming if training is interrupted. When using the ensemble setting with multiple seeds, models are saved separately as `wrn_seed<SEED>.pth` within the checkpoint directory.

## Output
Predictions are saved directly to `submission.csv` containing `id` and `label` columns, ready for competition submission.
