# ShiftGuard10

Submission for EE708 Project — 10-class robust image classification on 32×32 RGB images.

**Goal**: Maximize **Macro F1** across 10 classes under distribution shift (train ≠ test).

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision pyyaml scikit-learn pandas numpy tqdm pillow
```

> For GPU training, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Dataset

Place the competition data so the structure looks like:

```
Shiftguard10/
├── shift-guard-10-robust-image-classification-challenge/
│   ├── classes.txt
│   ├── train_labels.csv
│   ├── sample_submission.csv
│   ├── train_images/   (29,400 PNGs)
│   └── test_images/    (7,600 PNGs)
├── src/
├── configs/
└── README.md
```

## Training

### Quick Debug (CPU, ~2 min)

```bash
python src/train.py --model cct --debug
```

### Full Training (GPU)

The default configuration is heavily optimized for Long-Tailed Learning using **Deferred Re-Weighting (DRW)** and **LDAM Loss** trained from scratch for **300 epochs**.

```bash
# Train the primary model (CCT) using the tuned configuration (300 epochs)
python src/train.py --config configs/default.yaml
```

To train the WideResNet baseline:
```bash
python src/train.py --config configs/default.yaml --model wrn
```

### CLI Overrides

```bash
# Override epochs or batch size if necessary
python src/train.py --model cct --epochs 350 --batch-size 128 --lr 0.05
python src/train.py --model cct --resume checkpoints/best_cct.pth
```

Checkpoints are saved to `checkpoints/`.

## Inference

```bash
# Single model
python src/inference.py --checkpoint checkpoints/best_cct.pth

# With Test-Time Augmentation (5 views)
python src/inference.py --checkpoint checkpoints/best_cct.pth --tta 5

# Ensemble CCT + WRN with TTA
python src/inference.py --checkpoint checkpoints/best_cct.pth checkpoints/best_wrn.pth --tta 5
```

Output: `submission.csv` (7,600 rows with `id,label`).

## Project Structure

| File | Description |
|------|-------------|
| `src/dataset.py` | Dataset, augmentation (TrivialAugmentWide, CutOut, TTA with Rotation/Jitter) |
| `src/models/cct.py` | Compact Convolutional Transformer (CCT-7/3×1) |
| `src/models/wideresnet.py` | WideResNet-28-10 for CIFAR-style inputs |
| `src/loss.py` | implementation of **LDAM Loss** and **Class Balanced Loss** |
| `src/train.py` | Training loop — MixUp/CutMix, SWA, **DRW (Deferred Re-Weighting)** |
| `src/inference.py` | Inference with TTA and multi-model ensemble |
| `src/utils.py` | Metrics, MixUp/CutMix, checkpointing |
| `configs/default.yaml` | All hyperparameters |

## Key Techniques

To strictly abide by the competition rules (no pretrained models, no external data), our pipeline utilizes:
- **Deferred Re-Weighting (DRW)** — trains on the raw skewed distribution to learn high-quality features, then switches to tuning the decision boundary for classes with fewer samples in the late stage.
- **LDAM Loss** — mathematically forces a wider margin for minority classes (truck, ship), drastically improving recall without overfitting.
- **MixUp / CutMix Disable Switch** — turns off interpolation during the DRW phase to ensure strict decision boundary tuning.
- **Enhanced TTA + Ensemble** — combines multi-checkpoint probabilities with Random Rotation, Flip, and Color Jitter to robustly predict shifts.
- **SWA** — ensures flatter loss minima for overall generalization.
