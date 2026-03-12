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

```bash
# CCT — primary model (~3.7M params, transformer-based)
python src/train.py --model cct --epochs 200

# WideResNet-28-10 — baseline (~36M params, CNN-based)
python src/train.py --model wrn --epochs 200
```

### CLI Overrides

```bash
python src/train.py --model cct --epochs 300 --batch-size 128 --lr 0.05
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
| `src/dataset.py` | Dataset, augmentation (TrivialAugmentWide + CutOut), class-balanced sampler |
| `src/models/cct.py` | Compact Convolutional Transformer (CCT-7/3×1) |
| `src/models/wideresnet.py` | WideResNet-28-10 for CIFAR-style inputs |
| `src/train.py` | Training loop — MixUp/CutMix, cosine LR warmup, SWA, class-weighted loss |
| `src/inference.py` | Inference with TTA and multi-model ensemble |
| `src/utils.py` | Metrics, MixUp/CutMix, checkpointing |
| `configs/default.yaml` | All hyperparameters |

## Key Techniques

- **Class-balanced sampling + weighted loss** — handles imbalance (truck: 100 vs airplane: 5,000)
- **TrivialAugmentWide + CutOut + MixUp/CutMix** — robust augmentation
- **SWA** — flatter minima for better generalization under shift
- **TTA + Ensemble** — boosts macro F1 at inference
