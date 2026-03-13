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

We use a robust 3-Phase **Supervised Contrastive Learning (SupCon)** approach to maximize F1 under extreme class imbalance, trained completely from scratch.

### Supported Architectures
You can use `--model` or `--backbone` with: `cct`, `wrn`, `resnet50`, `convnext`, `effnet`.

### Quick Debug (CPU, ~2 min)
```bash
python src/supcon.py --backbone resnet50 --debug
python src/train.py --model resnet50 --debug
```

### Full Training (GPU)

**Phase 1: SupCon Pretraining**
Learn highly invariant, clustered class features without the linear classifier bias.
```bash
python src/supcon.py --backbone resnet50 --balanced-sampling --epochs 300 --gpu 0
```
*(This saves the backbone state to `checkpoints/supcon_resnet50_epoch300.pth`)*

**Phase 2: Linear Probing**
Freeze the backbone and warm up only the classification head safely.
```bash
python src/train.py --config configs/default.yaml --model resnet50 \
  --pretrained-backbone checkpoints/supcon_resnet50_epoch300.pth \
  --linear-probe --epochs 20 --gpu 0
```

**Phase 3: End-to-End Fine-Tuning**
Unfreeze the entire network and fine-tune with a small learning rate.
```bash
python src/train.py --config configs/default.yaml --model resnet50 \
  --pretrained-backbone checkpoints/supcon_resnet50_epoch300.pth \
  --epochs 100 --lr 0.01 --gpu 0
```

### Multi-GPU Execution
To train multiple architectures simultaneously on different GPUs:
```bash
# Terminal 1: Train ConvNeXt on GPU 0
python src/supcon.py --backbone convnext --balanced-sampling --gpu 0

# Terminal 2: Train EfficientNet on GPU 1
python src/supcon.py --backbone effnet --balanced-sampling --gpu 1
```

Checkpoints are saved to `checkpoints/`.

## Inference

```bash
# Single model inference (runs on GPU 0 by default)
python src/inference.py --checkpoint checkpoints/best_cct.pth

# With Test-Time Augmentation (5 views) on specific GPU
python src/inference.py --checkpoint checkpoints/best_cct.pth --tta 5 --gpu 1

# Ensemble CCT + WRN with TTA
python src/inference.py --checkpoint checkpoints/best_cct.pth checkpoints/best_wrn.pth --tta 5
```

Output: `submission.csv` (7,600 rows with `id,label`).

## Project Structure

| File | Description |
|------|-------------|
| `src/dataset.py` | Dataset, augmentation (SupCon transforms, TrivialAugment, TTA) |
| `src/models/cct.py` | Compact Convolutional Transformer (CCT-7/3×1) |
| `src/models/wideresnet.py` | WideResNet-28-10 for CIFAR-style inputs |
| `src/models/torchvision_models.py` | ResNet50, ConvNeXt, EffNet natively adapted to 32x32 |
| `src/supcon.py` | Supervised Contrastive Pretraining phase |
| `src/train.py` | Fine-tuning loop — supports `--linear-probe` |
| `src/inference.py` | Inference with TTA and multi-model ensemble |
| `src/utils.py` | Metrics, MixUp/CutMix, checkpointing |
| `configs/default.yaml` | All hyperparameters |

## Key Techniques

To strictly abide by the competition rules (no pretrained models, no external data), our pipeline utilizes:
- **Supervised Contrastive Learning (SupCon)** — pulls all images of the same class together in representation space to robustly cluster minority classes *before* the linear classifier is initialized.
- **Frozen Linear Probing** — protects the pretrained features from destruction by the randomly initialized dense head, acting as a phase-break.
- **Architectural Diversity Ensembling** — leverages Transformers, Wide ResNets, standard ResNets, and ConvNeXts (custom adapted for 32x32 image structures) to ensure powerful generalization against the hidden distribution shift.
- **Enhanced TTA + SWA** — combines multi-checkpoint outputs with Random Rotation, Flip, Color Jitter, and Weight Averaging for ultimate test-time stability.
