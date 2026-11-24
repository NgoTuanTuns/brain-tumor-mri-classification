# Brain Tumor MRI Classification (PyTorch)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?style=flat&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

End-to-end deep learning experiment for 4-class brain tumor classification on MRI scans (`glioma`, `meningioma`, `notumor`, `pituitary`). The project investigates overfitting in a custom baseline CNN through a controlled regularization ablation study (**Dropout**, **L2 Weight Decay**, **Data Augmentation**, **Early Stopping**) and benchmarks transfer learning via **ResNet18** fine-tuning.

## Dataset & Leakage Prevention

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle — Masoud Nickparvar), 7,200 images across 4 balanced classes.
- **Data Split (`SEED = 42`):**
  - `Training` (5,600 images) → stratified 80/20 split into **4,480 Train** (1,120/class) and **1,120 Validation** (280/class).
  - `Testing` → kept strictly untouched as an independent **1,600 Test** set (400/class).
- **Preprocessing:** Images are resized to `128×128` RGB and normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Leakage Control:** Data augmentation (`RandomHorizontalFlip`, `RandomRotation(±15°)`, mild `ColorJitter`) is applied **only** to the training subset; validation and test loaders use a deterministic evaluation transform.

## Experiments & Results

All 5 configurations were trained on a Google Colab T4 GPU and evaluated on the independent 1,600-image test set (see full curves, confusion matrices, and qualitative error analysis in [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb)):

| Model | Regularization / Setup | Test Loss | Test Acc | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| `BaselineCNN` | 3-block CNN, no regularization (25 epochs) | 2.7006 | 89.81% | 0.8965 |
| `Dropout + L2` | `Dropout2d(0.25)`, `Dropout(0.5)`, `weight_decay=1e-4` | 1.0215 | 90.50% | 0.9031 |
| `Augmentation` | `BaselineCNN` + train-only augmentation | 1.1610 | 92.94% | 0.9280 |
| `Combined + ES` | Dropout + L2 + Augmentation + Early Stopping (`best_weights`) | 0.5544 | 92.38% | 0.9219 |
| **`ResNet18 (Fine-tune)`** | **ImageNet pretrained, `Dropout(0.3)`, `lr=1e-4`, Early Stopping** | **0.3289** | **94.44%** | **0.9432** |

### Per-Class Test Metrics (`ResNet18 Fine-tune`)

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| `glioma` | 0.9731 | 0.8150 | 0.8871 | 400 |
| `meningioma` | 0.8833 | 0.9650 | 0.9223 | 400 |
| `notumor` | 0.9412 | 1.0000 | 0.9697 | 400 |
| `pituitary` | 0.9901 | 0.9975 | 0.9938 | 400 |

### Key Takeaways

- **Overfitting diagnosis:** `BaselineCNN` memorized the training set (`Train Acc = 100%`, `Train Loss ≈ 0.0000`), resulting in poorly calibrated predictions (`Test Loss = 2.7006`).
- **Regularization impact:** Combining Dropout, L2 weight decay, augmentation, and Early Stopping reduced Test Loss by **79.5%** (`2.7006 → 0.5544`).
- **Transfer learning gain:** Full fine-tuning of `ResNet18` (`11.18M` trainable parameters, early-stopped at epoch 12 restoring epoch 7 weights) further lowered Test Loss to **`0.3289`** (**-87.8%** vs baseline) and reached **`94.44%`** accuracy / **`0.9432`** Macro F1.
- **Error analysis:** `notumor` (`100.0%` recall) and `pituitary` (`99.75%` recall) are separated cleanly, while `glioma` (`81.5%` recall) accounts for most remaining errors (frequently confused with `meningioma`).

## Repository Structure

```text
.
├── configs/
│   └── experiments.yaml          # Hyperparameters and ablation configs for all 5 models
├── notebooks/
│   └── experiments.ipynb         # Full Colab T4 experiment run with plots & error analysis
├── src/mri_classifier/
│   ├── __init__.py
│   ├── data.py                   # Dataset download, stratified split, and DataLoader builders
│   ├── models.py                 # BaselineCNN, RegularizedCNN, and ResNet18 factory
│   ├── train.py                  # Config-driven CLI trainer with seeding and Early Stopping
│   └── evaluate.py               # Test evaluation (Loss, Acc, Macro F1, Confusion Matrix)
├── pyproject.toml                # Project dependencies managed with uv
├── uv.lock
└── README.md
```

## Getting Started

### Install dependencies

```bash
uv sync
```

### Run experiments from CLI

```bash
# Run a single experiment (e.g., resnet18_finetune, combined_es, baseline)
PYTHONPATH=src uv run python -m mri_classifier.train --config configs/experiments.yaml --exp resnet18_finetune

# Run all 5 configurations sequentially
PYTHONPATH=src uv run python -m mri_classifier.train --config configs/experiments.yaml --exp all
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
uv run python -m mri_classifier.train --config configs/experiments.yaml --exp resnet18_finetune
```

### View notebook results

Open [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb) directly on GitHub or in Google Colab (T4 GPU recommended for full retraining).
