# Satellite Image Classification with Digital Image Processing

This project classifies satellite imagery using deep learning models enhanced with digital image processing techniques, speciallizing for two types of distortion: haze and low light. It supports both pure spatial and spatial with frequency architectures.

## Project Structure

```
src/
├── dip/
│   └── dip_modules.py          # Image preprocessing methods
├── models/
│   └── model.py                # Models architectures
├── scripts/
│   ├── train.py                # Train pure spatial classifier
│   ├── dual_train.py           # Train dual-branch (spatial with frequency) classifier
│   ├── test.py                 # Evaluate  pure spatial
│   ├── dual_test.py            # Evaluate dual-branch model
│   └── finetune.py             # Fine-tuning script
└── utils/
    └── generate_example.py     # Demo image preprocessing effects
```

## Features

**Image Preprocessing Methods:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Unsharp masking
- Laplacian sharpening
- Dark channel prior 


## Requirements

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy pillow scikit-learn tqdm
```

## Dataset Setup

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── validation/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Usage

### Training spatial-only Model

```bash
cd src/scripts
python train.py \
  --train_dir ../../data/train \
  --val_dir ../../data/validation \
  --test_dir ../../data/test \
  --img_size 64 \
  --batch_size 128 \
  --epochs 50 \
  --lr 3e-4 \
  --out_model best_resnet18.pth
```

### Training Dual-Branch Model

The dual-branch model uses both RGB images and their DCT (Discrete Cosine Transform) representations:

```bash
cd src/scripts
python dual_train.py \
  --train_dir ../../data/train \
  --val_dir ../../data/validation \
  --test_dir ../../data/test \
  --img_size 64 \
  --batch_size 128 \
  --epochs 50 \
  --lr 4e-4 \
  --out_model best_dual_branch.pth
```

### Fine-tuning Models

Fine-tune a pretrained model on new data by only updating the classification head:

```bash
cd src/scripts
python finetune.py \
  --train_dir ../../data/train \
  --val_dir ../../data/validation \
  --base_model best_resnet18.pth \
  --class_order "beach,buildings,forest,harbor,freeway" \
  --img_size 64 \
  --batch_size 128 \
  --epochs 10 \
  --lr 1e-4 \
  --out_model finetuned_resnet18.pth
```


### Testing Models

spatial-only model:
```bash
python test.py --test_dir ../../data/test --model_path best_resnet18.pth --algo USM
```

Dual-branch model:
```bash
python dual_test.py --test_dir ../../data/test --model_path best_dual_branch.pth --algo USM
```

### Preprocessing Demo


```bash
cd src/utils
python generate_example.py
```

