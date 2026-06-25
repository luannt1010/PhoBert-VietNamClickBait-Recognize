# Clickbait Detector

A Vietnamese headline classification project that detects whether a news headline is **clickbait** or **non-clickbait** using PhoBERT.

## 1. Project Objective

This project builds a binary text classification model for Vietnamese news headlines.

Label meaning:

```text
1 = clickbait
0 = non-clickbait
```

Main pipeline:

```text
Raw headline
→ Text preprocessing
→ Vietnamese word segmentation with PyVi
→ PhoBERT tokenization
→ PhoBERT encoder
→ Binary classifier
→ Clickbait / Non-clickbait prediction
```

## 2. Project Structure

```text
clickbait_detect_proj/
│
├── data/
│   └── raw/
│       ├── train_clickbait.csv
│       └── val_clickbait.csv
│
├── artifacts/
│   ├── models/
│   │   ├── best.pth
│   │   └── last.pth
│   │
│   └── reports/
│       └── history.json
│
├── src/
│   └── clickbait_detector/
│       ├── __init__.py
│       ├── clickbait_dataset.py
│       ├── preprocessing.py
│       ├── net.py
│       ├── utils.py
│       └── inference.py
│
├── train.py
├── pyproject.toml
└── README.md
```

## 3. Environment Setup

### Option 1: Using Conda

```bash
conda create -n clickbait_env python=3.10 -y
conda activate clickbait_env
```

### Option 2: Using venv

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

This project uses `pyproject.toml`, so install the project in editable mode:

```bash
python -m pip install -e .
```

If you also want to install development tools such as `pytest`, `black`, and `isort`, run:

```bash
python -m pip install -e ".[dev]"
```

To check whether the package is installed correctly:

```bash
python -c "import clickbait_detector; print('Import successfully')"
```

## 5. Dataset Format

Place the dataset files inside:

```text
data/raw/
```

Example files (you can download it in .\notebooks\download_dataset.ipynb):

```text
data/raw/train_clickbait.csv
data/raw/val_clickbait.csv
```

The CSV files should have this format:

```csv
title,final_label
"Bạn sẽ không tin điều gì đã xảy ra sau khi cô gái mở chiếc hộp bí ẩn này",1
"Chính phủ công bố kế hoạch nâng cấp hệ thống giao thông đô thị trong năm 2026",0
```

Required columns:

```text
title: the news headline text
final_label: the binary label
```

Label meaning:

```text
1 = clickbait
0 = non-clickbait
```

## 6. Train the Model

Move to the project root directory:

```bash
cd .\PhoBert-VietNamClickBait-Recognize
```

Run training:

```bash
python train.py \
    --train_path ./data/raw/train_clickbait.csv \
    --val_path ./data/raw/val_clickbait.csv \
    --test_path ./data/raw/test_clickbait.csv \
    --save_path ./artifacts \
    --batch_size 8 \
    --lr 5e-4 \
    --epochs 1 \
    --max_len 50
```

After training, model checkpoints are saved in:

```text
artifacts/models/
```

Training history is saved in:

```text
artifacts/reports/
```

Expected output files:

```text
artifacts/models/best.pth
artifacts/models/last.pth
artifacts/reports/history.json
```

File meaning:

```text
best.pth      the best model checkpoint based on validation performance
last.pth      the checkpoint from the final epoch
history.json  training history including train_loss, val_loss, train_acc, and val_acc
```

## 7. Run Inference

After training, run inference with the saved model checkpoint.

Example with a clickbait headline:

```powershell
python -m clickbait_detector.inference \
    --weight_path "./artifacts/models/best.pth" \
    --input_sentence "Bạn sẽ không tin điều gì đã xảy ra sau khi cô gái mở chiếc hộp bí ẩn này" \
    --threshold 0.5 \
    --max_len 50
```

Example with a non-clickbait headline:

```powershell
python -m clickbait_detector.inference \
    --weight_path "./artifacts/models/best.pth" \
    --input_sentence "Chính phủ công bố kế hoạch nâng cấp hệ thống giao thông đô thị trong năm 2026" \
    --threshold 0.5 \
    --max_len 50
```

Important note:

```text
If the input sentence contains spaces, wrap it in double quotes.
```

Example output:

```text
Sentence: Bạn sẽ không tin điều gì đã xảy ra sau khi cô gái mở chiếc hộp bí ẩn này
Prediction: clickbait
Score: 0.8732
```

## 8. Python Imports

After installing the project with:

```bash
python -m pip install -e .
```

you can import modules like this:

```python
from clickbait_detector import ClickBaitDataset, Model, train, create_dataloader
```

Avoid importing like this:

```python
from src.clickbait_detector import Model
```

The correct package name is:

```python
clickbait_detector
```

The `src/` directory is only the source-code container.



