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

<image src="workflow/pipeline.jpg" alt="Result" width="600"/>


## 2. Project Structure

```text
clickbait_detect_proj/
│
├── data/
│   ├── processed/
│   │   └── combined_dataset.csv
│   └── raw/
│       ├── train_clickbait.csv
│       ├── test_clickbait.csv
│       ├── clickbait_dataset_vietnamese.csv
│       └── val_clickbait.csv
│
├── artifacts/
│   ├── models/
│   │   ├── best.pth
│   │   └── last.pth
│   │
│   └── reports/
│       ├── history.json
│       ├── precision_recall.png
│       ├── f1_roc.png
│       └── loss_acc.png
│
├── notebooks/
│   ├── download_dataset.ipynb
│   ├── eda.ipynb
│   └── process_imbalance.ipynb
│
├── src/
│   └── clickbait_detector/
│       ├── __init__.py
│       ├── clickbait_dataset.py
│       ├── preprocessing.py
│       ├── net.py
│       ├── utils.py
│       ├── crawl_data.py
│       └── inference.py
│
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
data/processed/
```

The CSV files should have this format:

```csv
title,label
"Bạn sẽ không tin điều gì đã xảy ra sau khi cô gái mở chiếc hộp bí ẩn này",1
"Chính phủ công bố kế hoạch nâng cấp hệ thống giao thông đô thị trong năm 2026",0
```

Required columns:

```text
title: the news headline text
label: the binary label
```

Label meaning:

```text
1 = clickbait
0 = non-clickbait
```

In this project, the model is trained using a combination of the article title, lead paragraph, and body content. To construct the training dataset, we merged four available data files: `test_clickbait.csv`, `train_clickbait.csv`, `val_clickbait.csv`, and `clickbait_dataset_vietnamese.csv`. In addition, we collected more Vietnamese news articles through web crawling and manually labeled them as clickbait or non-clickbait. This additional data was used to increase the dataset size and reduce the class imbalance problem, helping the model learn more effectively from both clickbait and non-clickbait samples.

You can download `test_clickbait.csv`, `train_clickbait.csv`, `val_clickbait.csv` in download_dataset notebook, `clickbait_dataset_vietnamese.csv` in kaggle, and combine it in process_imbalance notebook. If you can't do this, you also download it at https://drive.google.com/drive/folders/1DukjjtxNZPUfpRvd4f12ueQrDdyBf4qj?usp=sharing, I did it for you.


## 6. Train the Model

Move to the project root directory:

```bash
cd .\PhoBert-VietNamClickBait-Recognize
```

Arguments:

| Argument | Value | Description |
|---|---:|---|
| `--root_dir` | `"./data/processed/combined_v2.csv"` | Path to the processed dataset file used for training, validation, and testing |
| `--save_path` | `"./artifacts"` | Directory used to save checkpoints, model weights, logs, or other training artifacts |
| `--batch_size` | `16` | Number of samples processed in each training batch |
| `--backbone_lr` | `5e-6` | Learning rate used for fine-tuning the PhoBERT backbone |
| `--classify_lr` | `2e-5` | Learning rate used for the classifier layers |
| `--epochs` | `30` | Maximum number of training epochs |
| `--max_len` | `256` | Maximum token length for each input sequence after tokenization |
| `--patience` | `0` | Early stopping is disabled; the model is trained for the full number of epochs |
| `--dropout` | `0.3` | Dropout rate used to reduce overfitting in the classifier |

In this experiment, the PhoBERT backbone is fine-tuned with a smaller learning rate (`5e-6`), while the classifier layers use a larger learning rate (`2e-5`). The maximum sequence length is set to `256`, and early stopping is disabled by setting `patience = 0`.

Run training:

```bash
python train.py \
    --root_dir "./data/processed/combined_dataset.csv" \
    --save_path "./artifacts" \
    --batch_size 16 \
    --backbone_lr 5e-6 \
    --classify_lr 2e-5 \
    --epochs 30 \
    --max_len 256 \
    --patience 0 \
    --dropout 0.3
```

After training, model checkpoints are saved in:

```text
artifacts/models/
```

Training history and figures are saved in:

```text
artifacts/reports/
```

Expected output files:

```text
artifacts/models/best.pth
artifacts/models/last.pth
artifacts/reports/history.json
artifacts/reports/loss_acc.png
artifacts/reports/f1_roc.png
artifacts/reports/precision_recall.png
```

File meaning:

```text
best.pth      the best model checkpoint based on validation performance
last.pth      the checkpoint from the final epoch
history.json  training history including train_loss, val_loss, train_acc, and val_acc
precision_recall.png, f1_roc.png, loss_acc.png  plot metrics during training progress
```

## 7. Run Inference

After training, run inference with the saved model checkpoint. Training might take quite a long time, and if you can't, you can download it here https://drive.google.com/drive/folders/1H1pNDpdn2ikGbwLrdz4lY6ie6qADp4zZ?usp=sharing. I did it for you.

Example with a clickbait headline:

```powershell
python -m clickbait_detector.inference \
    --weight_path "./artifacts/models/best.pth" \
    --input_sentence "Bạn sẽ không tin điều gì đã xảy ra sau khi cô gái mở chiếc hộp bí ẩn này" \
    --threshold 0.5 \
    --max_len 256
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
Score: 0.743604838848114

Sentence: Chính phủ công bố kế hoạch nâng cấp hệ thống giao thông đô thị trong năm 2026
Prediction: non-clickbait
Score: 0.15985092520713806
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

## 9. Results

Loss and Accuracy figure:
<image src="artifacts/reports/loss_acc.png" alt="Result" width="600"/>

Precision and Recall figure:
<image src="artifacts/reports/precision_recall.png" alt="Result" width="600"/>

F1 and Roc curve at best epoch figure:
<image src="artifacts/reports/f1_roc.png" alt="Result" width="600"/>




