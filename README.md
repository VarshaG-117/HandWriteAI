# HandWriteAI

A full-stack AI web application for recognizing handwritten English letters A-Z from a drawing canvas, uploaded images, and PDFs.

## Features

- Flask API backend with `/predict`, `/upload-image`, and `/upload-pdf`
- TensorFlow/Keras CNN architecture for EMNIST Letters
- Canvas drawing with mouse and touch input
- Image upload for `jpg`, `jpeg`, and `png`
- PDF upload with connected-component character extraction
- Prediction history and top-3 confidence scores
- Responsive glassmorphism SaaS-style interface

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train The Model

Training downloads EMNIST Letters through `tensorflow_datasets` and saves `model.keras`.

```powershell
python train_model.py
```

The CNN uses augmentation, checkpointing, early stopping, and learning-rate reduction. With enough epochs and the full EMNIST Letters dataset, this architecture is intended to target 95%+ validation accuracy.

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

If `model.keras` is missing, the UI will still load but prediction endpoints return a clear model-not-ready error. Train the model first for real predictions.
