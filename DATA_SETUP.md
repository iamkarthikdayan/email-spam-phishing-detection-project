# Data & Models Setup

This repository contains **code and documentation only**. Data and models are stored externally to keep the repository lightweight and push times fast.

## Setup Instructions for Evaluators

### Option 1: Use Existing Data/Models (if available)

If you have access to the data and models:

```powershell
# Place your datasets here
mkdir -p data/
# Copy your CSV files to data/

# Place your trained models here
mkdir -p models/
# Copy your model artifacts (*.pkl, *.json) to models/
```

### Option 2: Re-train from Scratch

Run the training pipeline:

```powershell
python train_bert.py
python main.py
```

This will generate:
- `data/combined_dataset.csv` — merged dataset
- `data/embeddings/` — BERT embeddings (cached)
- `models/clf_binary.pkl` — Stage 1 classifier
- `models/clf_multiclass.pkl` — Stage 2 classifier
- `models/metadata.json` — model metadata

## Directory Structure

```
email_spam_project/
├── *.py                    # Source code (in repo)
├── app/                    # Flask API (in repo)
├── utils/                  # Utilities (in repo)
├── templates/              # HTML templates (in repo)
├── README.md              # Documentation (in repo)
├── requirements*.txt      # Dependencies (in repo)
│
├── data/                  # ❌ NOT in repo (add locally)
│   ├── combined_dataset.csv
│   ├── embeddings/
│   └── ...
│
└── models/                # ❌ NOT in repo (add locally)
    ├── clf_binary.pkl
    ├── clf_multiclass.pkl
    └── metadata.json
```

## For GitHub: Code + Docs Only

The repository pushes fast because it contains:
✅ Python source files
✅ Configuration & documentation
✅ Flask app & templates
❌ No large datasets
❌ No model weights

## Notes

- If you're using **Git LFS** for external storage, you can store large files separately and reference them via `.gitattributes`
- For production deployment, consider cloud storage (S3, GCS) for datasets and model versioning
