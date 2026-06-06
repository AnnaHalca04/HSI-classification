# Hyperspectral Image Classification

A deep learning project for hyperspectral image (HSI) classification using a 3D Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model classifies land cover types from hyperspectral patches across 5 classes.

---

## Dataset

- **Source:** Kaggle — HSI Classification competition
- **Input format:** `.npy` patch files (19×19×48)
- **Classes:** 5 land cover types (original classes 4 and 5 excluded due to data issues)
- **Output:** `submission.csv` with predicted labels

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Deep Learning | TensorFlow / Keras |
| Model | 3D CNN |
| Libraries | NumPy, pandas, scikit-learn |
| Environment | Kaggle Notebooks |

---

## Model Architecture

3D CNN designed for volumetric hyperspectral patches:

```
Input (19×19×48×1)
→ Conv3D(32, 3×3×7) + BatchNorm + MaxPool3D + Dropout(0.25)
→ Conv3D(64, 3×3×5) + BatchNorm + MaxPool3D + Dropout(0.25)
→ Conv3D(128, 3×3×3) + BatchNorm
→ Flatten
→ Dense(128) + Dropout(0.5)
→ Dense(5, softmax)
```

**Optimizer:** Adam (lr=0.0005)  
**Loss:** Categorical Crossentropy  
**Epochs:** 5 | **Batch size:** 32

---

## Key Design Decisions

- **3D convolutions** — chosen to exploit both spatial (19×19) and spectral (48 bands) dimensions simultaneously
- **Class weights** — applied to handle class imbalance in the training data
- **Classes 4 and 5 excluded** — due to missing or inconsistent data in those categories
- **Label remapping** — original labels {1,2,3,6,7} remapped to {0,1,2,3,4} for one-hot encoding

---

## Results

| Metric | Value |
|---|---|
| Macro F1 (validation) | 0.6080 |

Performance is constrained by the limited number of training epochs (5) and class imbalance. Further tuning with data augmentation and more epochs would likely improve results significantly.

---

## Project Structure

```
├── hsi_classification.py   # Full training + inference pipeline
├── README.md
```

---

## How to Run

This project is designed to run on Kaggle with the HSI classification dataset:

1. Upload the script to a Kaggle notebook
2. Connect the HSI classification dataset
3. Run — `submission.csv` will be generated automatically

---

## Project Status

Academic project — developed as part of the Machine Learning course at Politehnica University of Bucharest (2025–2026).
