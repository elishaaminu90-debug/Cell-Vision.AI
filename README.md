# 🔬 Cell-Vision.AI (MalariaScan)

A deep learning app that detects malaria-infected vs uninfected blood cells 
from microscope images using a Convolutional Neural Network (CNN).

## Overview
Malaria diagnosis traditionally requires a trained technician to manually 
examine blood smears under a microscope. This project automates that 
process — upload a cell image and get an instant infected/uninfected 
prediction with confidence score.

## Model Performance
Achieves 94% overall accuracy on the validation set:

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Parasitized (Infected) | 0.97 | 0.91 | 0.94 |
| Uninfected | 0.92 | 0.97 | 0.94 |

Trained and validated on 27,558 real microscope cell images from the 
NIH malaria dataset.

## Model Architecture
A 3-layer Convolutional Neural Network:
- 3 Conv2D + MaxPooling blocks (16 → 32 → 64 filters)
- Dropout layers for regularization (0.2 - 0.5)
- Dense layer (64 units) before final sigmoid output
- 826,529 total trainable parameters

## Tech Stack
- Python
- TensorFlow / Keras (model training)
- Streamlit (web interface)
- OpenCV & PIL (image processing)
- Scikit-learn (evaluation metrics)

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit app interface
- `malaria_cnn_model.h5` — trained CNN model
- `MalariaScan.ipynb` — full training notebook with EDA, model building, 
and evaluation
- `requirements.txt` — project dependencies

## Dataset
[Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) 
— NIH National Library of Medicine

## Live Demo
[[Link coming soon](https://cell-visionai-9eapgqw9e3nkg4ezfwkxjp.streamlit.app/)]

## Disclaimer
This tool is for educational and demonstration purposes only. It should 
not be used as a substitute for professional medical diagnosis.
