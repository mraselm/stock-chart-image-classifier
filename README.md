# Candlestick Chart CNN Classifier

A Convolutional Neural Network (CNN) built using TensorFlow/Keras to classify candlestick chart images into stock price movement categories: **up** or **down**. The model is trained on RGB chart images and tuned for improved performance using different learning rates.

## Project Overview

This project is part of a deep learning course (ML4BI2). The objective is to create a robust image classification model that predicts stock direction based on candlestick charts. It includes multiple training stages, model tuning, and final evaluation on a test set.

## Dataset

The dataset consists of RGB candlestick images categorized into two classes:
- `up`: Images where the stock price moved upward
- `down`: Images where the stock price moved downward

### Dataset Structure (after unzipping)

```
candlestick_dataset/
│
├── train/
│   ├── up/
│   └── down/
│
├── validation/
│   ├── up/
│   └── down/
│
├── test/
    ├── up/
    └── down/
```

> Note: Please unzip the dataset archive `candlestick_dataset.zip` in the same directory as the notebook before running.

## How to Run

This project is designed to run locally on **Jupyter Notebook**.

1. Clone or download this repository
2. Ensure you have Python 3.8+ and the required packages (see below)
3. Unzip the dataset file
4. Open `Candlestick_Chart_CNN_Classifier.ipynb` in Jupyter Notebook
5. Run the cells in order

### Required Libraries

Install the necessary packages using:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

> Tested on TensorFlow 2.13+

## Model Building Stages

### 1. CNN from Scratch (RGB)
- Built a simple 3-layer CNN model trained on RGB candlestick images.
- Flattened layer followed by dense layers.
- Achieved test accuracy of approximately **46.4%**.

### 2. 2-Layer CNN Architecture
- Reduced depth to two convolutional layers to evaluate simplicity.
- Modest improvement in training speed with similar accuracy.

### 3. Learning Rate Tuning
- Fine-tuned learning rate to improve validation performance.
- Best model found with **learning rate = 0.001**.
- Final test accuracy reached **52.1%**.

> The final model is saved as `model_lr_0.001.keras`.

## Evaluation

- Model performance evaluated using test set accuracy and confusion matrix.
- The predictions are visualized alongside classification accuracy.
- Findings show moderate capability to distinguish patterns in candlestick charts.

## Files Included

- `Candlestick_Chart_CNN_Classifier.ipynb`: Main notebook with all training and evaluation steps
- `improved_cnn.keras`: Model from earlier architecture (optional)
- `model_lr_0.001.keras`: Best-performing trained model
- `candlestick_dataset.zip`: Compressed image dataset (up/down) for train, validation, and test

## Author

Rasel Mia  
MSc Student, Business Intelligence  
Aarhus University  
