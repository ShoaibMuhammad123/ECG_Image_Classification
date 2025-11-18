# ECG Image Classification using Hybrid CNN-LSTM Model

## Project Overview

This project implements a Deep Learning model to classify Electrocardiogram (ECG) images into distinct clinical categories. The architecture utilizes a Hybrid CNN-LSTM approach, combining Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for capturing sequential dependencies within the feature maps of the ECG signals.

## 📊 Dataset Details

The model is trained on an image-based dataset of ECGs, divided into training and testing sets. The images are resized to 224x224 pixels and normalized during preprocessing.

The 4 Classification Categories:

Myocardial Infarction (MI): ECG images of patients experiencing a heart attack.

History of MI: ECG images of patients with a past history of heart attacks.

Abnormal Heartbeat: ECG images indicating irregular heart rhythms.

Normal: ECG images of healthy individuals.

### 🛠️ Tech Stack & Prerequisites

The project is built using Python and the PyTorch framework.

Language: Python 

Deep Learning Framework: PyTorch (torch, torchvision)

Data Manipulation: NumPy, Pandas

Visualization: Matplotlib

Metrics: Scikit-learn (Confusion Matrix)

Environment: Google Colab (GPU recommended)

## Model Architecture: CNN-LSTM

The model employs a hybrid architecture designed to leverage the strengths of both CNNs and LSTMs:

CNN Layers (Feature Extraction):

Two Convolutional layers (Conv2d) to detect patterns (edges, curves, peaks) in the ECG images.

Activation function: ReLU.

Dimensionality reduction using MaxPool2d.

LSTM Layer (Temporal Analysis):

An LSTM layer processes the flattened features to capture sequential dependencies across the image segments.

Fully Connected Layer:

The output of the last LSTM timestep is passed through a Linear layer to predict one of the 4 classes.

🚀 Step-by-Step Implementation Guide

### Model Training
Resizing the image, and converting image in the form of tensors and Normalize each image (Mean=[0.5], Std=[0.5])

Observed Training Performance: The loss consistently decreased from ~1.30 to ~0.03 over 20 epochs.

#### Evaluation

The model is evaluated on both the Training and Test datasets.

Metrics Used: Accuracy Score and Confusion Matrix.

Result:   100% Accuracy on the test set.


### Inference (Single Image Prediction)

The workflow includes a script to pass a single image from the dataloader into the model to generate a specific prediction.


📈 Results Snapshot

Training Loss: Decreased to 0.0352.

Train Accuracy: 100.00%

Test Accuracy: 100.00%
