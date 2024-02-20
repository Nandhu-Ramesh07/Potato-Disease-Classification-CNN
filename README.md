# Potato Disease Classification

Welcome to the Potato Disease Classification project! This repository contains a Convolutional Neural Network (CNN) model trained to classify potato diseases into three main categories: Early blight, Late blight, and Healthy.

## Streamlit App

You can view the Streamlit app for potato disease classification [here](https://potato-disease-classification-cnn.streamlit.app/).

## Overview

Potato is one of the most important food crops worldwide. However, it is susceptible to various diseases that can significantly impact yield and quality. Early detection and classification of these diseases are crucial for timely intervention and effective management.

## Dataset

The model was trained using a dataset of potato images containing examples of each disease category as well as healthy potato plants. The dataset was carefully curated and labeled to ensure accurate training of the model.

## Model Architecture

The CNN model architecture consists of [describe your model architecture here]. The model was trained using transfer learning on a pre-trained network to leverage features learned from a large dataset and fine-tuned on the potato disease dataset.

## Usage

To use the model for classification, you can follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies specified in the `requirements.txt` file.
3. Load the trained model using your preferred deep learning framework (e.g., TensorFlow).
4. Preprocess the input images (e.g., resize, normalize) before feeding them into the model for inference.
5. Obtain predictions from the model and interpret the results for potato disease classification.

Example code snippet (Python):
```python
# Load the trained model
model = load_model('potatoes_CNN.h5')

# Obtain predictions
# predictions = model.predict(input_img)

