# Potato Disease Classification

Welcome to the Potato Disease Classification project! This repository contains a Convolutional Neural Network (CNN) model trained to classify potato diseases into three main categories: Early blight, Late blight, and Healthy.

## Streamlit App

You can view the Streamlit app for potato disease classification [here](https://potato-disease-classification-cnn.streamlit.app/).

## Overview

Potato is one of the most important food crops worldwide. However, it is susceptible to various diseases that can significantly impact yield and quality. Early detection and classification of these diseases are crucial for timely intervention and effective management.

## Dataset

The model was trained using a dataset of potato images containing examples of each disease category as well as healthy potato plants. The dataset was carefully curated and labeled to ensure accurate training of the model.

## Model Architecture

The Convolutional Neural Network (CNN) model used for potato disease classification consists of the following layers:

1. **Convolutional Layers**: The model starts with a sequence of convolutional layers, each followed by a rectified linear unit (ReLU) activation function. These layers are responsible for detecting various features in the input images. The first convolutional layer has 32 filters with a kernel size of (3, 3), and subsequent layers increase the number of filters to capture more complex patterns.

2. **Dropout Layers**: Dropout layers are added after each convolutional layer to reduce overfitting by randomly dropping a fraction of the input units during training. A dropout rate of 0.5 is applied in each layer.

3. **Max Pooling Layers**: Max pooling layers are interspersed between the convolutional layers to downsample the feature maps, retaining the most important information while reducing the spatial dimensions. Each max pooling layer has a pooling size of (2, 2).

4. **Flatten Layer**: After the convolutional layers, the feature maps are flattened into a one-dimensional array using a flatten layer. This prepares the data for input into the fully connected layers.

5. **Fully Connected Layers**: The flattened features are passed through fully connected dense layers. The first dense layer consists of 64 neurons with a ReLU activation function, which helps introduce non-linearity into the model. The final dense layer has 3 neurons corresponding to the output classes (Early blight, Late blight, and Healthy) with a softmax activation function, which outputs probabilities for each class.

6. **Compilation**: The model is compiled using the Adam optimizer, which is an efficient variant of stochastic gradient descent. The loss function used is sparse categorical cross-entropy, suitable for multi-class classification tasks with integer labels. The model's performance is evaluated during training using the accuracy metric.

This model architecture aims to effectively capture and classify features from input potato images, enabling accurate detection of various diseases.

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

