# Artificial-Intelligence
# 🏠 Predicting House Prices Using ANN Deep Learning

This repository contains a Deep Learning project that predicts median house values using an **Artificial Neural Network (ANN)**. The model is built using TensorFlow/Keras and evaluates performance across multiple metrics.

## 📋 Project Overview
The goal of this project is to build a regression model that estimates housing prices based on geographical and socioeconomic features (like location, income, and room counts). The code covers the entire pipeline, including data cleaning, categorical mapping, feature normalization, neural network architecture design, and performance visualization.

## 🛠️ Features & Technical Workflow
- **Data Preprocessing**: Handles missing values dynamically (`dropna`) and encodes categorical features (`ocean_proximity`) into numeric values.
- **Feature Scaling**: Implements `MinMaxScaler` to normalize the structural inputs, ensuring optimal gradient descent steps during training.
- **Deep Learning Architecture**: A fully connected Multilayer Perceptron (MLP) built with Keras `Sequential` API containing dense layers with ReLU activation and a linear output layer for regression.
- **Regularization & Callbacks**: Uses `EarlyStopping` to monitor validation loss and prevent overfitting.
- **Performance Evaluation**: Evaluates predictions using standard regression metrics ($R^2$ Score, MAE, MSE, MSLE) and plots loss trends across epochs.

## 🏗️ Neural Network Architecture
The network topology consists of a deep linear stack of layers:
- **Input Layer**: Derived from the dataset features.
- **Hidden Layer 1**: 1000 units, ReLU activation.
- **Hidden Layer 2**: 500 units, ReLU activation.
- **Hidden Layer 3**: 250 units, ReLU activation.
- **Output Layer**: 1 unit, Linear activation (Predicts continuous house value).

## 🚀 Getting Started

### Prerequisites
Make sure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kera

Dataset Structure

The model expects a housing.csv dataset containing the following features:

    longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity, and the target variable median_house_value.

ocean_proximity Encoding Map:

    <1H OCEAN ➡️ 0

    INLAND ➡️ 1

    NEAR OCEAN ➡️ 2

    NEAR BAY ➡️ 3

    ISLAND ➡️ 4


📊 Model Evaluation Metrics

The final script calculates and prints performance indicators to verify evaluation consistency:

    Mean Absolute Error (MAE)

    Mean Squared Error (MSE)

    Mean Squared Log Error (MSLE)

    R² Score (Coefficient of Determination)

The script also automatically saves and renders a loss curve mapping training_loss against validation loss.



