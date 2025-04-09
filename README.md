# DeepFraud Neural Network

A simple, GPU-accelerated, from-scratch neural network built in Python using PyTorch and custom classes — designed to detect fraudulent transactions in the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 🔍 Overview

DeepFraud is a custom neural network implementation built without using high-level PyTorch modules like `nn.Linear` or `nn.Sequential`. Instead, it builds neural layers and neurons manually, providing an educational and transparent look into how neural networks work under the hood.

It supports GPU acceleration using CUDA for faster computation.

---

## 📁 Dataset

The model uses the **creditcard.csv** dataset from Kaggle:
- Highly imbalanced: Only ~0.17% of transactions are fraud
- Contains 31 columns: `Time`, `Amount`, and 28 anonymized features (`V1` to `V28`) + `Class` (target)

---

## ⚙️ Features

- Fully custom neural network: manual forward/backward passes
- Uses Sigmoid activation and binary cross-entropy loss
- Trains on CUDA if available
- Real-time loss tracking per epoch
- Cleanly structured with Neuron, NeuralLayer, and NeuralNetwork classes

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- pandas
- scikit-learn

You can install the required packages using:

```bash
pip install torch pandas scikit-learn
