# 🛒 African Foodstore Anomaly Detection App 🇬🇧🌍

This project builds a real-time anomaly detection system for a typical African Food Store's e-commerce data in the UK.

It covers everything from data generation, model training, to a live Streamlit web app where users can upload their transactions and detect anomalies.

---

## 🚀 Project Overview

**Features:**
- Generate synthetic African food e-commerce transactions
- Train and compare three anomaly detection models:
  - Isolation Forest
  - AutoEncoder (Deep Learning)
  - One-Class SVM
- Select and save the best performing model
- Deploy a Streamlit app for real-time anomaly detection
- Allow users to upload CSV files and download detected results

---

## 🏗 Project Structure

```bash
african-foodstore-anomaly/
├── app.py                  # Streamlit frontend
├── data/                    # Synthetic dataset and test data
│   ├── ecommerce_data.csv
│   ├── test_data.csv
│   └── generate_data.py
├── models/                  # Model training and saved models
│   ├── train_models.py
│   ├── best_model.pkl or best_model.h5
│   ├── scaler.pkl
├── notebooks/               # EDA and preprocessing notebooks
│   ├── eda_preprocessing.ipynb
│   └── model_training.ipynb
├── requirements.txt         # Project dependencies
├── README.md                 # Project documentation
└── .gitignore
