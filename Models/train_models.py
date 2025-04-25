# models/train_models.py

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load preprocessed data
X = np.load("../data/X_scaled.npy")
y = np.load("../data/y_labels.npy")

# Train Isolation Forest
iso_model = IsolationForest(contamination=0.05, random_state=42)
iso_preds = iso_model.fit_predict(X)
iso_preds = [1 if x == -1 else 0 for x in iso_preds]
iso_f1 = f1_score(y, iso_preds)
print("Isolation Forest F1 Score:", iso_f1)

# Train Autoencoder
input_dim = X.shape[1]
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=30, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)

X_pred = autoencoder.predict(X)
recon_error = np.mean(np.square(X - X_pred), axis=1)
threshold = np.percentile(recon_error, 95)
auto_preds = (recon_error > threshold).astype(int)
auto_f1 = f1_score(y, auto_preds)
print("AutoEncoder F1 Score:", auto_f1)

# Train One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
svm_preds = svm_model.fit_predict(X)
svm_preds = [1 if x == -1 else 0 for x in svm_preds]
svm_f1 = f1_score(y, svm_preds)
print("One-Class SVM F1 Score:", svm_f1)

# Compare and Save Best Model
best_model = None
best_model_name = None

if iso_f1 >= auto_f1 and iso_f1 >= svm_f1:
    best_model = iso_model
    best_model_name = "Isolation Forest"
    joblib.dump(best_model, "../models/best_model.pkl")
elif auto_f1 >= iso_f1 and auto_f1 >= svm_f1:
    best_model = autoencoder
    best_model_name = "Autoencoder"
    best_model.save("../models/best_model.h5")
else:
    best_model = svm_model
    best_model_name = "One-Class SVM"
    joblib.dump(best_model, "../models/best_model.pkl")

print(f"✅ Best model ({best_model_name}) saved successfully!")

# Optional: Save scaler if needed for app later
# (In our preprocessing, scaler is very important for consistent prediction)
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "../models/scaler.pkl")
print("Scaler saved successfully!")
