# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# --- Load saved model and scaler ---
MODEL_DIR = "models"
model_path_pkl = os.path.join(MODEL_DIR, "best_model.pkl")
model_path_h5 = os.path.join(MODEL_DIR, "best_model.h5")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

scaler = joblib.load(scaler_path)

# Try loading model (either sklearn model or keras model)
model = None
model_type = None

if os.path.exists(model_path_pkl):
    model = joblib.load(model_path_pkl)
    model_type = "sklearn"
elif os.path.exists(model_path_h5):
    model = load_model(model_path_h5)
    model_type = "keras"
else:
    st.error("No model found! Please train a model first.")
    st.stop()

# --- Streamlit App UI ---
st.title("African Food Store - Anomaly Detection App 🛒🇬🇧")
st.write("Upload your e-commerce transaction data and detect anomalies!")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    try:
        # Preprocessing
        drop_cols = ['transaction_id', 'timestamp']
        for col in drop_cols:
            if col in data.columns:
                data = data.drop(col, axis=1)

        cat_cols = ['product_name', 'category', 'payment_method', 'customer_region']
        for col in cat_cols:
            if col in data.columns:
                data[col] = data[col].astype('category').cat.codes

        X_input = scaler.transform(data)

        # Predict
        if model_type == "sklearn":
            preds = model.predict(X_input)
            preds = [1 if x == -1 else 0 for x in preds]
        else:  # keras model (autoencoder)
            X_recon = model.predict(X_input)
            recon_error = np.mean(np.square(X_input - X_recon), axis=1)
            threshold = np.percentile(recon_error, 95)
            preds = (recon_error > threshold).astype(int)

        # Add predictions to DataFrame
        result_df = pd.DataFrame()
        result_df["is_anomaly"] = preds
        final_df = pd.concat([data.reset_index(drop=True), result_df], axis=1)

        st.success(f"Detection completed! Found {sum(preds)} anomalies.")
        st.dataframe(final_df)

        # Download results
        csv = final_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error while processing: {e}")
