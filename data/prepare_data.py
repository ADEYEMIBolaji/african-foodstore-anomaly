# data/prepare_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the synthetic dataset
df = pd.read_csv("../data/ecommerce_data.csv")

# Preprocess
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_model = df.drop(['transaction_id', 'timestamp'], axis=1)

label_encoders = {}
for col in ['product_name', 'category', 'payment_method', 'customer_region']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model.drop('is_anomaly', axis=1)
y = df_model['is_anomaly']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save("../data/X_scaled.npy", X_scaled)
np.save("../data/y_labels.npy", y)
print("✅ Data preprocessing complete. Files saved!")
