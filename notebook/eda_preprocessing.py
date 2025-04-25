# 📊 EDA + Preprocessing for African Food Store Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("../data/ecommerce_data.csv")
print("Dataset Shape:", df.shape)
df.head()

# Check for missing values
df.isnull().sum()

# Check data types
df.dtypes

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot category distribution
plt.figure(figsize=(8,4))
sns.countplot(x='category', data=df)
plt.title('Product Category Distribution')
plt.xticks(rotation=45)
plt.show()

# Check anomalies distribution
sns.countplot(x='is_anomaly', data=df)
plt.title('Normal vs Anomalous Transactions')
plt.show()

# Price distribution
plt.figure(figsize=(6,4))
sns.histplot(df['total_amount'], bins=30, kde=True)
plt.title('Total Amount Distribution')
plt.show()


# Drop transaction_id and timestamp for modeling
df_model = df.drop(['transaction_id', 'timestamp'], axis=1)

# Encode categorical variables
label_encoders = {}
for col in ['product_name', 'category', 'payment_method', 'customer_region']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Separate labels
X = df_model.drop('is_anomaly', axis=1)
y = df_model['is_anomaly']

# Normalize numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save for modeling
import numpy as np
np.save("../data/X_scaled.npy", X_scaled)
np.save("../data/y_labels.npy", y)
print("Preprocessed data saved!")
