import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Product setup
products = {
    "Yam": "Tubers",
    "Cassava": "Tubers",
    "Egusi": "Spices",
    "Suya Spice": "Spices",
    "Plantain": "Fruits",
    "Jollof Rice Pack": "Grains",
    "Palm Oil": "Oil",
    "Ogbono": "Seeds",
    "Stockfish": "Proteins",
    "Chin Chin": "Snacks"
}

payment_methods = ["Cash", "Mobile Money", "Card"]
regions = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"]


# Function to create a transaction
def generate_transaction(idx, anomaly=False):
    product = random.choice(list(products.keys()))
    category = products[product]
    quantity = np.random.randint(1, 10)
    price = round(np.random.uniform(2.0, 30.0), 2)

    if anomaly:
        quantity *= random.choice([50, 100])
        price *= random.choice([0.1, 10])

    total = round(quantity * price, 2)
    payment = random.choice(payment_methods)
    region = random.choice(regions)
    timestamp = datetime.now() - timedelta(days=random.randint(0, 10), minutes=random.randint(0, 1440))

    return {
        "transaction_id": idx,
        "product_name": product,
        "category": category,
        "quantity": quantity,
        "price_per_unit": price,
        "total_amount": total,
        "payment_method": payment,
        "customer_region": region,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }


# Generate normal + anomaly data
normal_data = [generate_transaction(i) for i in range(15)]
anomaly_data = [generate_transaction(i + 100, anomaly=True) for i in range(5)]

# Combine and shuffle
test_data = normal_data + anomaly_data
random.shuffle(test_data)

# Save
df_test = pd.DataFrame(test_data)
df_test.to_csv("../data/test_data.csv", index=False)
print("✅ Test data saved to data/test_data.csv")
