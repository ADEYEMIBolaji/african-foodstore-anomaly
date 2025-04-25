import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Sample categories and products
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


def generate_transaction(idx, inject_anomaly=False):
    product = random.choice(list(products.keys()))
    category = products[product]
    quantity = np.random.randint(1, 10)
    price = round(np.random.uniform(1.5, 25.0), 2)

    if inject_anomaly:
        quantity *= random.choice([20, 50, 100])  # unrealistic bulk
        price *= random.choice([0.1, 10])  # too low or too high
        payment = random.choice(payment_methods)
    else:
        payment = random.choices(payment_methods, weights=[0.4, 0.3, 0.3])[0]

    total = round(quantity * price, 2)
    region = random.choice(regions)
    timestamp = datetime.now() - timedelta(days=random.randint(0, 60), minutes=random.randint(0, 1440))

    return {
        "transaction_id": idx,
        "product_name": product,
        "category": category,
        "quantity": quantity,
        "price_per_unit": price,
        "total_amount": total,
        "payment_method": payment,
        "customer_region": region,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "is_anomaly": int(inject_anomaly)
    }


def create_dataset(n_normal=950, n_anomalies=50, output_path="data/ecommerce_data.csv"):
    data = [generate_transaction(i) for i in range(n_normal)]
    data += [generate_transaction(i + n_normal, inject_anomaly=True) for i in range(n_anomalies)]
    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    create_dataset()
