import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Data/processed_data.csv")

df = df.dropna()

# ==========================================
# Features
# ==========================================

features = [
    "Daily_Return",
    "Volatility"
]

X = df[features]

# ==========================================
# Isolation Forest
# ==========================================

model = IsolationForest(
    contamination=0.01,
    random_state=42
)

df["Anomaly"] = model.fit_predict(X)

# -1 = Anomaly
#  1 = Normal

anomalies = df[df["Anomaly"] == -1]

print("=" * 50)
print("ANOMALY DETECTION")
print("=" * 50)

print(
    "Total Anomalies:",
    len(anomalies)
)

print(
    anomalies[
        [
            "Index",
            "Date",
            "Close",
            "Daily_Return",
            "Volatility"
        ]
    ].head(20)
)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(12,6))

plt.scatter(
    df.index,
    df["Daily_Return"],
    alpha=0.5,
    label="Normal"
)

plt.scatter(
    anomalies.index,
    anomalies["Daily_Return"],
    label="Anomaly"
)

plt.title("Market Anomaly Detection")
plt.xlabel("Data Points")
plt.ylabel("Daily Return")

plt.legend()
plt.grid()

plt.show()

print("\nAnomaly Detection Completed")

import joblib

joblib.dump(
    model,
    "models/anomaly_model.pkl"
)

print("\nAnomaly Model Saved Successfully")

anomalies = df[
    df["Anomaly"] == -1
]

anomalies.to_csv(
    "outputs/anomalies.csv",
    index=False
)

print(
    f"Total Anomalies: {len(anomalies)}"
)

print(
    "Anomalies Saved Successfully"
)