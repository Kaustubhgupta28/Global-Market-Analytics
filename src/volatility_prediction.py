import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Data/processed_data.csv")

# ==========================================
# Remove Missing Values
# ==========================================

df = df.dropna()

# ==========================================
# Create Volatility Classes
# ==========================================

low = df["Volatility"].quantile(0.33)
high = df["Volatility"].quantile(0.66)

def classify_volatility(x):

    if x <= low:
        return 0

    elif x <= high:
        return 1

    else:
        return 2

df["Risk_Level"] = df["Volatility"].apply(
    classify_volatility
)

# ==========================================
# Features
# ==========================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA20",
    "MA50"
]

X = df[features]

y = df["Risk_Level"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

# ==========================================
# Results
# ==========================================

print("=" * 50)
print("VOLATILITY PREDICTION")
print("=" * 50)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

print(
    classification_report(
        y_test,
        pred
    )
)

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

print(importance)

print("\nVolatility Prediction Completed")

import joblib

joblib.dump(
    model,
    "models/volatility_model.pkl"
)

print("\nVolatility Model Saved Successfully")

