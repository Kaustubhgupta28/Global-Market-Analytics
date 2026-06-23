import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Data/processed_data.csv")

# ==========================================
# Create Target
# ==========================================

df["Tomorrow_Close"] = (
    df.groupby("Index")["Close"]
      .shift(-1)
)

df["Target"] = (
    df["Tomorrow_Close"] > df["Close"]
).astype(int)

# ==========================================
# Remove Missing Values
# ==========================================

df = df.dropna()

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
    "MA50",
    "Volatility"
]

X = df[features]

y = df["Target"]

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
# Logistic Regression
# ==========================================

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print(
    "Accuracy:",
    accuracy_score(y_test, lr_pred)
)

print(
    classification_report(
        y_test,
        lr_pred
    )
)

# ==========================================
# Random Forest
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n")
print("=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(
    "Accuracy:",
    accuracy_score(y_test, rf_pred)
)

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
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

print("\nDirection Prediction Completed")

import joblib

joblib.dump(
    rf,
    "models/direction_model.pkl"
)

print("\nDirection Model Saved Successfully")

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA20",
    "MA50",
    "Volatility"
]

joblib.dump(
    features,
    "models/features.pkl"
)

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA20",
    "MA50",
    "Volatility"
]

joblib.dump(
    features,
    "models/features.pkl"
)

print("Features Saved Successfully")

