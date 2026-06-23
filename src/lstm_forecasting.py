import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Data/processed_data.csv")

# NSEI only

df = df[
    df["Index"] == "NSEI"
]

df = df.sort_values(
    by="Date"
)

data = df["Close"].values.reshape(-1,1)

# ==========================================
# Scaling
# ==========================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    data
)

# ==========================================
# Sequence Creation
# ==========================================

X = []
y = []

for i in range(60, len(scaled_data)):

    X.append(
        scaled_data[i-60:i,0]
    )

    y.append(
        scaled_data[i,0]
    )

X = np.array(X)
y = np.array(y)

X = np.reshape(
    X,
    (
        X.shape[0],
        X.shape[1],
        1
    )
)

# ==========================================
# Model
# ==========================================

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(
            X.shape[1],
            1
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(50)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(1)
)

model.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================================
# Training
# ==========================================

model.fit(
    X,
    y,
    epochs=5,
    batch_size=32
)

print("\nLSTM Training Completed")

model.save(
    "models/lstm_model.keras"
)

print(
    "\nLSTM Model Saved Successfully"
)

last_sequence = X[-1]

future_predictions = []

for i in range(30):
    pred = model.predict(
        last_sequence.reshape(1, 60, 1),
        verbose=0
    )[0][0]

    future_predictions.append(pred)

    last_sequence = np.append(
        last_sequence[1:],
        pred
    )
forecast_df = pd.DataFrame({
    "Forecast": future_predictions
})

forecast_df.to_csv(
    "outputs/forecast_results.csv",
    index=False
)

print("Forecast Results Saved")