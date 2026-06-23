import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Data/processed_data.csv")

# Date conversion
df["Date"] = pd.to_datetime(df["Date"])

# ==========================================
# Dataset Overview
# ==========================================

print("=" * 50)
print("DATASET SHAPE")
print(df.shape)

print("\nINDICES PRESENT")
print(df["Index"].unique())

print("\nTOTAL INDICES")
print(df["Index"].nunique())

# ==========================================
# Graph 1 : NIFTY Trend
# ==========================================

nifty = df[df["Index"] == "NSEI"]

plt.figure(figsize=(12,6))
plt.plot(nifty["Date"], nifty["Close"])

plt.title("NIFTY 50 Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid()

plt.show()

# ==========================================
# Graph 2 : NASDAQ Trend
# ==========================================

nasdaq = df[df["Index"] == "IXIC"]

plt.figure(figsize=(12,6))
plt.plot(nasdaq["Date"], nasdaq["Close"])

plt.title("NASDAQ Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid()

plt.show()

# ==========================================
# Graph 3 : NIFTY vs NASDAQ
# ==========================================

plt.figure(figsize=(14,6))

plt.plot(
    nifty["Date"],
    nifty["Close"],
    label="NIFTY"
)

plt.plot(
    nasdaq["Date"],
    nasdaq["Close"],
    label="NASDAQ"
)

plt.title("NIFTY vs NASDAQ")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.legend()
plt.grid()

plt.show()

# ==========================================
# Graph 4 : Daily Return Distribution
# ==========================================

plt.figure(figsize=(10,6))

df["Daily_Return"].hist(
    bins=50
)

plt.title("Daily Return Distribution")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")

plt.grid()

plt.show()

# ==========================================
# Graph 5 : Top Volatile Markets
# ==========================================

volatility = (
    df.groupby("Index")["Daily_Return"]
      .std()
      .sort_values(
          ascending=False
      )
)

print("\n")
print("=" * 50)
print("MARKET VOLATILITY")
print(volatility)

plt.figure(figsize=(10,6))

volatility.plot(
    kind="bar"
)

plt.title("Market Volatility Comparison")
plt.xlabel("Market Index")
plt.ylabel("Volatility")

plt.grid()

plt.show()

# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 50)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 50)