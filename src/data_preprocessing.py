import pandas as pd

df = pd.read_csv("Data/indexData.csv")

print("=" * 50)
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\n" + "="*50)
print("Dataset Info")
print(df.info())

print("\n" + "="*50)
print("Missing Values")
print(df.isnull().sum())

print("\n" + "="*50)
print("Number of Unique Indices")
print(df["Index"].nunique())

print("\nIndices Present:")
print(df["Index"].unique())

print("\nDate Range:")
print(df["Date"].min())
print(df["Date"].max())

print("\n" + "="*50)
print("Missing Values")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nMissing Values by Column:")
print(df.isnull().sum())

missing_rows = df[df.isnull().any(axis=1)]

print("Rows with Missing Values:")
print(missing_rows.head())

print("\nTotal Rows with Missing Values:")
print(len(missing_rows))

print("Before Cleaning:", df.shape)

df = df.dropna()

print("After Cleaning:", df.shape)
print("\nMissing Values by Column:")
print(df.isnull().sum())

# Date conversion
df["Date"] = pd.to_datetime(df["Date"])

# Daily Return %
df["Daily_Return"] = (
    (df["Close"] - df["Open"]) /
    df["Open"]
) * 100

print("\nDaily Return Created Successfully")

print(
    df[
        ["Index",
         "Date",
         "Open",
         "Close",
         "Daily_Return"]
    ].head()
)


# Moving Averages

df["MA20"] = (
    df.groupby("Index")["Close"]
      .transform(
          lambda x: x.rolling(20).mean()
      )
)

df["MA50"] = (
    df.groupby("Index")["Close"]
      .transform(
          lambda x: x.rolling(50).mean()
      )
)

print("\nMA20 and MA50 Created Successfully")

df["Volatility"] = (
    df.groupby("Index")["Daily_Return"]
      .transform(
          lambda x: x.rolling(20).std()
      )
)

print("\nVolatility Created Successfully")

df.to_csv(
    "Data/processed_data.csv",
    index=False
)

print("\nProcessed Dataset Saved!")

