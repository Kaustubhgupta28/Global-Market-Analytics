import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Data/processed_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

print("=" * 50)
print("CORRELATION ANALYSIS")
print("=" * 50)

# ==========================================
# Create Pivot Table
# ==========================================

market_data = df.pivot_table(
    index="Date",
    columns="Index",
    values="Close"
)

print("\nMarket Data Shape:")
print(market_data.shape)

# ==========================================
# Correlation Matrix
# ==========================================

corr_matrix = market_data.corr()

print("\nCorrelation Matrix")
print(corr_matrix)

# ==========================================
# Heatmap
# ==========================================

plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Global Market Correlation Heatmap")

plt.tight_layout()
plt.show()

# ==========================================
# Top Correlated Markets
# ==========================================

correlation_pairs = (
    corr_matrix.unstack()
    .sort_values(ascending=False)
)

print("\nTop Correlations")
print(correlation_pairs.head(30))

# ==========================================
# NSEI vs NASDAQ
# ==========================================

if "NSEI" in market_data.columns and "IXIC" in market_data.columns:

    plt.figure(figsize=(12,6))

    plt.scatter(
        market_data["NSEI"],
        market_data["IXIC"],
        alpha=0.5
    )

    plt.title("NSEI vs NASDAQ Relationship")
    plt.xlabel("NSEI")
    plt.ylabel("NASDAQ")

    plt.grid()

    plt.show()

# ==========================================
# Average Correlation
# ==========================================

avg_corr = corr_matrix.mean().sort_values(
    ascending=False
)

print("\nAverage Correlation")
print(avg_corr)

# ==========================================
# Save Correlation Matrix
# ==========================================

corr_matrix.to_csv(
    "outputs/correlation_matrix.csv"
)

print("\nCorrelation Matrix Saved Successfully")

print("\nEDA Correlation Analysis Completed")