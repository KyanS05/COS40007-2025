import pandas as pd
import matplotlib.pyplot as plt

file_path = "Portfolio 1/Folds5x2_pp.csv"
df = pd.read_csv(file_path)

# Define class labels using quantiles
df["PE_category"], bins = pd.qcut(df["PE"], q=4, labels=[1, 2, 3, 4], retbins=True)
df = df.drop(columns=["PE"]) # Drop original continuous target column

df.to_csv("Portfolio 1/converted_powerplant.csv", index=False)

# Plot distribution of classes
plt.figure(figsize=(8, 5))
df["PE_category"].value_counts().sort_index().plot(kind="bar", edgecolor="black")
plt.xlabel("PE Category")
plt.ylabel("Sample Count")
plt.title("Distribution of PE Categories (Quantile Binned)")
plt.xticks(ticks=[0, 1, 2, 3], labels=["Low", "Mid-Low", "Mid-High", "High"], rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

print("✅ Converted dataset saved as converted_powerplant.csv")