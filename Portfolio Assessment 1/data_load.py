import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the cleaned dataset
file_path = "Folds5x2_pp.csv"
df = pd.read_csv(file_path)

# Identify Target Variable & Predictors
target_variable = "PE"  # Power Output
predictors = ["AT", "V", "AP", "RH"]

# Univariate Analysis (Feature Distributions)
plt.figure(figsize=(10,6))
df.hist(figsize=(10,6), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplots for Outlier Detection
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Boxplots of Features (Outlier Detection)", fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Multivariate Analysis (Feature Relationships)
# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot Analysis (Feature Interactions)
sns.pairplot(df)
plt.suptitle("Pairwise Feature Relationships", fontsize=14)
plt.show()

# Generate Summary Statistics
eda_summary = {
    "Rows": df.shape[0],
    "Columns": df.shape[1],
    "Missing Values": df.isnull().sum().sum(),
    "Feature Correlations": df.corr().to_dict()
}

print("\nSummary of EDA:")
print(eda_summary)
